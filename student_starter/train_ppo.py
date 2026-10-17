"""Train PPO from scratch on the 3D pixel-only MetaDrive sky-road arena."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from device import DEVICE_CHOICES, device_status, resolve_device
from model import CNNActorCritic
from parallel_env import ParallelRacingEnv


CUDA_TORCH_INSTALL = (
    r".\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall "
    r"torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124"
)

CLIP_EPS = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
HORIZON = 512
EPOCHS = 3
MINIBATCH = 256
NUM_ENVS = 4
ENTROPY_COEF = 0.005
VALUE_COEF = 0.25
MAX_GRAD_NORM = 1.0
SAVE_EVERY = 50_000
TARGET_KL = 0.03
# Per-module learning-rate multipliers. The encoder/critic produce enormous
# gradients (measured ~2500 raw norm) which, under global grad clipping,
# starve the tiny actor head (raw norm ~17) — the policy barely moved for
# 500k steps. Scaling the actor head up rebalances effective update sizes.
ACTOR_LR_MULTIPLIER = 8.0
CRITIC_LR_MULTIPLIER = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pixel-only PPO driver in MetaDrive.")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total environment steps.")
    parser.add_argument("--render", action="store_true", help="Open the Panda3D 3D window while training.")
    parser.add_argument("--lr", type=float, default=1.0e-4, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto", help="Training device.")
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS, help="Parallel MetaDrive workers.")
    parser.add_argument("--horizon", type=int, default=HORIZON, help="PPO rollout length per worker.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="PPO optimization epochs per rollout.")
    parser.add_argument("--minibatch", type=int, default=MINIBATCH, help="PPO minibatch size.")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY, help="Checkpoint interval in env steps.")
    parser.add_argument("--target-kl", type=float, default=TARGET_KL, help="Stop PPO epochs early above this KL.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/policy.pt", help="Checkpoint path.")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume a saved training checkpoint.")
    parser.add_argument("--logdir", type=str, default="runs/ppo_pixels", help="TensorBoard log directory.")
    return parser.parse_args()


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_value: float | np.ndarray,
    gamma: float = GAMMA,
    gae_lambda: float = GAE_LAMBDA,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros_like(rewards[0], dtype=np.float32)
    for step in reversed(range(len(rewards))):
        next_nonterminal = 1.0 - dones[step]
        next_value = last_value if step == len(rewards) - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_value * next_nonterminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae
    returns = advantages + values
    return advantages, returns


def checkpoint_path_for_step(base_path: Path, global_step: int) -> Path:
    suffix = base_path.suffix or ".pt"
    return base_path.with_name(f"{base_path.stem}_{global_step:06d}_steps{suffix}")


def save_checkpoint(
    model: CNNActorCritic,
    optimizer: optim.Optimizer,
    base_path: Path,
    global_step: int,
    update_idx: int,
    args: argparse.Namespace,
) -> Path:
    versioned_path = checkpoint_path_for_step(base_path, global_step)
    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "action_dim": model.action_dim,
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": {
            "global_step": global_step,
            "update_idx": update_idx,
            "map_name": "sky_chicane",
            "num_envs": args.num_envs,
            "horizon": args.horizon,
            "minibatch": args.minibatch,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
        },
    }
    for path in (versioned_path, base_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    return versioned_path


def load_training_checkpoint(
    path: Path,
    model: CNNActorCritic,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    metadata = checkpoint.get("metadata", {})
    return int(metadata.get("global_step", 0)), int(metadata.get("update_idx", 0))


def main() -> None:
    args = parse_args()
    if min(args.steps, args.horizon, args.minibatch, args.num_envs, args.save_every) <= 0:
        raise ValueError("steps, horizon, minibatch, num-envs, and save-every must all be positive")
    if args.steps % args.num_envs != 0:
        raise ValueError("--steps must be divisible by --num-envs so global-step accounting stays exact")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        device = resolve_device(args.device)
    except RuntimeError as exc:
        if args.device == "cuda":
            raise RuntimeError(
                f"{exc}\n\nFrom the student_starter folder, run:\n  {CUDA_TORCH_INSTALL}"
            ) from exc
        raise
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(device_status(device))
    model = CNNActorCritic().to(device)

    # Parameter groups: the actor head needs a much larger effective step than
    # the encoder/critic, whose gradients dominate the global clip norm.
    actor_params = list(model.actor.parameters())
    critic_params = list(model.critic.parameters()) + [model.log_std]
    encoder_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    optimizer = optim.Adam(
        [
            {"params": encoder_params, "lr": args.lr},
            {"params": actor_params, "lr": args.lr * ACTOR_LR_MULTIPLIER},
            {"params": critic_params, "lr": args.lr * CRITIC_LR_MULTIPLIER},
        ],
        eps=1e-5,
    )
    optimizer_lrs = {
        "encoder": args.lr,
        "actor": args.lr * ACTOR_LR_MULTIPLIER,
        "critic": args.lr * CRITIC_LR_MULTIPLIER,
    }
    print(f"optimizer groups (base lr={args.lr}): {optimizer_lrs}")
    writer = SummaryWriter(args.logdir)

    checkpoint_path = Path(args.checkpoint)
    global_step = 0
    update_idx = 0
    if args.resume_from:
        resume_path = Path(args.resume_from)
        global_step, update_idx = load_training_checkpoint(resume_path, model, optimizer, device)
        print(f"resumed checkpoint={resume_path} step={global_step} update={update_idx}")
    next_save_step = ((global_step // args.save_every) + 1) * args.save_every
    last_saved_step = -1

    env = ParallelRacingEnv(num_envs=args.num_envs, seed=args.seed, render=args.render)
    obs = env.observations
    ep_returns = np.zeros(args.num_envs, dtype=np.float64)
    ep_lengths = np.zeros(args.num_envs, dtype=np.int64)
    reason_counts: dict[str, int] = {}
    total_finished = 0
    start_time = time.time()
    print(
        f"workers={args.num_envs} horizon_per_worker={args.horizon} "
        f"rollout_capacity={args.num_envs * args.horizon}"
    )

    try:
        while global_step < args.steps:
            remaining_vector_steps = (args.steps - global_step) // args.num_envs
            rollout_steps = min(args.horizon, remaining_vector_steps)
            obs_buf = np.zeros((rollout_steps, args.num_envs, *env.observation_shape), dtype=np.uint8)
            action_buf = np.zeros((rollout_steps, args.num_envs, 2), dtype=np.float32)
            logprob_buf = np.zeros((rollout_steps, args.num_envs), dtype=np.float32)
            reward_buf = np.zeros((rollout_steps, args.num_envs), dtype=np.float32)
            done_buf = np.zeros((rollout_steps, args.num_envs), dtype=np.float32)
            value_buf = np.zeros((rollout_steps, args.num_envs), dtype=np.float32)

            for step in range(rollout_steps):
                obs_buf[step] = obs
                obs_t = torch.as_tensor(obs, device=device)
                with torch.no_grad():
                    bounded_action, log_prob, _entropy, value = model.act(obs_t, deterministic=False)

                actions = bounded_action.cpu().numpy().astype(np.float32)
                next_obs, rewards, terminated, truncated, infos = env.step(actions)
                dones = np.logical_or(terminated, truncated)
                action_buf[step] = actions
                logprob_buf[step] = log_prob.cpu().numpy()
                reward_buf[step] = rewards
                done_buf[step] = dones.astype(np.float32)
                value_buf[step] = value.cpu().numpy()

                ep_returns += rewards
                ep_lengths += 1
                global_step += args.num_envs
                obs = next_obs

                for worker_id in np.flatnonzero(dones):
                    terminal_reason = str(infos[worker_id].get("terminal_reason", "truncated"))
                    writer.add_scalar("episode_reward", ep_returns[worker_id], global_step)
                    writer.add_scalar("episode_length", ep_lengths[worker_id], global_step)
                    writer.add_scalar("episode_completion", float(infos[worker_id].get("route_completion", 0.0)), global_step)
                    writer.add_histogram("episode/terminal_reason", np.array([hash(terminal_reason) % 1000]), global_step)
                    reason_counts[terminal_reason] = reason_counts.get(terminal_reason, 0) + 1
                    total_finished += terminal_reason == "finish"
                    print(
                        f"step={global_step:8d} worker={worker_id} "
                        f"episode_return={ep_returns[worker_id]:8.2f} "
                        f"episode_length={ep_lengths[worker_id]:4d} "
                        f"completion={infos[worker_id].get('route_completion', 0.0):.3f} "
                        f"reason={terminal_reason}"
                    )
                    ep_returns[worker_id] = 0.0
                    ep_lengths[worker_id] = 0

            with torch.no_grad():
                _dist, bootstrap_values = model(torch.as_tensor(obs, device=device))
                last_values = bootstrap_values.cpu().numpy().astype(np.float32)
                last_values[done_buf[-1] > 0.5] = 0.0

            advantages, returns = compute_gae(reward_buf, done_buf, value_buf, last_values)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            batch_size = rollout_steps * args.num_envs
            b_obs = torch.as_tensor(obs_buf.reshape(batch_size, *env.observation_shape), device=device)
            b_actions = torch.as_tensor(action_buf.reshape(batch_size, 2), device=device)
            b_old_logprobs = torch.as_tensor(logprob_buf.reshape(batch_size), device=device)
            b_advantages = torch.as_tensor(advantages.reshape(batch_size), device=device)
            b_returns = torch.as_tensor(returns.reshape(batch_size), device=device)

            last_policy_loss = 0.0
            last_value_loss = 0.0
            last_entropy = 0.0
            last_approx_kl = 0.0
            indices = np.arange(batch_size)
            for _epoch in range(args.epochs):
                np.random.shuffle(indices)
                stop_update = False
                for start in range(0, batch_size, args.minibatch):
                    mb = indices[start : start + args.minibatch]
                    new_logprobs, entropy, values = model.evaluate_actions(b_obs[mb], b_actions[mb])
                    logratio = new_logprobs - b_old_logprobs[mb]
                    ratio = logratio.exp()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    unclipped = ratio * b_advantages[mb]
                    clipped = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * b_advantages[mb]
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = 0.5 * (b_returns[mb] - values).pow(2).mean()
                    entropy_loss = entropy.mean()

                    loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()

                    last_policy_loss = float(policy_loss.item())
                    last_value_loss = float(value_loss.item())
                    last_entropy = float(entropy_loss.item())
                    last_approx_kl = float(approx_kl.item())

                    if args.target_kl > 0 and last_approx_kl > args.target_kl:
                        stop_update = True
                        break
                if stop_update:
                    break

            update_idx += 1
            reasons = " ".join(f"{k}={v}" for k, v in sorted(reason_counts.items()))
            writer.add_scalar("episode/finished_total", total_finished, global_step)
            reason_counts.clear()
            writer.add_scalar("policy_loss", last_policy_loss, global_step)
            writer.add_scalar("value_loss", last_value_loss, global_step)
            writer.add_scalar("entropy", last_entropy, global_step)
            writer.add_scalar("approx_kl", last_approx_kl, global_step)
            writer.add_scalar("actions/mean_steer", float(action_buf[..., 0].mean()), global_step)
            writer.add_scalar("actions/mean_abs_steer", float(np.abs(action_buf[..., 0]).mean()), global_step)
            writer.add_scalar("actions/mean_throttle", float(action_buf[..., 1].mean()), global_step)
            writer.add_scalar("actions/mean_brake", float(np.maximum(-action_buf[..., 1], 0.0).mean()), global_step)

            elapsed = max(time.time() - start_time, 1e-6)
            print(
                f"update={update_idx:4d} step={global_step:8d} "
                f"policy_loss={last_policy_loss:8.4f} value_loss={last_value_loss:8.4f} "
                f"entropy={last_entropy:6.3f} kl={last_approx_kl:6.4f} "
                f"steer={action_buf[..., 0].mean():6.3f} abs_steer={np.abs(action_buf[..., 0]).mean():6.3f} "
                f"throttle={action_buf[..., 1].mean():6.3f} "
                f"brake={np.maximum(-action_buf[..., 1], 0.0).mean():6.3f} "
                f"fps={global_step / elapsed:6.1f} {reasons}"
            )

            if global_step >= next_save_step:
                while global_step >= next_save_step:
                    next_save_step += args.save_every
                versioned_path = save_checkpoint(
                    model, optimizer, checkpoint_path, global_step, update_idx, args
                )
                last_saved_step = global_step
                print(f"saved checkpoint -> {versioned_path} (latest -> {checkpoint_path})")

        if last_saved_step != global_step:
            versioned_path = save_checkpoint(model, optimizer, checkpoint_path, global_step, update_idx, args)
        else:
            versioned_path = checkpoint_path_for_step(checkpoint_path, global_step)
        print(f"final checkpoint -> {versioned_path} (latest -> {checkpoint_path})")
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
