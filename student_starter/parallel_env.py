"""Small spawn-based vector environment for running MetaDrive in parallel."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import Connection
import traceback
from typing import Any

import numpy as np


def _worker_main(connection: Connection, worker_id: int, seed: int, render: bool) -> None:
    env = None
    episode_index = 0
    try:
        from env import RacingEnv

        env = RacingEnv(render=render, seed=seed)
        observation, info = env.reset(seed=seed % 100_000)
        connection.send(("ready", (observation, info)))

        while True:
            command, payload = connection.recv()
            if command == "step":
                observation, reward, terminated, truncated, info = env.step(payload)
                if render:
                    env.render()
                done = bool(terminated or truncated)
                if done:
                    episode_index += 1
                    terminal_info = info
                    observation, _reset_info = env.reset(seed=(seed + episode_index) % 100_000)
                    info = terminal_info
                connection.send(
                    (
                        "result",
                        (observation, float(reward), bool(terminated), bool(truncated), info),
                    )
                )
            elif command == "close":
                break
            else:
                raise ValueError(f"worker {worker_id} received unknown command: {command!r}")
    except (EOFError, KeyboardInterrupt):
        pass
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class ParallelRacingEnv:
    """Run one MetaDrive engine per spawned process and batch the results."""

    def __init__(self, num_envs: int, seed: int, render: bool = False) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")

        self.num_envs = num_envs
        self._closed = False
        context = mp.get_context("spawn")
        self._connections: list[Connection] = []
        self._processes: list[mp.Process] = []

        try:
            for worker_id in range(num_envs):
                parent_connection, child_connection = context.Pipe()
                worker_seed = (seed + worker_id * 10_000) % 100_000
                process = context.Process(
                    target=_worker_main,
                    args=(child_connection, worker_id, worker_seed, render and worker_id == 0),
                    name=f"metadrive-worker-{worker_id}",
                    daemon=True,
                )
                process.start()
                child_connection.close()
                self._connections.append(parent_connection)
                self._processes.append(process)

            initial = [self._receive(index, expected="ready") for index in range(num_envs)]
        except BaseException:
            self.close()
            raise

        self.observations = np.stack([item[0] for item in initial], axis=0)
        self.initial_infos = [item[1] for item in initial]
        self.observation_shape = self.observations.shape[1:]

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, 2):
            raise ValueError(f"expected actions shape {(self.num_envs, 2)}, got {actions.shape}")

        for connection, action in zip(self._connections, actions):
            connection.send(("step", action))
        results = [self._receive(index, expected="result") for index in range(self.num_envs)]

        observations = np.stack([item[0] for item in results], axis=0)
        rewards = np.asarray([item[1] for item in results], dtype=np.float32)
        terminated = np.asarray([item[2] for item in results], dtype=np.bool_)
        truncated = np.asarray([item[3] for item in results], dtype=np.bool_)
        infos = [item[4] for item in results]
        self.observations = observations
        return observations, rewards, terminated, truncated, infos

    def _receive(self, worker_id: int, expected: str) -> Any:
        try:
            status, payload = self._connections[worker_id].recv()
        except EOFError as exc:
            exit_code = self._processes[worker_id].exitcode
            raise RuntimeError(
                f"MetaDrive worker {worker_id} exited unexpectedly (exit code {exit_code})"
            ) from exc
        if status == "error":
            raise RuntimeError(f"MetaDrive worker {worker_id} failed:\n{payload}")
        if status != expected:
            raise RuntimeError(
                f"MetaDrive worker {worker_id} returned {status!r}; expected {expected!r}"
            )
        return payload

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        for connection in self._connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self._processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        for connection in self._connections:
            connection.close()

    def __enter__(self) -> "ParallelRacingEnv":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()
