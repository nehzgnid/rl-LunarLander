#!/usr/bin/env python3
from __future__ import annotations

import argparse

from agent_ppo.workflow.env_factory import BASE_ENV_MODE
from agent_ppo.workflow.train_workflow import workflow


algorithm_name_list = ["ppo"]
algorithm_name = "ppo"


def parse_args():
    parser = argparse.ArgumentParser(description="Train LunarLander PPO baseline.")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Training device.")
    return parser.parse_args()


if __name__ == "__main__":
    if algorithm_name not in algorithm_name_list:
        raise ValueError(f"Unsupported algorithm_name: {algorithm_name}")
    args = parse_args()
    workflow(
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        device=args.device,
        env_mode=BASE_ENV_MODE,
    )
