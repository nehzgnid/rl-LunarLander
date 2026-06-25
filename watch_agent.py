#!/usr/bin/env python3
from __future__ import annotations

import argparse

from agent_ppo.workflow.env_factory import BASE_ENV_MODE, ENV_MODES
from agent_ppo.workflow.watch_workflow import workflow


def parse_args():
    parser = argparse.ArgumentParser(description="Watch/evaluate the LunarLander PPO model.")
    parser.add_argument("--env-mode", type=str, default=BASE_ENV_MODE, choices=ENV_MODES, help="Environment mode to watch.")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--n-timesteps", type=int, default=None)
    parser.add_argument("--exp-id", type=int, default=None, help="Experiment id. Defaults to the latest saved run.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Model inference device.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    workflow(
        no_render=args.no_render,
        n_timesteps=args.n_timesteps,
        exp_id=args.exp_id,
        device=args.device,
        env_mode=args.env_mode,
    )
