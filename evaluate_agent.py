#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Evaluate the LunarLander PPO baseline without rendering."""

import argparse

from agent_ppo.workflow.env_factory import BASE_ENV_MODE
from agent_ppo.workflow.evaluate_workflow import workflow


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the LunarLander PPO baseline.")
    parser.add_argument("--exp-id", type=int, default=None, help="Experiment id. Defaults to the latest saved run.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--best", action="store_true", help="Evaluate best_model.zip with best_vecnormalize.pkl.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Evaluation device.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    workflow(
        exp_id=args.exp_id,
        episodes=args.episodes,
        use_best=args.best,
        device=args.device,
        env_mode=BASE_ENV_MODE,
    )
