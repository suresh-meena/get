#!/usr/bin/env python3
"""
Unified entry point for GET experiments.

Quick start:
    python run.py --task stage1_wedge_triangle --model fullget --epochs 50
    python run.py --task stage1_wedge_triangle --compare fullget,pairwiseget --seeds 3
    python run.py --list-tasks
    python run.py --list-models
"""
import argparse
import subprocess
import sys

from get.data.protocol import TASK_SPECS

ALL_TASKS = sorted(TASK_SPECS.keys())

ALL_MODELS = [
    "fullget", "pairwiseget", "quadratic_only",
    "get_ham_global",
    "et", "etfaithful", "gt", "bwgnn", "gin", "gcn", "gat",
]


def main():
    parser = argparse.ArgumentParser(description="GET Experiment Runner")
    parser.add_argument("--task", help="Task name (e.g. stage1_wedge_triangle)")
    parser.add_argument("--model", default="fullget", help="Model name (e.g. fullget, pairwiseget)")
    parser.add_argument("--compare", help="Compare models: comma-separated (e.g. fullget,pairwiseget)")
    parser.add_argument("--seeds", type=int, default=1, help="Number of random seeds")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--extra", default="", help="Extra args passed to protocol runner")
    args = parser.parse_args()

    if args.list_tasks:
        print("\nAvailable tasks:")
        for t in ALL_TASKS:
            print(f"  {t}")
        print()
        return

    if args.list_models:
        print("\nAvailable models:")
        for m in ALL_MODELS:
            print(f"  {m}")
        print()
        return

    if args.compare:
        cmd = [
            sys.executable, "scripts/compare.py",
            "--task", args.task or "stage1_wedge_triangle",
            "--models", args.compare,
            "--seeds", str(args.seeds),
            "--epochs", str(args.epochs),
            "--device", args.device,
        ]
        if args.extra:
            cmd += ["--extra", args.extra]
    elif args.task:
        cmd = [
            sys.executable, "experiments/run_protocol.py",
            "--task", args.task,
            "--model_name", args.model,
            "--seed", "123",
            "--epochs", str(args.epochs),
            "--device", args.device,
            "--num_runs", str(args.seeds),
        ]
        if args.extra:
            cmd += args.extra.split()
    else:
        parser.print_help()
        return

    print(f"\n{'='*60}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
