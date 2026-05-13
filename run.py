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


STAGE1_TASKS = [
    "stage1_wedge_triangle", "stage1_triangle_regression", "stage1_cycle_parity",
    "stage1_max3sat", "stage1_xorsat", "stage1_srg_discrimination",
]
STAGE2_TASKS = ["stage2_csl", "stage2_brec"]
STAGE3_TASKS = ["stage3_zinc", "stage3_molhiv", "stage3_molpcba",
                "stage3_peptides_struct_probe", "stage3_peptides_func_probe"]
STAGE4_TASKS = [
    "stage4_tu_proteins", "stage4_tu_nci1", "stage4_tu_enzymes",
    "stage4_tu_mutagenicity", "stage4_yelpchi_anomaly",
]

ALL_TASKS = STAGE1_TASKS + STAGE2_TASKS + STAGE3_TASKS + STAGE4_TASKS

ALL_MODELS = [
    "fullget", "pairwiseget", "quadratic_only",
    "get_ham_global", "get_ham_cls", "get_ham_full",
    "et", "gt", "bwgnn", "gin", "gcn", "gat",
]


def main():
    parser = argparse.ArgumentParser(description="GET Experiment Runner")
    parser.add_argument("--task", help=f"Task name (e.g. stage1_wedge_triangle)")
    parser.add_argument("--model", default="fullget", help=f"Model name (e.g. fullget, pairwiseget)")
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
        models = [m.strip() for m in args.compare.split(",")]
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
