import argparse
from experiments.stage4.runner import main as run_stage2_main

def main():
    parser = argparse.ArgumentParser(description="Deprecated wrapper. Use experiments/stage4/runner.py directly.")
    parser.add_argument("--task", choices=["tu", "anomaly"], required=True)
    parser.add_argument("--dataset", default="MUTAG")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    args, unknown = parser.parse_known_args()

    mapped_task = "graph_classification" if args.task == "tu" else "graph_anomaly"
    import sys
    sys.argv = [
        "run_stage2.py",
        "--task",
        mapped_task,
        "--dataset",
        args.dataset,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
    ] + unknown
    return run_stage2_main()

if __name__ == "__main__":
    raise SystemExit(main())
