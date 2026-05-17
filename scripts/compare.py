import argparse, json, subprocess, sys, tempfile
from pathlib import Path

def _run(task, model, seed, epochs, device):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        out = f.name
    r = subprocess.run([sys.executable, "experiments/run_protocol.py",
        "--task", task, "--model_name", model, "--seed", str(seed),
        "--epochs", str(epochs), "--device", device,
        "--output_file", out, "--batch_size", "64", "--num_workers", "0"],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  {model} seed={seed}: FAILED ({r.stderr[:100]})")
        return None
    data = json.loads(Path(out).read_bytes())
    Path(out).unlink(missing_ok=True)
    return data

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="stage1_wedge_triangle")
    p.add_argument("--models", default="fullget_local,pairwise_only")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    print(f"\n  Task: {args.task}  Models: {', '.join(models)}  Seeds: {args.seeds}  Epochs: {args.epochs}\n")

    results = {m: [] for m in models}
    for model in models:
        for s in range(args.seeds):
            seed = 123 + s
            data = _run(args.task, model, seed, args.epochs, args.device)
            if data:
                test = data.get("test", {})
                score = test.get("auc") or test.get("acc") or test.get("mae")
                if score is not None:
                    results[model].append(score)
                    print(f"  {model:<15} seed={seed:<4} {score:.4f}")
                else:
                    print(f"  {model:<15} seed={seed:<4} no score")
            else:
                results[model].append(None)

    print(f"\n  {'Model':<15} {'Mean':<8} {'Std':<8} Scores")
    print(f"  {'-'*60}")
    for model in models:
        scores = [s for s in results[model] if s is not None]
        if scores:
            mu = sum(scores)/len(scores)
            sd = (sum((s-mu)**2 for s in scores)/len(scores))**0.5
            print(f"  {model:<15} {mu:<8.4f} {sd:<8.4f} {', '.join(f'{s:.4f}' for s in scores)}")
        else:
            print(f"  {model:<15} FAILED")

if __name__ == "__main__":
    main()
