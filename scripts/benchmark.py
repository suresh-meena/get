import argparse, json, subprocess, sys, tempfile
from pathlib import Path

STAGE1 = ["stage1_wedge_triangle", "stage1_triangle_regression", "stage1_cycle_parity",
          "stage1_max3sat", "stage1_xorsat", "stage1_srg_discrimination"]

def _run(task, model, seed, epochs, device):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        out = f.name
    r = subprocess.run([sys.executable, "experiments/run_protocol.py",
        "--task", task, "--model_name", model, "--seed", str(seed),
        "--epochs", str(epochs), "--device", device,
        "--output_file", out, "--batch_size", "64", "--num_workers", "0"],
        capture_output=True, text=True)
    if r.returncode != 0: return None
    data = json.loads(Path(out).read_bytes())
    Path(out).unlink(missing_ok=True)
    return data

def _score(data, task_type):
    if data is None: return None
    if "summary" in data:
        key = "test_mae_mean" if task_type == "regression" else ("test_auc_mean" if task_type in ("binary", "node_binary", "multilabel") else "test_acc_mean")
        return data["summary"].get(key)
    test = data.get("test", {})
    return test.get("auc") or test.get("acc") or test.get("mae")

def _task_type(task):
    from get.data.protocol import TASK_SPECS
    return TASK_SPECS[task].task_type

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="", help="comma-sep tasks (default: Stage 1)")
    p.add_argument("--models", default="fullget,pairwiseget")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="outputs/benchmark_results.json")
    args = p.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else STAGE1
    models = [m.strip() for m in args.models.split(",")]

    print(f"\n  {len(tasks)} tasks x {len(models)} models x {args.seeds} seeds\n")
    all_results = {}

    for task in tasks:
        tt = _task_type(task)
        print(f"  [{task}]")
        task_res = {}
        for model in models:
            scores = []
            for s in range(args.seeds):
                seed = 123 + s
                data = _run(task, model, seed, args.epochs, args.device)
                score = _score(data, tt)
                if score is not None: scores.append(score)
                print(f"    {model:<15} seed={seed:<4} {score:.4f}" if score is not None else f"    {model:<15} seed={seed:<4} FAIL")
            if scores:
                mu = sum(scores)/len(scores)
                sd = (sum((s-mu)**2 for s in scores)/len(scores))**0.5
                task_res[model] = {"mean": round(mu,4), "std": round(sd,4)}
                print(f"    {'  ->':>20} {mu:.4f} +/- {sd:.4f}")
            else:
                task_res[model] = {"mean": None, "std": None}
            print()
        all_results[task] = task_res

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2))

    print(f"  {'Task':<30}", end="")
    for m in models: print(f" {m:>20}", end="")
    print(f"\n  {'-'*(30 + 22*len(models))}")
    for task in tasks:
        print(f"  {task:<30}", end="")
        for m in models:
            r = all_results[task].get(m, {})
            print(f" {r['mean']:.4f}+-{r['std']:.4f}" if r.get('mean') else f" {'FAILED':>20}", end="")
        print()
    print(f"\n  Saved to {args.output}")

if __name__ == "__main__":
    main()
