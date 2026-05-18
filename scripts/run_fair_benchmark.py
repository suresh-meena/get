"""Fair benchmark with matched parameter counts across all models."""
import json, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "benchmark_matched"
OUT.mkdir(parents=True, exist_ok=True)

PYTHON = "/data/ayand/miniforge3/envs/get/bin/python"

TASKS = [
    "stage1_wedge_triangle",
    "stage1_triangle_regression",
    "stage1_cycle_parity",
    "stage1_max3sat",
    "stage1_xorsat",
    "stage1_srg_discrimination",
    "stage2_csl",
]

# Model configs with matched ~200K params
# Each entry: (model_name, [extra_cli_args])
MODELS = [
    ("fullget_local", ["--hidden_dim", "128", "--num_heads", "4", "--head_dim", "32", "--num_blocks", "1", "--num_steps", "8"]),
    ("fullget_global", ["--hidden_dim", "128", "--num_heads", "4", "--head_dim", "32", "--num_blocks", "1", "--num_steps", "8"]),
    ("pairwise_only", ["--hidden_dim", "128", "--num_heads", "4", "--head_dim", "32", "--num_blocks", "1", "--num_steps", "8"]),
    ("quadratic_only", ["--hidden_dim", "128", "--num_heads", "4", "--head_dim", "32", "--num_blocks", "1", "--num_steps", "8"]),
    ("nomotif_local", ["--hidden_dim", "128", "--num_heads", "4", "--head_dim", "32", "--num_blocks", "1", "--num_steps", "8"]),
    ("et", ["--hidden_dim", "128", "--num_heads", "2", "--head_dim", "64", "--num_blocks", "2", "--num_steps", "1"]),
    ("gt", ["--hidden_dim", "56"]),
    ("gin", ["--hidden_dim", "256"]),
    ("gcn", ["--hidden_dim", "448"]),
    ("gat", ["--hidden_dim", "448", "--num_heads", "4"]),
]

def count_params(model_name, extra_args):
    """Quick param count without running a full experiment."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        out = f.name
    cmd = [PYTHON, str(ROOT / "experiments/run_protocol.py"),
           "--task", "stage1_wedge_triangle",
           "--model_name", model_name,
           "--epochs", "1", "--device", "cuda",
           "--batch_size", "2", "--num_workers", "0",
           "--no-compile", "--pos_k", "0", "--max_graphs", "4",
           "--output_file", out] + extra_args
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    params = None
    if r.returncode == 0:
        try:
            data = json.loads(Path(out).read_bytes())
            params = data.get("parameter_count", 0)
        except Exception:
            params = "PARSE_ERR"
    else:
        params = f"ERR({r.stderr[-100:]})"
    Path(out).unlink(missing_ok=True)
    return params

def run_experiment(task, model_name, extra_args, gpu_id, epochs=100):
    out_file = OUT / f"{task}__{model_name}.json"
    if out_file.exists():
        return json.loads(out_file.read_bytes())

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_out = f.name

    cmd = [PYTHON, str(ROOT / "experiments/run_protocol.py"),
           "--task", task, "--model_name", model_name,
           "--seed", "123", "--epochs", str(epochs),
           "--device", "cuda", "--output_file", tmp_out,
           "--batch_size", "64", "--num_workers", "4",
           "--no-compile", "--pos_k", "0", "--max_graphs", "256",
    ] + extra_args

    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0

    status = "OK" if r.returncode == 0 else "FAIL"
    data = {}
    if r.returncode == 0:
        try:
            data = json.loads(Path(tmp_out).read_bytes())
        except Exception:
            status = "PARSE_FAIL"
    else:
        data = {"stderr": r.stderr[-500:]}

    summary = {
        "task": task, "model": model_name,
        "status": status, "runtime_seconds": elapsed,
        "test": data.get("test", {}),
        "best_val_score": data.get("best_val_score"),
        "epochs_ran": data.get("epochs_ran"),
        "peak_cuda_memory_mb": data.get("peak_cuda_memory_mb"),
        "parameter_count": data.get("parameter_count"),
        "error": data.get("stderr", "") if status != "OK" else None,
    }
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    Path(tmp_out).unlink(missing_ok=True)
    return summary

def print_table(results):
    """Print a formatted results table."""
    task_types = {
        'stage1_wedge_triangle': ('Binary', 'auc'),
        'stage1_triangle_regression': ('Regression', 'mae'),
        'stage1_cycle_parity': ('Binary', 'auc'),
        'stage1_max3sat': ('Binary', 'auc'),
        'stage1_xorsat': ('Binary', 'auc'),
        'stage1_srg_discrimination': ('Binary', 'auc'),
        'stage2_csl': ('Multiclass', 'acc'),
    }

    print(f"\n{'='*110}")
    print(f"  FAIR BENCHMARK: Matched ~200K params  |  100 epochs  |  2× RTX 3090")
    print(f"{'='*110}")

    for task in TASKS:
        tt, metric = task_types[task]
        print(f"\n  {task}  ({tt}, metric={metric})")
        print(f"  {'Model':<20} {'Score':<12} {'Params':<10} {'Runtime':<10} {'Mem':<10}")
        print(f"  {'-'*60}")

        best = float('inf') if metric == 'mae' else float('-inf')
        best_model = None
        for model_name, _ in MODELS:
            r = results.get(task, {}).get(model_name)
            if not r or r['status'] != 'OK':
                print(f"  {model_name:<20} {'FAILED':<12}")
                continue
            test = r.get('test', {}) or {}
            score = test.get(metric, 'N/A')
            params = r.get('parameter_count', 0)
            rt = r.get('runtime_seconds', 0) / 60
            mem = r.get('peak_cuda_memory_mb', 0)
            if isinstance(score, (int, float)):
                if (metric == 'mae' and score < best) or (metric != 'mae' and score > best):
                    best = score
                    best_model = model_name
            print(f"  {model_name:<20} {score if isinstance(score, str) else f'{score:.4f}':<12} {params:<10} {rt:.1f}m    {mem:<8.0f}MB")
        if best_model:
            print(f"  {'→ Best:':>8} {best_model:<20} ({best if isinstance(best, str) else f'{best:.4f}'})")

if __name__ == "__main__":
    import os
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0, choices=[0, 1])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    # Show param counts
    print(f"\n  Parameter counts (quick check):")
    for model_name, extra in MODELS:
        pcount = count_params(model_name, extra)
        print(f"    {model_name:<20} {pcount} params")

    if args.dry_run:
        sys.exit(0)

    all_exps = [(t, m, e) for t in TASKS for m, e in MODELS]
    my_exps = [e for i, e in enumerate(all_exps) if i % 2 == args.gpu]

    print(f"\n  [GPU{args.gpu}] Running {len(my_exps)} experiments...\n")
    results = {}
    for idx, (task, model_name, extra_args) in enumerate(my_exps):
        done = (OUT / f"{task}__{model_name}.json").exists()
        tag = "cached" if done else "running"
        print(f"  [{args.gpu}] ({idx+1}/{len(my_exps)}) {task} / {model_name} ... {tag}", flush=True)
        r = run_experiment(task, model_name, extra_args, args.gpu)
        results.setdefault(task, {})[model_name] = r

    print(f"\n  [GPU{args.gpu}] Done.")
