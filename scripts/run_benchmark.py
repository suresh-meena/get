"""
Batch experiment runner for synthetic + expressivity benchmarks.
Splits across 2 GPUs, runs one experiment per GPU at a time.
"""
import json, os, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "outputs" / "benchmark"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONDA_PYTHON = "/data/ayand/miniforge3/envs/get/bin/python"

TASKS = [
    "stage1_wedge_triangle",
    "stage1_triangle_regression",
    "stage1_cycle_parity",
    "stage1_max3sat",
    "stage1_xorsat",
    "stage1_srg_discrimination",
    "stage2_csl",
]

MODELS = [
    "fullget_local",
    "fullget_global",
    "pairwise_only",
    "quadratic_only",
    "nomotif_local",
    "et",
    "gin",
    "gcn",
]

def run_experiment(task, model, gpu_id, epochs=50):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    out_file = RESULTS_DIR / f"{task}__{model}__gpu{gpu_id}.json"
    if out_file.exists():
        return

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_out = f.name

    base_cmd = [
        CONDA_PYTHON, str(ROOT / "experiments/run_protocol.py"),
        "--task", task,
        "--model_name", model,
        "--seed", "123",
        "--epochs", str(epochs),
        "--device", "cuda",
        "--output_file", tmp_out,
        "--batch_size", "64",
        "--num_workers", "4",
        "--no-compile",
        "--pos_k", "0",
        "--max_graphs", "256",
    ]

    t0 = time.time()
    result = subprocess.run(
        base_cmd, capture_output=True, text=True, timeout=600
    )
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else "FAIL"

    data = {}
    if result.returncode == 0:
        try:
            data = json.loads(Path(tmp_out).read_bytes())
        except Exception:
            status = "PARSE_FAIL"
            data = {"stderr": result.stderr[:500]}
    else:
        data = {"stderr": result.stderr[-500:]}

    summary = {
        "task": task,
        "model": model,
        "gpu": gpu_id,
        "status": status,
        "runtime_seconds": elapsed,
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True, choices=[0, 1])
    args = parser.parse_args()
    gpu_id = args.gpu

    # Each GPU gets every other experiment
    all_exps = [(t, m) for t in TASKS for m in MODELS]
    my_exps = [e for i, e in enumerate(all_exps) if i % 2 == gpu_id]

    n = len(my_exps)
    print(f"  [GPU{gpu_id}] Running {n} experiments")
    for idx, (task, model) in enumerate(my_exps):
        done = (RESULTS_DIR / f"{task}__{model}__gpu{gpu_id}.json").exists()
        tag = "cached" if done else "running"
        print(f"  [GPU{gpu_id}] ({idx+1}/{n}) {task} / {model} ... {tag}", flush=True)
        if not done:
            run_experiment(task, model, gpu_id)

if __name__ == "__main__":
    main()
