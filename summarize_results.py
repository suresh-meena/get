import json
from pathlib import Path
from collections import defaultdict


def _extract_score(res: dict) -> str:
    # Standardized schema: task > model_name > seed > {train, val, test, history, ...}
    for key in ("test", "summary"):
        block = res.get(key, {})
        if isinstance(block, dict):
            for metric in ("auc", "acc", "mae"):
                val = block.get(metric)
                if val is not None:
                    name = "AUC" if metric == "auc" else ("Acc" if metric == "acc" else "MAE")
                    return f"{float(val):.4f} ({name})"
    # Multi-run summary
    summary = res.get("summary", {})
    for metric in ("test_auc_mean", "test_acc_mean", "test_mae_mean"):
        val = summary.get(metric)
        if val is not None:
            std = summary.get(metric.replace("mean", "std"), 0.0)
            name = metric.split("_")[-2].upper()
            return f"{float(val):.4f} +/- {float(std):.4f} ({name})"
    return "N/A"


def summarize_results():
    results_dir = Path("outputs/protocol")
    if not results_dir.exists():
        print("No results directory found at outputs/protocol/")
        return

    data = defaultdict(dict)

    for f in sorted(results_dir.glob("*.json")):
        if f.name == "manifest.json":
            continue
        try:
            res = json.loads(f.read_text())
            task = res.get("task", "unknown")
            model = res.get("model_name", "unknown")
            seed = res.get("seed", "?")
            key = f"{model}_seed{seed}" if res.get("num_runs", 1) > 1 else model
            data[task][key] = _extract_score(res)
        except Exception as e:
            print(f"Error parsing {f}: {e}")

    tasks = sorted(data.keys())
    models = sorted({m for task in data for m in data[task]})

    with open("results_summary.md", "w") as md:
        md.write("# Graph Energy Transformer Benchmark Results\n\n")
        header = "| Task | " + " | ".join(models) + " |\n"
        sep = "| :--- | " + " | ".join(["---"] * len(models)) + " |\n"
        md.write(header)
        md.write(sep)
        for task in tasks:
            row = f"| {task} | "
            row += " | ".join(data[task].get(m, "-") for m in models)
            row += " |\n"
            md.write(row)

    print(f"\nSummarized {len(data)} tasks, {len(models)} model variants to results_summary.md")


if __name__ == "__main__":
    summarize_results()
