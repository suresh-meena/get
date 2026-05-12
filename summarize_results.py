import json
import os
from pathlib import Path
from collections import defaultdict

def summarize_results():
    results_dir = Path("outputs/protocol")
    if not results_dir.exists():
        print("No results directory found.")
        return

    # task -> model -> result_dict
    data = defaultdict(dict)
    
    for f in results_dir.glob("*.json"):
        if f.name == "last_metrics.json":
            continue
        try:
            with open(f, "r") as json_file:
                res = json.load(json_file)
                task = res.get("task")
                runtime = res.get("runtime_config", {})
                model = runtime.get("model_name", "unknown")
                
                # Handle splits for anomaly
                if "_split1" in f.name:
                    model = f"{model} (1%)"
                elif "_split40" in f.name:
                    model = f"{model} (40%)"
                
                metrics = res.get("metrics", {}).get("test", {})
                if not metrics:
                    metrics = res.get("metrics", {}).get("final_test", {})
                
                score = "N/A"
                if "auc" in metrics:
                    score = f"{metrics['auc']:.4f} (AUC)"
                elif "acc" in metrics:
                    score = f"{metrics['acc']:.4f} (Acc)"
                elif "mae" in metrics:
                    score = f"{metrics['mae']:.4f} (MAE)"
                
                data[task][model] = score
        except Exception as e:
            print(f"Error parsing {f}: {e}")

    tasks = sorted(data.keys())
    models = sorted({str(m) for task in data for m in data[task] if m is not None})
    
    # Write to markdown
    with open("results_summary.md", "w") as md:
        md.write("# 📊 Graph Energy Transformer Benchmark Results\n\n")
        
        # Header
        header = "| Task | " + " | ".join(models) + " |\n"
        separator = "| :--- | " + " | ".join(["---"] * len(models)) + " |\n"
        md.write(header)
        md.write(separator)
        
        for task in tasks:
            row = f"| {task} | "
            row += " | ".join([data[task].get(m, "-") for m in models])
            row += " |\n"
            md.write(row)

if __name__ == "__main__":
    summarize_results()
