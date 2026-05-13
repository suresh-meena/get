import argparse, json
from collections import defaultdict
from pathlib import Path

def _score(res):
    for key in ("test", "summary"):
        block = res.get(key, {})
        if isinstance(block, dict):
            for metric in ("auc", "acc", "mae"):
                val = block.get(metric)
                if val is not None:
                    return float(val)
    summary = res.get("summary", {})
    for metric in ("test_auc_mean", "test_acc_mean", "test_mae_mean"):
        val = summary.get(metric)
        if val is not None:
            std = summary.get(metric.replace("mean", "std"), 0.0)
            return (float(val), float(std))
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="outputs/protocol")
    parser.add_argument("--models", default="", help="filter by model, comma-sep")
    args = parser.parse_args()

    filter_models = [m.strip() for m in args.models.split(",")] if args.models else None
    data = defaultdict(lambda: defaultdict(list))

    for f in sorted(Path(args.dir).glob("*.json")):
        if f.name == "manifest.json": continue
        try:
            res = json.loads(f.read_text())
            task, model = res.get("task", "?"), res.get("model_name", "?")
            if filter_models and model not in filter_models: continue
            s = _score(res)
            if s: data[task][model].append(s)
        except: pass

    tasks, models = sorted(data), sorted({m for t in data for m in data[t]})
    if not tasks: print("No results"); return

    print(f"\n  Results: {len(tasks)} tasks, {len(models)} models\n")
    for task in tasks:
        print(f"  [{task}]")
        for m in models:
            scores = data[task].get(m, [])
            if scores:
                if len(scores) == 1:
                    s = scores[0]
                    val = s if isinstance(s, (int, float)) else s[0]
                    print(f"    {m:<20} {val:.4f}")
                else:
                    vals = [s if isinstance(s, (int, float)) else s[0] for s in scores]
                    mu = sum(vals)/len(vals)
                    sd = (sum((v-mu)**2 for v in vals)/len(vals))**0.5
                    print(f"    {m:<20} {mu:.4f}±{sd:.4f}  (n={len(vals)})")
            else:
                print(f"    {m:<20} -")
        print()

if __name__ == "__main__":
    main()
