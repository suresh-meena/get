import argparse, json, sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="result JSON or directory")
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--output", "-o", default="energy_traces.png")
    args = parser.parse_args()

    path = Path(args.input)
    files = sorted(path.glob(args.pattern)) if path.is_dir() else ([path] if path.is_file() else [])

    traces = []
    for f in files:
        data = json.loads(f.read_text())
        trace = data.get("energy_trace") or data.get("solver_energy_trace")
        if isinstance(trace, list) and len(trace) > 0:
            model = data.get("model_name", f.stem)
            traces.append((model, trace))
            print(f"  {f.name}: {model}, {len(trace)} steps, {trace[0]:.2f} -> {trace[-1]:.2f}")

    if not traces:
        print("No energy traces found.")
        return

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\nFound {len(traces)} traces. Install matplotlib to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for model, trace in traces:
        ax.plot(trace, label=model, alpha=0.8)
    ax.set(xlabel="Inference Step", ylabel="Energy", title="Energy Decay")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\n  Plot saved to {args.output}")

if __name__ == "__main__":
    main()
