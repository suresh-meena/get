import argparse
from pathlib import Path
import torch
import json
import numpy as np

from get import FullGET, PairwiseGET, collate_get_batch
from experiments.common import save_results, get_num_params, set_seed

class BRECComparer:
    def __init__(self, eps=1e-6):
        self.eps = eps
    def distinguish(self, z1, z2):
        if z1.shape[0] != z2.shape[0]:
            return True
        return torch.norm(torch.sort(z1, dim=0)[0] - torch.sort(z2, dim=0)[0], p='fro').item() > self.eps

def get_node_embeddings(model, g, device, collapse_motif_types=False):
    model.eval()
    batch = collate_get_batch([g]).to(device)
    if collapse_motif_types and batch.t_tau.numel() > 0:
        batch.t_tau.zero_()
    # This is a random-feature discrimination probe, not a supervised
    # benchmark. Armijo makes the deterministic inference map less step-size
    # sensitive across graph pairs.
    with torch.no_grad():
        X, _ = model(batch, task_level='node', inference_mode='armijo')
    return X


def _normalize_graph(g):
    if isinstance(g, dict):
        if "x" in g and "edges" in g:
            return {"x": g["x"].float(), "edges": [(int(u), int(v)) for u, v in g["edges"]]}
    raise ValueError("BREC graph format must contain `x` and `edges`.")


def _load_brec_pairs(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"BREC input not found: {p}")
    if p.suffix == ".pt":
        payload = torch.load(p, weights_only=False)
    elif p.suffix in {".json", ".jsonl"}:
        if p.suffix == ".jsonl":
            payload = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        else:
            payload = json.loads(p.read_text())
    else:
        raise ValueError("Unsupported BREC format. Use .pt, .json, or .jsonl")

    pairs = []
    if isinstance(payload, dict):
        for cat, items in payload.items():
            for item in items:
                g1 = _normalize_graph(item["g1"])
                g2 = _normalize_graph(item["g2"])
                pairs.append((str(cat), g1, g2))
    elif isinstance(payload, list):
        for item in payload:
            cat = str(item.get("category", "Unknown"))
            g1 = _normalize_graph(item["g1"])
            g2 = _normalize_graph(item["g2"])
            pairs.append((cat, g1, g2))
    else:
        raise ValueError("Unsupported BREC payload shape.")
    return pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--pairwise_hidden_dim", type=int, default=0, help="Defaults to hidden_dim; set higher for parameter matching.")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Node embedding dimension used for random-feature comparison.")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--brec_file", type=str, required=True, help="Path to BREC pairs (.pt/.json/.jsonl)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairwise_dim = args.pairwise_hidden_dim if args.pairwise_hidden_dim > 0 else args.hidden_dim
    comparer = BRECComparer(eps=args.eps)

    print("-" * 50)
    print("EXPERIMENT: BREC Random-Feature Expressivity Probe")
    print(f"DEVICE:     {device}")
    print(f"SEEDS:      {args.seeds}")
    print("-" * 50)

    pairs = _load_brec_pairs(args.brec_file)
    if not pairs:
        raise ValueError("BREC input contains no graph pairs.")
    in_dim = int(pairs[0][1]["x"].size(1))
    cats = sorted({cat for cat, _, _ in pairs})

    factories = {
        "PairwiseGET": lambda: PairwiseGET(
            in_dim=in_dim,
            d=pairwise_dim,
            num_classes=args.embedding_dim,
            num_steps=8,
        ),
        "FullGET": lambda: FullGET(
            in_dim=in_dim,
            d=args.hidden_dim,
            num_classes=args.embedding_dim,
            R=2,
            lambda_3=0.5,
            num_steps=8,
        ),
        "FullGETCollapsedMotifTypes": lambda: FullGET(
            in_dim=in_dim,
            d=args.hidden_dim,
            num_classes=args.embedding_dim,
            R=2,
            lambda_3=0.5,
            num_steps=8,
        ),
    }

    results = {}
    for model_name, factory in factories.items():
        collapse_types = model_name == "FullGETCollapsedMotifTypes"
        seed_rates = []
        per_category = {c: [] for c in cats}
        for seed in args.seeds:
            set_seed(seed)
            model = factory().to(device)
            counts = {c: {"total": 0, "dist": 0} for c in cats}
            print(f"Running {model_name} seed={seed} params={get_num_params(model)}")
            for cat, g1, g2 in pairs:
                counts[cat]["total"] += 1
                z1 = get_node_embeddings(model, g1, device, collapse_motif_types=collapse_types)
                z2 = get_node_embeddings(model, g2, device, collapse_motif_types=collapse_types)
                if comparer.distinguish(z1, z2):
                    counts[cat]["dist"] += 1

            total = sum(item["total"] for item in counts.values())
            dist = sum(item["dist"] for item in counts.values())
            seed_rate = float(dist / total) if total else 0.0
            seed_rates.append(seed_rate)
            for c, item in counts.items():
                rate = float(item["dist"] / item["total"]) if item["total"] else 0.0
                per_category[c].append(rate)

        results[model_name] = {
            "mean_rate": float(np.mean(seed_rates)) if seed_rates else 0.0,
            "std_rate": float(np.std(seed_rates)) if seed_rates else 0.0,
            "per_seed_rate": seed_rates,
            "per_category": {
                c: {
                    "mean_rate": float(np.mean(vals)) if vals else 0.0,
                    "std_rate": float(np.std(vals)) if vals else 0.0,
                    "per_seed_rate": vals,
                }
                for c, vals in per_category.items()
            },
        }
        print(f"{model_name}: {results[model_name]['mean_rate']:.3f} ± {results[model_name]['std_rate']:.3f}")

    metadata = {
        "interpretation": "Random-feature graph-pair discrimination. This probes realized separation of the deterministic architecture at initialization; it is not a supervised proof of k-WL expressivity.",
        "eps": args.eps,
        "hidden_dim": args.hidden_dim,
        "pairwise_hidden_dim": pairwise_dim,
        "embedding_dim": args.embedding_dim,
        "seeds": args.seeds,
    }
    save_results("brec_results", results, metadata=metadata)

if __name__ == "__main__":
    main()
