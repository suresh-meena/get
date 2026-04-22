import argparse
from pathlib import Path
import torch
import numpy as np
import networkx as nx
import json

from get import FullGET, collate_get_batch
from experiments.common import save_results

class BRECComparer:
    def __init__(self, eps=1e-6): self.eps = eps
    def distinguish(self, z1, z2):
        if z1.shape[0] != z2.shape[0]: return True
        return torch.norm(torch.sort(z1, dim=0)[0] - torch.sort(z2, dim=0)[0], p='fro').item() > self.eps

def get_node_embeddings(model, g, device):
    batch = collate_get_batch([g]).to(device)
    x = batch.x.view(-1, 1).float() if batch.x.dim() == 1 else batch.x
    X = model.node_encoder(x)
    X, _, _ = model._run_fixed_solver(X, batch, model._build_static_projections(batch))
    return model.get_layer.layernorm(X)


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
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--brec_file", type=str, required=True, help="Path to BREC pairs (.pt/.json/.jsonl)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FullGET(in_dim=1, d=args.hidden_dim, num_classes=1).to(device)
    comparer = BRECComparer(eps=args.eps)
    
    pairs = _load_brec_pairs(args.brec_file)
    cats = sorted({cat for cat, _, _ in pairs})
    results = {c: {"total": 0, "dist": 0} for c in cats}
    
    print("Running expressivity comparison...")
    for cat, g1, g2 in pairs:
        results[cat]["total"] += 1
        z1, z2 = get_node_embeddings(model, g1, device), get_node_embeddings(model, g2, device)
        if comparer.distinguish(z1, z2):
            results[cat]["dist"] += 1

    for c, res in results.items():
        rate = (res["dist"] / res["total"]) if res["total"] else 0.0
        res["rate"] = rate
        print(f"{c:10}: {res['dist']}/{res['total']} distinguished ({rate:.3f})")
    save_results("brec_results", results)

if __name__ == "__main__":
    main()
