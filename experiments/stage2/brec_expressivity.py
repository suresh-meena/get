import argparse
from pathlib import Path
import torch
import json
import numpy as np
import urllib.request
import zipfile
from tempfile import NamedTemporaryFile
import shutil
import networkx as nx

from get import ETFaithful, FullGET, PairwiseGET, collate_get_batch
from experiments.shared.common import save_results, get_num_params, set_seed

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
    if isinstance(g, (bytes, str)):
        g_bytes = g if isinstance(g, bytes) else g.encode("ascii")
        graph = nx.from_graph6_bytes(g_bytes)
        num_nodes = graph.number_of_nodes()
        x = torch.ones(num_nodes, 1, dtype=torch.float32)
        edges = [(int(u), int(v)) for u, v in graph.edges()]
        return {"x": x, "edges": edges}
    raise ValueError("BREC graph format must contain `x` and `edges`.")


def _load_brec_pairs(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"BREC input not found: {p}")
    if p.suffix == ".pt":
        payload = torch.load(p, weights_only=False)
    elif p.suffix == ".npy":
        payload = np.load(p, allow_pickle=True)
        if isinstance(payload, np.ndarray):
            if payload.shape == ():
                payload = payload.item()
            else:
                payload = payload.tolist()
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


def _pair_category(pair_idx):
    if pair_idx < 60:
        return "Basic"
    if pair_idx < 160:
        return "Regular"
    if pair_idx < 260:
        return "Extension"
    if pair_idx < 360:
        return "CFI"
    if pair_idx < 380:
        return "4-Vertex_Condition"
    return "Distance_Regular"


def _is_graph_like(item):
    return isinstance(item, (bytes, str)) or (isinstance(item, dict) and "x" in item and "edges" in item)


def _convert_raw_brec_payload(payload):
    if isinstance(payload, np.ndarray):
        if payload.shape == ():
            payload = payload.item()
        else:
            payload = payload.tolist()

    if isinstance(payload, dict):
        pairs = []
        for cat, items in payload.items():
            for item in items:
                g1 = _normalize_graph(item["g1"])
                g2 = _normalize_graph(item["g2"])
                pairs.append((str(cat), g1, g2))
        return pairs

    if isinstance(payload, list):
        if not payload:
            return []
        first = payload[0]
        if isinstance(first, dict) and "g1" in first and "g2" in first:
            pairs = []
            for item in payload:
                cat = str(item.get("category", "Unknown"))
                g1 = _normalize_graph(item["g1"])
                g2 = _normalize_graph(item["g2"])
                pairs.append((cat, g1, g2))
            return pairs
        if isinstance(first, (tuple, list)) and len(first) == 3 and _is_graph_like(first[1]) and _is_graph_like(first[2]):
            return [(str(item[0]), _normalize_graph(item[1]), _normalize_graph(item[2])) for item in payload]
        if len(payload) % 2 != 0:
            raise ValueError("Raw BREC payload has an odd number of graphs; cannot infer pairs.")
        pairs = []
        for pair_idx in range(len(payload) // 2):
            g1 = _normalize_graph(payload[2 * pair_idx])
            g2 = _normalize_graph(payload[2 * pair_idx + 1])
            pairs.append((_pair_category(pair_idx), g1, g2))
        return pairs

    raise ValueError(f"Unsupported raw BREC payload type: {type(payload)!r}")


def _save_brec_pairs_jsonl(pairs, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for category, g1, g2 in pairs:
            record = {
                "category": category,
                "g1": {
                    "x": g1["x"].tolist(),
                    "edges": g1["edges"],
                },
                "g2": {
                    "x": g2["x"].tolist(),
                    "edges": g2["edges"],
                },
            }
            f.write(json.dumps(record) + "\n")
    return out_path


def _download_brec_archive(data_root="data", url=None):
    root = Path(data_root) / "BREC"
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "BREC_data_all.zip"
    if archive_path.exists() and archive_path.stat().st_size > 0:
        return archive_path

    download_url = url or "https://github.com/GraphPKU/BREC/raw/refs/heads/Release/BREC_data_all.zip"
    print(f"Downloading BREC archive from {download_url}")
    with NamedTemporaryFile(delete=False, dir=str(root), suffix=".zip") as tmp:
        tmp_path = Path(tmp.name)
        with urllib.request.urlopen(download_url) as response:
            shutil.copyfileobj(response, tmp)
    tmp_path.replace(archive_path)
    return archive_path


def _extract_brec_archive(archive_path, data_root="data"):
    root = Path(data_root) / "BREC"
    extract_dir = root / "downloaded"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(extract_dir)
    return extract_dir


def _find_raw_brec_npy(extract_dir):
    candidates = list(Path(extract_dir).rglob("brec_v3.npy"))
    if candidates:
        return candidates[0]
    candidates = list(Path(extract_dir).rglob("*.npy"))
    if candidates:
        return candidates[0]
    return None


def _resolve_brec_file(brec_file, data_root="data"):
    candidates = []
    if brec_file:
        candidates.append(Path(brec_file))

    root = Path(data_root)
    candidates.extend(
        [
            root / "BREC" / "brec_v3.npy",
            root / "BREC" / "brec_pairs.pt",
            root / "BREC" / "brec_pairs.json",
            root / "BREC" / "brec_pairs.jsonl",
            root / "brec_v3.npy",
            root / "brec_pairs.pt",
            root / "brec_pairs.json",
            root / "brec_pairs.jsonl",
        ]
    )

    checked = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.exists():
            return candidate

    archive_path = _download_brec_archive(data_root=data_root)
    extract_dir = _extract_brec_archive(archive_path, data_root=data_root)
    for candidate in [
        extract_dir / "brec_v3.npy",
        extract_dir / "Data" / "raw" / "brec_v3.npy",
        extract_dir / "brec_pairs.pt",
        extract_dir / "brec_pairs.json",
        extract_dir / "brec_pairs.jsonl",
    ]:
        checked.append(str(candidate))
        if candidate.exists():
            return candidate

    raw_npy = _find_raw_brec_npy(extract_dir)
    if raw_npy is not None:
        checked.append(str(raw_npy))
        raw_payload = np.load(raw_npy, allow_pickle=True)
        pairs = _convert_raw_brec_payload(raw_payload)
        generated = Path(data_root) / "BREC" / "generated_brec_pairs.jsonl"
        _save_brec_pairs_jsonl(pairs, generated)
        return generated

    raise FileNotFoundError(
        "No BREC input file found after download. Checked: " + ", ".join(checked)
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--pairwise_hidden_dim", type=int, default=0, help="Defaults to hidden_dim; set higher for parameter matching.")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Node embedding dimension used for random-feature comparison.")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--brec_file", type=str, default=None, help="Path to BREC pairs (.pt/.json/.jsonl). If omitted, common repo-local paths are checked.")
    parser.add_argument("--data_root", type=str, default="data", help="Root directory used when resolving default BREC paths.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairwise_dim = args.pairwise_hidden_dim if args.pairwise_hidden_dim > 0 else args.hidden_dim
    comparer = BRECComparer(eps=args.eps)

    print("-" * 50)
    print("EXPERIMENT: BREC Random-Feature Expressivity Probe")
    print(f"DEVICE:     {device}")
    print(f"SEEDS:      {args.seeds}")
    print("-" * 50)

    brec_file = _resolve_brec_file(args.brec_file, data_root=args.data_root)
    print(f"Using BREC input: {brec_file}")
    pairs = _load_brec_pairs(brec_file)
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
        "ETFaithful": lambda: ETFaithful(
            in_dim=in_dim,
            d=args.hidden_dim,
            num_classes=args.embedding_dim,
            num_steps=8,
            pe_k=0,
            rwse_k=0,
            mask_mode="sparse",
            et_official_mode=False,
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
