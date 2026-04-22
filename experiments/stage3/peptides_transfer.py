import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from get import FullGET, PairwiseGET, GINBaseline, ETFaithful, collate_get_batch
from get.data import CachedGraphDataset
from experiments.common import build_dataloader_kwargs, set_seed, save_results


def load_peptides(task: str, root="data/LRGB"):
    from torch_geometric.datasets import LRGBDataset
    if task not in {"func", "struct"}:
        raise ValueError("task must be one of: func, struct")
    name = "Peptides-func" if task == "func" else "Peptides-struct"
    tr = LRGBDataset(root=root, name=name, split="train")
    val = LRGBDataset(root=root, name=name, split="val")
    ts = LRGBDataset(root=root, name=name, split="test")

    def _to_dict(ds):
        out = []
        for d in ds:
            x = d.x.float() if d.x is not None else torch.ones(d.num_nodes, 1)
            y = d.y.float().view(-1)
            item = {
                "x": x,
                "edges": list(zip(d.edge_index[0].tolist(), d.edge_index[1].tolist())),
                "y": y.unsqueeze(0) if y.ndim == 0 else y.unsqueeze(0) if y.ndim == 1 else y,
            }
            if d.edge_attr is not None:
                item["edge_attr"] = d.edge_attr.float()
            out.append(item)
        return out

    return _to_dict(tr), _to_dict(val), _to_dict(ts)


def _evaluate_ap(y_true, y_score):
    try:
        from sklearn.metrics import average_precision_score
        y_true = np.asarray(y_true, dtype=np.float64)
        y_score = np.asarray(y_score, dtype=np.float64)
        if y_true.ndim == 1:
            return float(average_precision_score(y_true, y_score))
        aps = []
        for c in range(y_true.shape[1]):
            if len(set(y_true[:, c].tolist())) < 2:
                continue
            aps.append(average_precision_score(y_true[:, c], y_score[:, c]))
        return float(np.mean(aps)) if aps else 0.0
    except Exception:
        return 0.0


def _train_multilabel(model, tr, val, ts, epochs, batch_size, device):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss()
    loader_kwargs = build_dataloader_kwargs(device)
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, collate_fn=collate_get_batch, **loader_kwargs)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=collate_get_batch, **loader_kwargs)
    ts_loader = DataLoader(ts, batch_size=batch_size, shuffle=False, collate_fn=collate_get_batch, **loader_kwargs)
    best = {"val": -1.0, "test": 0.0}

    def _collect(loader):
        ys, ps = [], []
        model.eval()
        with torch.no_grad():
            for b in loader:
                b = b.to(device)
                out, _ = model(b, task_level="graph")
                prob = torch.sigmoid(out)
                ys.append(b.y.view(out.shape).cpu().numpy())
                ps.append(prob.cpu().numpy())
        return np.concatenate(ys, axis=0), np.concatenate(ps, axis=0)

    for _ in range(epochs):
        model.train()
        for b in tr_loader:
            b = b.to(device)
            opt.zero_grad()
            out, _ = model(b, task_level="graph")
            y = b.y.view(out.shape).float()
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        yv, pv = _collect(val_loader)
        yt, pt = _collect(ts_loader)
        val_ap = _evaluate_ap(yv, pv)
        test_ap = _evaluate_ap(yt, pt)
        if val_ap > best["val"]:
            best = {"val": val_ap, "test": test_ap}
    return {"metric": float(best["test"])}


def _train_regression(model, tr, val, ts, epochs, batch_size, device):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = torch.nn.L1Loss()
    loader_kwargs = build_dataloader_kwargs(device)
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, collate_fn=collate_get_batch, **loader_kwargs)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=collate_get_batch, **loader_kwargs)
    ts_loader = DataLoader(ts, batch_size=batch_size, shuffle=False, collate_fn=collate_get_batch, **loader_kwargs)
    best = {"val": float("inf"), "test": float("inf")}

    def _mae(loader):
        ys, ps = [], []
        model.eval()
        with torch.no_grad():
            for b in loader:
                b = b.to(device)
                out, _ = model(b, task_level="graph")
                ys.append(b.y.view_as(out).cpu().numpy())
                ps.append(out.cpu().numpy())
        y = np.concatenate(ys, axis=0)
        p = np.concatenate(ps, axis=0)
        return float(np.mean(np.abs(y - p)))

    for _ in range(epochs):
        model.train()
        for b in tr_loader:
            b = b.to(device)
            opt.zero_grad()
            out, _ = model(b, task_level="graph")
            y = b.y.view_as(out).float()
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val_mae = _mae(val_loader)
        test_mae = _mae(ts_loader)
        if val_mae < best["val"]:
            best = {"val": val_mae, "test": test_mae}
    return {"metric": float(best["test"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["func", "struct"], required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", default="data/LRGB")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr_raw, val_raw, ts_raw = load_peptides(task=args.task, root=args.data_root)

    tr = CachedGraphDataset(tr_raw, name=f"peptides_{args.task}_train", max_motifs=16, pe_k=16).cached_data
    val = CachedGraphDataset(val_raw, name=f"peptides_{args.task}_val", max_motifs=16, pe_k=16).cached_data
    ts = CachedGraphDataset(ts_raw, name=f"peptides_{args.task}_test", max_motifs=16, pe_k=16).cached_data
    in_dim = tr[0]["x"].size(1)
    out_dim = int(tr[0]["y"].numel())

    models = {
        "PairwiseGET": PairwiseGET(in_dim, args.hidden_dim, out_dim),
        "FullGET": FullGET(in_dim, args.hidden_dim, out_dim, lambda_3=1.0),
        "ETFaithful": ETFaithful(in_dim, args.hidden_dim, out_dim, pe_k=16, num_steps=6),
    }
    try:
        models["GIN"] = GINBaseline(in_dim, args.hidden_dim, out_dim)
    except Exception:
        pass

    results = {}
    for name, model in models.items():
        print(f"--- {name} on Peptides-{args.task} ---")
        if args.task == "func":
            res = _train_multilabel(model, tr, val, ts, args.epochs, args.batch_size, device)
            print(f"{name} Test AP: {res['metric']:.4f}")
        else:
            res = _train_regression(model, tr, val, ts, args.epochs, args.batch_size, device)
            print(f"{name} Test MAE: {res['metric']:.4f}")
        results[name] = res

    save_results(f"exp9_peptides_{args.task}_results", results)


if __name__ == "__main__":
    main()
