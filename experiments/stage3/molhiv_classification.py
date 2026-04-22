import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from get import FullGET, PairwiseGET, GINBaseline, ETFaithful, collate_get_batch
from get.data import CachedGraphDataset
from experiments.common import build_dataloader_kwargs, set_seed, save_results


def load_molhiv(root="data/OGB"):
    try:
        from ogb.graphproppred import PygGraphPropPredDataset
    except Exception as e:
        raise RuntimeError("ogb is required for molhiv. Install `ogb`.") from e

    ds = PygGraphPropPredDataset(name="ogbg-molhiv", root=root)
    split = ds.get_idx_split()

    def _convert(indices):
        out = []
        for idx in indices.tolist():
            d = ds[idx]
            y = d.y.view(-1)
            if y.numel() == 0 or torch.isnan(y[0]):
                continue
            x = d.x.float() if d.x is not None else torch.ones(d.num_nodes, 1)
            edge_attr = d.edge_attr.float() if d.edge_attr is not None else None
            item = {
                "x": x,
                "edges": list(zip(d.edge_index[0].tolist(), d.edge_index[1].tolist())),
                "y": torch.tensor([float(y[0].item())], dtype=torch.float32),
            }
            if edge_attr is not None:
                item["edge_attr"] = edge_attr
            out.append(item)
        return out

    return _convert(split["train"]), _convert(split["valid"]), _convert(split["test"])


def _safe_auc(y_true, y_score):
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(int(y >= 0.5) for y in y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


def _train_binary(model, train_ds, val_ds, test_ds, epochs, batch_size, device):
    model = model.to(device)
    y = torch.tensor([float(g["y"].item()) for g in train_ds], dtype=torch.float32)
    pos = float((y > 0.5).sum().item())
    neg = float((y <= 0.5).sum().item())
    if pos > 0 and neg > 0:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    loader_kwargs = build_dataloader_kwargs(device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_get_batch, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_get_batch, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_get_batch, **loader_kwargs)

    best_auc = -1.0
    best = None
    history = {"train_loss": [], "val_auc": [], "test_auc": []}

    def _eval(loader):
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for b in loader:
                b = b.to(device)
                out, _ = model(b, task_level="graph")
                prob = torch.sigmoid(out.view(-1))
                ys.extend(b.y.view(-1).cpu().numpy().tolist())
                ps.extend(prob.cpu().numpy().tolist())
        return _safe_auc(ys, ps)

    for _ in range(epochs):
        model.train()
        losses = []
        for b in train_loader:
            b = b.to(device)
            opt.zero_grad()
            out, _ = model(b, task_level="graph")
            loss = criterion(out.view(-1), b.y.view(-1).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().item()))
        val_auc = _eval(val_loader)
        test_auc = _eval(test_loader)
        history["train_loss"].append(float(np.mean(losses) if losses else 0.0))
        history["val_auc"].append(val_auc)
        history["test_auc"].append(test_auc)
        if val_auc > best_auc:
            best_auc = val_auc
            best = test_auc
    return {"metric": float(best if best is not None else 0.5), "history": history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", default="data/OGB")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr_raw, val_raw, ts_raw = load_molhiv(root=args.data_root)

    tr = CachedGraphDataset(tr_raw, name="molhiv_train", max_motifs=16, pe_k=16).cached_data
    val = CachedGraphDataset(val_raw, name="molhiv_val", max_motifs=16, pe_k=16).cached_data
    ts = CachedGraphDataset(ts_raw, name="molhiv_test", max_motifs=16, pe_k=16).cached_data
    in_dim = tr[0]["x"].size(1)

    models = {
        "PairwiseGET": PairwiseGET(in_dim, args.hidden_dim, 1),
        "FullGET": FullGET(in_dim, args.hidden_dim, 1, lambda_3=1.0),
        "ETFaithful": ETFaithful(in_dim, args.hidden_dim, 1, pe_k=16, num_steps=6),
    }
    try:
        models["GIN"] = GINBaseline(in_dim, args.hidden_dim, 1)
    except Exception:
        pass

    results = {}
    for name, model in models.items():
        print(f"--- {name} on ogbg-molhiv ---")
        results[name] = _train_binary(model, tr, val, ts, args.epochs, args.batch_size, device)
        print(f"{name} Test ROC-AUC: {results[name]['metric']:.4f}")

    save_results("exp8_molhiv_results", results)


if __name__ == "__main__":
    main()
