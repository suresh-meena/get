import argparse
import torch

from get import FullGET, PairwiseGET, GINBaseline, ETFaithful
from get.data import CachedGraphDataset
from experiments.common import set_seed, GETTrainer, save_results

def load_molhiv(root="data/OGB"):
    try:
        from ogb.graphproppred import PygGraphPropPredDataset
    except Exception as e:
        raise RuntimeError("ogb is required for molhiv. Install `ogb`.") from e

    # PyTorch 2.6 defaults torch.load(weights_only=True), but OGB's cached
    # processed dataset uses torch_geometric objects that require the legacy
    # unpickling path. Temporarily force weights_only=False for this load.
    original_torch_load = torch.load
    def _torch_load_legacy(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)
    torch.load = _torch_load_legacy
    try:
        ds = PygGraphPropPredDataset(name="ogbg-molhiv", root=root)
    finally:
        torch.load = original_torch_load

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
                "edge_index": d.edge_index.to(dtype=torch.long).contiguous(),
                "y": torch.tensor([float(y[0].item())], dtype=torch.float32),
            }
            if edge_attr is not None:
                item["edge_attr"] = edge_attr
            out.append(item)
        return out
    return _convert(split["train"]), _convert(split["valid"]), _convert(split["test"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", default="data/OGB")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr_raw, val_raw, ts_raw = load_molhiv(root=args.data_root)

    tr = CachedGraphDataset(tr_raw, name="molhiv_train", max_motifs=16, pe_k=16, rwse_k=20).cached_data
    val = CachedGraphDataset(val_raw, name="molhiv_val", max_motifs=16, pe_k=16, rwse_k=20).cached_data
    ts = CachedGraphDataset(ts_raw, name="molhiv_test", max_motifs=16, pe_k=16, rwse_k=20).cached_data
    in_dim = tr[0]["x"].size(1)

    models = {
        "PairwiseGET": PairwiseGET(in_dim, int(args.hidden_dim * 1.73), 1, pe_k=16, rwse_k=20),
        "FullGET": FullGET(in_dim, args.hidden_dim, 1, pe_k=16, rwse_k=20, lambda_3=1.0),
        "ETFaithful": ETFaithful(in_dim, args.hidden_dim, 1, pe_k=16, rwse_k=20, num_steps=6),
        "GIN": GINBaseline(in_dim, args.hidden_dim, 1)
    }

    results = {}
    for name, model in models.items():
        trainer = GETTrainer(model, task_type='binary', device=device, model_name=name, lr=1e-3, weight_decay=1e-5)
        results[name] = trainer.run(tr, val, ts, args.epochs, args.batch_size, use_weighted_loss=True)
        print(f"{name} Test ROC-AUC: {results[name]['metric']:.4f}")

    save_results("exp8_molhiv_results", results)

if __name__ == "__main__":
    main()
