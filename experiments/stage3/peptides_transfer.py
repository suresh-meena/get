import argparse
import torch

from get import FullGET, PairwiseGET, GINBaseline, ETFaithful
from get.data import CachedGraphDataset
from experiments.shared.common import add_cached_structural_features, set_seed, GETTrainer, save_results

def load_peptides(task: str, root="data/LRGB"):
    from torch_geometric.datasets import LRGBDataset
    if task not in {"func", "struct"}:
        raise ValueError("task must be one of: func, struct")
    name = "Peptides-func" if task == "func" else "Peptides-struct"
    tr, val, ts = LRGBDataset(root=root, name=name, split="train"), LRGBDataset(root=root, name=name, split="val"), LRGBDataset(root=root, name=name, split="test")
    def _to_dict(ds):
        out = []
        for d in ds:
            x = d.x.float() if d.x is not None else torch.ones(d.num_nodes, 1)
            y = d.y.float().view(-1)
            item = {
                "x": x,
                "edge_index": d.edge_index.to(dtype=torch.long).contiguous(),
                "y": y.unsqueeze(0) if y.ndim == 0 else y.unsqueeze(0) if y.ndim == 1 else y,
            }
            if d.edge_attr is not None:
                item["edge_attr"] = d.edge_attr.float().contiguous()
            out.append(item)
        return out
    return _to_dict(tr), _to_dict(val), _to_dict(ts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["func", "struct"], required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", default="data/LRGB")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr_raw, val_raw, ts_raw = load_peptides(task=args.task, root=args.data_root)

    tr = CachedGraphDataset(tr_raw, name=f"peptides_{args.task}_train", max_motifs=16, pe_k=16, rwse_k=20).cached_data
    val = CachedGraphDataset(val_raw, name=f"peptides_{args.task}_val", max_motifs=16, pe_k=16, rwse_k=20).cached_data
    ts = CachedGraphDataset(ts_raw, name=f"peptides_{args.task}_test", max_motifs=16, pe_k=16, rwse_k=20).cached_data
    in_dim = tr[0]["x"].size(1)
    out_dim = int(tr[0]["y"].numel())
    tr_gin = add_cached_structural_features(tr)
    val_gin = add_cached_structural_features(val)
    ts_gin = add_cached_structural_features(ts)

    models = {
        "PairwiseGET": (PairwiseGET(in_dim, int(args.hidden_dim * 1.73), out_dim, pe_k=16, rwse_k=20), tr, val, ts),
        "FullGET": (FullGET(in_dim, args.hidden_dim, out_dim, pe_k=16, rwse_k=20, lambda_3=1.0), tr, val, ts),
        "ETFaithful": (ETFaithful(in_dim, args.hidden_dim, out_dim, pe_k=16, rwse_k=20, num_steps=6), tr, val, ts),
        "GIN+Struct": (GINBaseline(tr_gin[0]["x"].size(1), args.hidden_dim, out_dim), tr_gin, val_gin, ts_gin),
    }

    results = {}
    task_type = 'multilabel' if args.task == 'func' else 'regression'
    for name, (model, train_data, val_data, test_data) in models.items():
        print(f"--- Training {name} on Peptides-{args.task} ---")
        trainer = GETTrainer(model, task_type=task_type, device=device, model_name=name, lr=1e-3, weight_decay=1e-5)
        res = trainer.run(train_data, val_data, test_data, args.epochs, args.batch_size)
        print(f"{name} Test Metric ({'AP' if args.task == 'func' else 'MAE'}): {res['metric']:.4f}")
        results[name] = res

    save_results(f"exp9_peptides_{args.task}_results", results)

if __name__ == "__main__":
    main()
