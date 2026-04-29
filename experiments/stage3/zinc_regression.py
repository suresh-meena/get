import argparse
import torch

from get import FullGET, PairwiseGET, GINBaseline
from get.data import CachedGraphDataset
from experiments.common import set_seed, GETTrainer, save_results


def add_cached_structural_features(dataset):
    augmented = []
    for item in dataset:
        copied = dict(item)
        feats = [copied["x"]]
        if copied.get("pe") is not None:
            feats.append(copied["pe"].to(dtype=copied["x"].dtype))
        if copied.get("rwse") is not None:
            feats.append(copied["rwse"].to(dtype=copied["x"].dtype))
        copied["x"] = torch.cat(feats, dim=-1)
        augmented.append(copied)
    return augmented

def load_zinc_subset(root="data/ZINC"):
    from torch_geometric.datasets import ZINC
    train, val, test = ZINC(root=root, subset=True, split="train"), \
                       ZINC(root=root, subset=True, split="val"), \
                       ZINC(root=root, subset=True, split="test")

    def to_dict(ds):
        return [
            {
                "x": d.x.float(),
                "edge_index": d.edge_index.to(dtype=torch.long).contiguous(),
                "y": d.y,
                "edge_attr": d.edge_attr.float().contiguous().view(-1, 1),
            }
            for d in ds
        ]

    return to_dict(train), to_dict(val), to_dict(test)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--model", choices=["pairwise", "full", "gin"], default="full")
    parser.add_argument("--rwse_k", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_tr, raw_val, raw_ts = load_zinc_subset()
    
    if args.num_samples > 0:
        raw_tr = raw_tr[:args.num_samples]
        raw_val = raw_val[:args.num_samples // 5]
        raw_ts = raw_ts[:args.num_samples // 5]

    tr_ds = CachedGraphDataset(raw_tr, name="zinc_train", max_motifs=16, pe_k=16, rwse_k=args.rwse_k)
    val_ds = CachedGraphDataset(raw_val, name="zinc_val", max_motifs=16, pe_k=16, rwse_k=args.rwse_k)
    ts_ds = CachedGraphDataset(raw_ts, name="zinc_test", max_motifs=16, pe_k=16, rwse_k=args.rwse_k)
    tr_cached, val_cached, ts_cached = tr_ds.cached_data, val_ds.cached_data, ts_ds.cached_data

    in_dim = tr_cached[0]['x'].size(1)
    if args.model == "full":
        model = FullGET(in_dim, args.hidden_dim, 1, num_steps=16, pe_k=16, rwse_k=args.rwse_k, lambda_3=1.0, update_damping=0.1)
        train_data, val_data, test_data = tr_cached, val_cached, ts_cached
    elif args.model == "pairwise":
        model = PairwiseGET(in_dim, int(args.hidden_dim * 1.73), 1, pe_k=16, rwse_k=args.rwse_k)
        train_data, val_data, test_data = tr_cached, val_cached, ts_cached
    else:
        train_data = add_cached_structural_features(tr_cached)
        val_data = add_cached_structural_features(val_cached)
        test_data = add_cached_structural_features(ts_cached)
        model = GINBaseline(train_data[0]['x'].size(1), args.hidden_dim, 1)
        args.model = "gin_struct"

    print(f"\n--- Training {args.model} on ZINC ---")
    trainer = GETTrainer(model, task_type='regression', device=device, lr=5e-4)
    res = trainer.run(train_data, val_data, test_data, args.epochs, args.batch_size)
    print(f"ZINC Test MAE: {res['metric']:.4f}")

    save_results(f"zinc_{args.model}", res)

if __name__ == "__main__":
    main()
