import argparse
import torch
import json
from pathlib import Path

from get import FullGET, PairwiseGET, GINBaseline, ETFaithful
from get.data import CachedGraphDataset
from experiments.shared.common import add_cached_structural_features, set_seed, GETTrainer, save_results

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
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--rwse_k", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
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
    
    models_to_run = {
        "ETFaithful": lambda: ETFaithful(in_dim, args.hidden_dim, 1, num_blocks=8, num_heads=12, head_dim=64, pe_k=15, rwse_k=args.rwse_k, eta=0.1, K=args.hidden_dim*4, mask_mode="sparse", et_official_mode=False),
        "FullGET": lambda: FullGET(in_dim, args.hidden_dim, 1, num_blocks=8, num_steps=1, num_heads=12, pe_k=16, rwse_k=args.rwse_k, lambda_3=1.0, beta_3=1.2, update_damping=0.1),
        "PairwiseGET": lambda: PairwiseGET(in_dim, int(args.hidden_dim * 1.73), 1, pe_k=16, rwse_k=args.rwse_k, lambda_2=3.0, beta_2=1.5),
        "GIN": lambda: GINBaseline(add_cached_structural_features(tr_cached)[0]['x'].size(1), args.hidden_dim, 1)
    }

    results = {}
    for name, model_fn in models_to_run.items():
        print(f"\n--- Training {name} on ZINC ---")
        if name == "GIN":
            train_data = add_cached_structural_features(tr_cached)
            val_data = add_cached_structural_features(val_cached)
            test_data = add_cached_structural_features(ts_cached)
        else:
            train_data, val_data, test_data = tr_cached, val_cached, ts_cached
            
        model = model_fn()
        trainer = GETTrainer(model_fn(), task_type='regression', device=device, model_name=name, lr=1e-4, weight_decay=1e-4)
        res = trainer.run(train_data, val_data, test_data, args.epochs, args.batch_size)
        results[name] = res
        print(f"ZINC {name} Test MAE: {res['metric']:.4f}")

    save_results("zinc_results", results)

if __name__ == "__main__":
    main()
