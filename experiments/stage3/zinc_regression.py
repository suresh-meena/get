import argparse
import torch

from get.data import CachedGraphDataset
from experiments.shared.common import add_cached_structural_features, set_seed, GETTrainer, save_results
from experiments.shared.model_config import instantiate_models_from_catalog, load_training_defaults

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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--rwse_k", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_config", default="configs/models/catalog.yaml")
    args = parser.parse_args()
    training_defaults = load_training_defaults(args.model_config)

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
        "ETFaithful": (tr_cached, val_cached, ts_cached),
        "FullGET": (tr_cached, val_cached, ts_cached),
        "PairwiseGET": (tr_cached, val_cached, ts_cached),
        "GIN": (add_cached_structural_features(tr_cached), add_cached_structural_features(val_cached), add_cached_structural_features(ts_cached)),
    }

    results = {}
    for name, (train_data, val_data, test_data) in models_to_run.items():
        print(f"\n--- Training {name} on ZINC ---")
        model_context = {
            "in_dim": in_dim,
            "gin_in_dim": add_cached_structural_features(tr_cached)[0]["x"].size(1),
            "num_classes": 1,
            "hidden_dim": args.hidden_dim,
            "pairwise_hidden_dim": int(args.hidden_dim * 1.73),
            "num_steps": 1,
            "get_num_heads": 12,
            "get_num_blocks": 8,
            "lambda_3": 1.0,
            "get_norm_style": "et",
            "get_pairwise_et_mask": False,
            "get_pe_k": 16,
            "rwse_k": args.rwse_k,
            "et_num_blocks": 8,
            "et_num_heads": 12,
            "et_head_dim_or_none": 64,
            "et_pe_k": 16,
            "et_mask_mode": "sparse",
            "et_official_mode": False,
            "et_node_cap": None,
        }
        model = instantiate_models_from_catalog(args.model_config, context=model_context, names=[name])[name]
        trainer = GETTrainer(
            model,
            task_type='regression',
            device=device,
            model_name=name,
            lr=5e-5,
            weight_decay=1e-4,
            use_amp=training_defaults.get("use_amp", None),
            amp_dtype=training_defaults.get("amp_dtype", None),
        )
        results[name] = trainer.run(train_data, val_data, test_data, 50, args.batch_size)
        print(f"{name} Test MAE: {results[name]['metric']:.4f}")
        
        # Save incremental results
        save_results("exp10_zinc_results", results, metadata=vars(args))

if __name__ == "__main__":
    main()
