import argparse
import torch

from get.data import CachedGraphDataset
from experiments.shared.common import add_cached_structural_features, set_seed, GETTrainer, save_results
from experiments.shared.model_config import instantiate_models_from_catalog, load_training_defaults

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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", default="data/OGB")
    parser.add_argument("--model_config", default="configs/models/catalog.yaml")
    args = parser.parse_args()
    training_defaults = load_training_defaults(args.model_config)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr_raw, val_raw, ts_raw = load_molhiv(root=args.data_root)

    tr = CachedGraphDataset(tr_raw, name="molhiv_train", max_motifs=16, pe_k=16, rwse_k=20).cached_data
    val = CachedGraphDataset(val_raw, name="molhiv_val", max_motifs=16, pe_k=16, rwse_k=20).cached_data
    ts = CachedGraphDataset(ts_raw, name="molhiv_test", max_motifs=16, pe_k=16, rwse_k=20).cached_data
    in_dim = tr[0]["x"].size(1)
    tr_gin = add_cached_structural_features(tr)
    val_gin = add_cached_structural_features(val)
    ts_gin = add_cached_structural_features(ts)

    models = {
        "PairwiseGET": ("PairwiseGET", tr, val, ts),
        "FullGET": ("FullGET", tr, val, ts),
        "ETFaithful": ("ETFaithful", tr, val, ts),
        "GIN+Struct": ("GIN", tr_gin, val_gin, ts_gin),
    }

    results = {}
    for name, (model_name, train_data, val_data, test_data) in models.items():
        model_context = {
            "in_dim": in_dim,
            "gin_in_dim": tr_gin[0]["x"].size(1),
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
            "rwse_k": 20,
            "et_num_blocks": 8,
            "et_num_heads": 12,
            "et_head_dim_or_none": 64,
            "et_pe_k": 16,
            "et_mask_mode": "sparse",
            "et_official_mode": False,
            "et_node_cap": None,
        }
        model = instantiate_models_from_catalog(args.model_config, context=model_context, names=[model_name])[model_name]
        trainer = GETTrainer(
            model,
            task_type='binary',
            device=device,
            model_name=name,
            lr=1e-4,
            weight_decay=1e-4,
            use_amp=training_defaults.get("use_amp", None),
            amp_dtype=training_defaults.get("amp_dtype", None),
        )
        results[name] = trainer.run(train_data, val_data, test_data, 50, args.batch_size, use_weighted_loss=True)
        print(f"{name} Test ROC-AUC: {results[name]['metric']:.4f}")
        
        # Save incremental results
        save_results("exp8_molhiv_results", results, metadata=vars(args))

if __name__ == "__main__":
    main()
