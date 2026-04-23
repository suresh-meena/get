import argparse
import torch
import numpy as np

from get import FullGET, PairwiseGET, GINBaseline
from get.data import CachedGraphDataset
from experiments.common import set_seed, GETTrainer, save_results

def load_zinc_subset(root="data/ZINC"):
    from torch_geometric.datasets import ZINC
    train, val, test = ZINC(root=root, subset=True, split="train"), \
                       ZINC(root=root, subset=True, split="val"), \
                       ZINC(root=root, subset=True, split="test")
    def to_dict(ds): return [{"x": d.x.float(), "edges": list(zip(d.edge_index[0].tolist(), d.edge_index[1].tolist())), "y": d.y, "edge_attr": d.edge_attr.float().view(-1, 1)} for d in ds]
    return to_dict(train), to_dict(val), to_dict(test)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--model", choices=["pairwise", "full", "gin"], default="full")
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_tr, raw_val, raw_ts = load_zinc_subset()
    
    tr_ds = CachedGraphDataset(raw_tr, name="zinc_train", max_motifs=16, pe_k=16)
    val_ds = CachedGraphDataset(raw_val, name="zinc_val", max_motifs=16, pe_k=16)
    ts_ds = CachedGraphDataset(raw_ts, name="zinc_test", max_motifs=16, pe_k=16)

    in_dim = tr_ds[0]['x'].size(1)
    if args.model == "full":
        model = FullGET(in_dim, args.hidden_dim, 1, pe_k=16, lambda_3=1.0)
    elif args.model == "pairwise":
        model = PairwiseGET(in_dim, int(args.hidden_dim * 1.73), 1, pe_k=16)
    else:
        model = GINBaseline(in_dim, args.hidden_dim, 1)

    print(f"\n--- Training {args.model} on ZINC ---")
    trainer = GETTrainer(model, task_type='regression', device=device, lr=1e-3)
    res = trainer.run(tr_ds, val_ds, ts_ds, args.epochs, 256)
    print(f"ZINC Test MAE: {res['metric']:.4f}")

    save_results(f"zinc_{args.model}_results", res)

if __name__ == "__main__":
    main()
