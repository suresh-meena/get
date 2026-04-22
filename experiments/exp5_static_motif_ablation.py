import argparse
import torch
import numpy as np

from get import FullGET, PairwiseGET
from get.data import add_structural_node_features
from experiments.stage1.wedge_discrimination import generate_matched_dataset
from experiments.common import set_seed, GETTrainer, save_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pairs", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=96)
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_ds = generate_matched_dataset(num_pairs=args.num_pairs)
    
    # Split: 80/20
    split = int(0.8 * len(raw_ds))
    train_raw, test_raw = raw_ds[:split], raw_ds[split:]
    
    # 1. Base Pairwise (No motif)
    print("\n--- 1. Pairwise GET (No Motif) ---")
    pw_res = GETTrainer(PairwiseGET(1, args.hidden_dim, 1), device=device).run(train_raw, test_raw, test_raw, args.epochs, 32)
    
    # 2. Pairwise + Static features
    print("\n--- 2. Pairwise GET + Static Motif Features ---")
    tr_static = [add_structural_node_features(g, include_degree=False, include_motif_counts=True) for g in train_raw]
    ts_static = [add_structural_node_features(g, include_degree=False, include_motif_counts=True) for g in test_raw]
    st_res = GETTrainer(PairwiseGET(3, args.hidden_dim, 1), device=device).run(tr_static, ts_static, ts_static, args.epochs, 32)
    
    # 3. Dynamic Energy
    print("\n--- 3. Full GET (Dynamic Energy) ---")
    dyn_res = GETTrainer(FullGET(1, args.hidden_dim, 1, lambda_3=1.0), device=device).run(train_raw, test_raw, test_raw, args.epochs, 32)

    results = {"pairwise": pw_res, "static": st_res, "dynamic": dyn_res}
    save_results("exp5_motif_ablation_results", results)

if __name__ == "__main__":
    main()
