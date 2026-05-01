import argparse
import torch

from get import ETFaithful, PairwiseGET, FullGET, GINBaseline
from experiments.shared.common import save_results, set_seed, split_grouped_dataset
from experiments.stage1.common import generate_cycle_parity_dataset, generate_srg_dataset, run_model_suite

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="srg", choices=["srg", "cycle"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_pairs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pe_k", type=int, default=0)
    parser.add_argument("--inject_feature_noise", type=float, default=0.0, help="Std of Gaussian noise added to node features.")
    args = parser.parse_args()

    if args.pe_k > 0:
        print(f"Using positional encodings with pe_k={args.pe_k}.")
    else:
        print("Running without positional encodings (default Stage-1 setting).")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"--- Generating {args.task.upper()} Dataset ---")
    if args.task == "srg":
        full_ds = generate_srg_dataset(args.num_pairs, seed=args.seed)
    else:
        full_ds = generate_cycle_parity_dataset(args.num_pairs, n=20, seed=args.seed)

    train_ds, val_ds, test_ds = split_grouped_dataset(full_ds, "graph_id", seed=args.seed)
    print(f"Split sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    
    if args.inject_feature_noise > 0.0:
        print(f"Injecting node feature noise std={args.inject_feature_noise}.")
        for d in train_ds + val_ds + test_ds:
            d["x"] = torch.ones_like(d["x"]) + torch.randn_like(d["x"]) * float(args.inject_feature_noise)

    in_dim = 1
    pe_k = args.pe_k
    results = run_model_suite(
        train_ds,
        val_ds,
        test_ds,
        [
            {
                "name": "PairwiseGET",
                "model": PairwiseGET(in_dim, 128, 1, num_steps=16, pe_k=pe_k, lambda_2=10.0, update_damping=0.05, grad_clip_norm=0.1),
                "trainer_kwargs": {"lr": 5e-4, "max_grad_val": 0.05, "weight_decay": 1e-4},
                "report_label": "Test AUC",
            },
            {
                "name": "FullGET",
                "model": FullGET(in_dim, 128, 1, num_steps=16, pe_k=pe_k, lambda_3=10.0, beta_3=5.0, update_damping=0.05, grad_clip_norm=0.1),
                "trainer_kwargs": {"lr": 5e-4, "max_grad_val": 0.05, "weight_decay": 1e-4},
                "report_label": "Test AUC",
            },
            {
                "name": "ETFaithful",
                "model": ETFaithful(in_dim, 128, 1, num_steps=16, pe_k=pe_k, mask_mode="sparse", et_official_mode=False),
                "trainer_kwargs": {"lr": 5e-4, "max_grad_val": 0.05, "weight_decay": 1e-4},
                "report_label": "Test AUC",
            },
            {
                "name": "GIN",
                "model": GINBaseline(in_dim, 128, 1, num_layers=4),
                "trainer_kwargs": {"lr": 5e-4, "max_grad_val": 0.05, "weight_decay": 1e-4},
                "report_label": "Test AUC",
            },
        ],
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        task_type="binary",
    )

    metadata = {
        "interpretation": "Supervised structural diagnostic. SRG and cycle parity are evidence of realized discrimination only; they are not formal proofs of WL expressivity.",
        "task": args.task,
        "num_pairs": args.num_pairs,
        "seed": args.seed,
        "pe_k": args.pe_k,
        "inject_feature_noise": args.inject_feature_noise,
    }
    save_results(f"exp1_{args.task}_bench", results, metadata=metadata)

if __name__ == "__main__":
    main()
