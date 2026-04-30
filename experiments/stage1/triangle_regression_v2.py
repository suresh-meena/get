import argparse
import torch

from get import ETFaithful, FullGET, PairwiseGET, GINBaseline
from experiments.shared.common import set_seed, save_results, split_grouped_dataset
from experiments.stage1.common import generate_true_triangle_regression_dataset, run_model_suite

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_graphs", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_nodes", type=int, default=24)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = generate_true_triangle_regression_dataset(
        num_graphs=args.num_graphs,
        n_nodes=args.n_nodes,
        seed=args.seed,
    )
    
    train_ds, val_ds, test_ds = split_grouped_dataset(dataset, "graph_id", seed=args.seed)

    results = run_model_suite(
        train_ds,
        val_ds,
        test_ds,
        [
            {"name": "FullGET", "model": FullGET(1, 64, 1, num_steps=16, R=2, lambda_3=1.0, update_damping=0.1), "trainer_kwargs": {"lr": 5e-4}, "report_label": "Test MAE"},
            {"name": "ETFaithful", "model": ETFaithful(1, 64, 1, num_steps=16, mask_mode="sparse", et_official_mode=False), "trainer_kwargs": {"lr": 5e-4}, "report_label": "Test MAE"},
            {"name": "PairwiseGET", "model": PairwiseGET(1, 110, 1, num_steps=8), "trainer_kwargs": {"lr": 5e-4}, "report_label": "Test MAE"},
            {"name": "GIN", "model": GINBaseline(1, 64, 1), "trainer_kwargs": {"lr": 5e-4}, "report_label": "Test MAE"},
        ],
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        task_type="regression",
    )

    save_results("exp1_triangle_regression_v2", results)

if __name__ == "__main__":
    main()
