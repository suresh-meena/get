import argparse
import torch
from get.models.get_model import GETModel
from experiments.shared.common import GETTrainer, set_seed, split_grouped_dataset
from experiments.stage1.wedge_discrimination import generate_matched_dataset

def run_linear_ablation(gpu_id, variant):
    device = f"cuda:{gpu_id}"
    set_seed(42)
    dataset = generate_matched_dataset(num_pairs=2000, rwse_k=8)
    train_data, val_data, test_data = split_grouped_dataset(dataset, "pair_id", seed=42)
    in_dim = train_data[0]['x'].size(1)
    
    if variant == "get_linear_sum_rwse":
        model = GETModel(in_dim, 128, 1, num_steps=8, lambda_3=1.0, update_damping=0.05, lambda_sum=1.0, rwse_k=8)
        name = "GET_LinearSum_RWSE8"
    elif variant == "get_deep":
        # More steps and more damping to move further in state space
        model = GETModel(in_dim, 128, 1, num_steps=32, lambda_3=1.0, update_damping=0.1, lambda_sum=1.0, rwse_k=8)
        name = "GET_Deep32_D0.1"
    elif variant == "get_no_readout_norm":
        model = GETModel(in_dim, 128, 1, num_steps=8, lambda_3=1.0, update_damping=0.05, lambda_sum=1.0, rwse_k=8)
        # Monkey patch readout to remove LayerNorm
        import torch.nn as nn
        model.readout = nn.Sequential(
            nn.Linear(4 * 128, 2 * 128), nn.GELU(),
            nn.Linear(2 * 128, 128), nn.GELU(),
            nn.Linear(128, 1)
        )
        name = "GET_NoReadoutNorm"
    else:
        raise ValueError(f"Unknown variant {variant}")

    trainer = GETTrainer(model.to(device), task_type='binary', device=device, model_name=name, lr=5e-4)
    res = trainer.run(train_data, val_data, test_data, epochs=50, batch_size=32)
    print(f"VARIANT: {name} | Test AUC: {res['metric']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--variant", type=str, required=True)
    args = parser.parse_args()
    run_linear_ablation(args.gpu, args.variant)
