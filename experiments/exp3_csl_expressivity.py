import argparse
import json
import math
import random
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from get import FullGET, GINBaseline, PairwiseGET, build_adamw_optimizer
from get.compile_utils import maybe_compile_model
from get.data import collate_get_batch


CSL_CLASSES = [2, 3, 4, 5, 6, 9, 11, 12, 13, 16]


def generate_csl(n=41, k=2):
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        graph.add_edge(i, (i + 1) % n)
        graph.add_edge(i, (i + k) % n)
    return graph


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_csl_dataset(graphs_per_class=15, seed=0):
    rng = random.Random(seed)
    dataset = []
    labels = []
    n = 41
    print("Generating CSL dataset (official 150-graph protocol)")
    for label, k in enumerate(tqdm(CSL_CLASSES, desc="Classes")):
        graph = generate_csl(n, k)
        for _ in range(graphs_per_class):
            perm = list(range(n))
            rng.shuffle(perm)
            mapping = {i: perm[i] for i in range(n)}
            graph_perm = nx.relabel_nodes(graph, mapping)
            x = torch.ones(n, 1)
            dataset.append(
                {
                    "x": x,
                    "edges": list(graph_perm.edges()),
                    "y": torch.tensor([label], dtype=torch.long),
                }
            )
            labels.append(label)
    rng.shuffle(dataset)
    labels = np.array([int(g["y"].item()) for g in dataset], dtype=np.int64)
    return dataset, labels


def build_csl_folds(labels, seed=0):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_indices = [test_idx for _, test_idx in skf.split(np.zeros(len(labels)), labels)]
    splits = []
    for test_fold in range(5):
        val_fold = (test_fold + 1) % 5
        train_folds = [fold_indices[i] for i in range(5) if i not in {test_fold, val_fold}]
        train_idx = np.concatenate(train_folds)
        val_idx = fold_indices[val_fold]
        test_idx = fold_indices[test_fold]
        splits.append((train_idx, val_idx, test_idx))
    return splits


def slice_dataset(dataset, indices):
    return [dataset[i] for i in indices]


def evaluate_accuracy(model, data, batch_size, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch_list = data[start : start + batch_size]
            batch = collate_get_batch(batch_list).to(device)
            out, _ = model(batch, task_level="graph")
            preds = out.argmax(dim=-1)
            all_preds.extend(preds.reshape(-1).cpu().numpy().tolist())
            all_labels.extend(batch.y.reshape(-1).cpu().numpy().tolist())
    return accuracy_score(all_labels, all_preds)


def train_one_fold(
    model_name,
    model,
    train_data,
    val_data,
    test_data,
    epochs=120,
    batch_size=8,
    device="cpu",
    max_grad_norm=0.5,
    lr=2e-4,
    weight_decay=1e-4,
    patience=25,
    compile_model=False,
):
    model = model.to(device)
    model = maybe_compile_model(model, compile_model, model_name=model_name)
    optimizer = build_adamw_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = None
    best_epoch = -1
    bad_epochs = 0
    history = {"train_loss": [], "val_acc": [], "grad_norm": [], "bad_batches": []}

    def create_batches(data, bs):
        items = list(data)
        random.shuffle(items)
        return [items[i : i + bs] for i in range(0, len(items), bs)]

    pbar = tqdm(range(epochs), desc=f"Training {model_name}", leave=False)
    for epoch in pbar:
        model.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        grad_steps = 0
        bad_batches = 0

        for batch_list in create_batches(train_data, batch_size):
            batch = collate_get_batch(batch_list).to(device)
            optimizer.zero_grad()
            out, _ = model(batch, task_level="graph")
            if not torch.isfinite(out).all():
                bad_batches += 1
                continue
            loss = criterion(out, batch.y.view(-1))
            if not torch.isfinite(loss):
                bad_batches += 1
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if not math.isfinite(float(grad_norm)):
                optimizer.zero_grad(set_to_none=True)
                bad_batches += 1
                continue
            optimizer.step()
            total_loss += float(loss.item())
            total_grad_norm += float(grad_norm)
            grad_steps += 1

        val_acc = evaluate_accuracy(model, val_data, batch_size, device)
        scheduler.step(val_acc)

        if val_acc > best_val + 1e-8:
            best_val = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        avg_grad_norm = total_grad_norm / max(grad_steps, 1)
        avg_loss = total_loss / max(grad_steps, 1)
        pbar.set_postfix(
            {
                "loss": avg_loss,
                "val_acc": val_acc,
                "grad": avg_grad_norm,
                "bad": bad_batches,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)
        history["grad_norm"].append(avg_grad_norm)
        history["bad_batches"].append(bad_batches)

        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc = evaluate_accuracy(model, test_data, batch_size, device)
    return {
        "best_val_acc": best_val,
        "best_epoch": best_epoch,
        "test_acc": test_acc,
        "history": history,
    }


def aggregate_metrics(fold_results):
    test_accs = np.array([r["test_acc"] for r in fold_results], dtype=np.float64)
    val_accs = np.array([r["best_val_acc"] for r in fold_results], dtype=np.float64)
    return {
        "mean_test_acc": float(test_accs.mean()),
        "std_test_acc": float(test_accs.std(ddof=1) if len(test_accs) > 1 else 0.0),
        "mean_val_acc": float(val_accs.mean()),
        "std_val_acc": float(val_accs.std(ddof=1) if len(val_accs) > 1 else 0.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_per_class", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for supported models during training.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dataset, labels = generate_csl_dataset(graphs_per_class=args.graphs_per_class, seed=args.seed)
    splits = build_csl_folds(labels, seed=args.seed)

    def make_pairwise():
        return PairwiseGET(
            in_dim=1,
            d=64,
            num_classes=10,
            num_steps=8,
            eta=0.01,
            eta_max=0.05,
            beta_2=2.0,
            grad_clip_norm=0.5,
            state_clip_norm=5.0,
            beta_max=3.0,
        )

    def make_gin():
        return GINBaseline(in_dim=1, d=64, num_classes=10, num_layers=4)

    def make_full():
        return FullGET(
            in_dim=1,
            d=64,
            num_classes=10,
            num_steps=8,
            R=2,
            lambda_3=0.35,
            lambda_m=0.0,
            beta_2=2.0,
            beta_3=0.5,
            eta=0.01,
            eta_max=0.05,
            grad_clip_norm=0.5,
            state_clip_norm=5.0,
            beta_max=3.0,
            compile=False,
        )

    model_specs = [
        ("PairwiseGET (1-WL)", make_pairwise, 0.5, 1e-4),
        ("GIN (1-WL)", make_gin, 1.0, 2e-4),
        ("FullGET (R=2, stable)", make_full, 0.5, 1e-4),
    ]

    all_results = {}
    fold_summaries = []

    for model_name, factory, max_grad_norm, lr in model_specs:
        fold_results = []
        print(f"\nRunning 5-fold CSL CV for {model_name}")
        for fold_id, (train_idx, val_idx, test_idx) in enumerate(splits):
            set_seed(args.seed + fold_id)
            train_data = slice_dataset(dataset, train_idx)
            val_data = slice_dataset(dataset, val_idx)
            test_data = slice_dataset(dataset, test_idx)
            model = factory()
            result = train_one_fold(
                model_name=f"{model_name} [fold {fold_id + 1}/5]",
                model=model,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                max_grad_norm=max_grad_norm,
                lr=lr,
                patience=25,
                compile_model=args.compile,
            )
            result["fold"] = fold_id + 1
            fold_results.append(result)
            print(
                f"Fold {fold_id + 1}: val_acc={result['best_val_acc']:.4f}, "
                f"test_acc={result['test_acc']:.4f}, best_epoch={result['best_epoch']}"
            )

        metrics = aggregate_metrics(fold_results)
        all_results[model_name] = {"folds": fold_results, "summary": metrics}
        fold_summaries.append(
            {
                "model": model_name,
                "test_accs": [r["test_acc"] for r in fold_results],
                "mean_test_acc": metrics["mean_test_acc"],
                "std_test_acc": metrics["std_test_acc"],
            }
        )

    print("\nRESULTS: CSL EXPRESSIVITY (5-FOLD CV)")
    for model_name, result in all_results.items():
        summary = result["summary"]
        print(
            f"{model_name}: test_acc={summary['mean_test_acc']:.4f} ± {summary['std_test_acc']:.4f}, "
            f"val_acc={summary['mean_val_acc']:.4f} ± {summary['std_val_acc']:.4f}"
        )

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "exp3_csl_expressivity_cv.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    xs = np.arange(len(model_specs))
    means = [all_results[name]["summary"]["mean_test_acc"] for name, _, _, _ in model_specs]
    stds = [all_results[name]["summary"]["std_test_acc"] for name, _, _, _ in model_specs]
    labels_text = [name.split(" ")[0] for name, _, _, _ in model_specs]
    plt.bar(xs, means, yerr=stds, capsize=4)
    plt.xticks(xs, labels_text)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("CSL 5-Fold CV Test Accuracy")
    plt.tight_layout()

    figure_path = output_dir / "exp3_csl_expressivity.png"
    plt.savefig(figure_path)
    print(f"Plot saved to {figure_path}")


if __name__ == "__main__":
    main()
