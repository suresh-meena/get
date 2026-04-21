import argparse
import math
import random

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

from get import FullGET, PairwiseGET, GINBaseline, build_adamw_optimizer
from get.data import collate_get_batch


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


def generate_csl_dataset(samples_per_class=50, seed=0):
    rng = random.Random(seed)
    dataset = []
    ks = [2, 3, 4, 5, 6, 9, 11, 12, 13, 16]
    n = 41
    print("Generating CSL dataset (uniform features)")
    for label, k in enumerate(tqdm(ks, desc="Classes")):
        graph = generate_csl(n, k)
        for _ in range(samples_per_class):
            perm = list(range(n))
            rng.shuffle(perm)
            mapping = {i: perm[i] for i in range(n)}
            graph_perm = nx.relabel_nodes(graph, mapping)
            x = torch.ones(n, 1)
            dataset.append({"x": x, "edges": list(graph_perm.edges()), "y": torch.tensor([label], dtype=torch.long)})
    rng.shuffle(dataset)
    return dataset


def train_and_eval(model_name, model, dataset, epochs=120, batch_size=8, device="cpu", max_grad_norm=0.5, lr=2e-4, weight_decay=1e-4, seed=0):
    model = model.to(device)
    optimizer = build_adamw_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    criterion = nn.CrossEntropyLoss()

    train_size = int(0.8 * len(dataset))
    train_data = dataset[:train_size]
    test_data = dataset[train_size:]
    batch_rng = random.Random(seed)

    def create_batches(data, bs):
        items = list(data)
        batch_rng.shuffle(items)
        return [items[i : i + bs] for i in range(0, len(items), bs)]

    best_acc = 0.0
    best_state = None
    history = {"train_loss": [], "test_acc": [], "grad_norm": [], "bad_batches": []}

    pbar = tqdm(range(epochs), desc=f"Training {model_name}")
    for _ in pbar:
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
            loss = criterion(out, batch.y.squeeze())
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

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_list in create_batches(test_data, batch_size):
                batch = collate_get_batch(batch_list).to(device)
                out, _ = model(batch, task_level="graph")
                preds = out.argmax(dim=-1)
                all_preds.extend(preds.reshape(-1).cpu().numpy().tolist())
                all_labels.extend(batch.y.reshape(-1).cpu().numpy().tolist())

        acc = accuracy_score(all_labels, all_preds)
        scheduler.step(acc)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        avg_grad_norm = total_grad_norm / max(grad_steps, 1)
        avg_loss = total_loss / max(grad_steps, 1)
        pbar.set_postfix({"loss": avg_loss, "test_acc": acc, "grad": avg_grad_norm, "bad": bad_batches, "lr": optimizer.param_groups[0]["lr"]})
        history["train_loss"].append(avg_loss)
        history["test_acc"].append(acc)
        history["grad_norm"].append(avg_grad_norm)
        history["bad_batches"].append(bad_batches)

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_acc, history, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_per_class", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dataset = generate_csl_dataset(samples_per_class=args.samples_per_class, seed=args.seed)

    model_pw = PairwiseGET(in_dim=1, d=64, num_classes=10, num_steps=8, eta=0.01, eta_max=0.05, beta_2=2.0, grad_clip_norm=0.5, state_clip_norm=5.0, beta_max=3.0)
    acc_pw, hist_pw, _ = train_and_eval("PairwiseGET (1-WL)", model_pw, dataset, epochs=args.epochs, batch_size=args.batch_size, device=device, max_grad_norm=0.5, lr=1e-4, seed=args.seed)

    model_gin = GINBaseline(in_dim=1, d=64, num_classes=10, num_layers=4)
    acc_gin, hist_gin, _ = train_and_eval("GIN (1-WL)", model_gin, dataset, epochs=args.epochs, batch_size=args.batch_size, device=device, max_grad_norm=1.0, lr=2e-4, seed=args.seed + 1)

    model_full = FullGET(in_dim=1, d=64, num_classes=10, num_steps=8, R=2, lambda_3=0.35, lambda_m=0.0, beta_2=2.0, beta_3=0.5, eta=0.01, eta_max=0.05, grad_clip_norm=0.5, state_clip_norm=5.0, beta_max=3.0, compile=False)
    acc_full, hist_full, _ = train_and_eval("FullGET (R=2, stable)", model_full, dataset, epochs=args.epochs, batch_size=args.batch_size, device=device, max_grad_norm=0.5, lr=1e-4, seed=args.seed + 2)

    print("\nRESULTS: CSL EXPRESSIVITY")
    print(f"PairwiseGET Accuracy: {acc_pw:.4f}")
    print(f"GIN Baseline Accuracy: {acc_gin:.4f}")
    print(f"FullGET Accuracy:     {acc_full:.4f}")


if __name__ == "__main__":
    main()
