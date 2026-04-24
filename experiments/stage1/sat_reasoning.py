import argparse
import random
import torch
from tqdm.auto import tqdm
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from get import FullGET, PairwiseGET, collate_get_batch  # noqa: E402
from experiments.common import set_seed, build_dataloader_kwargs, get_num_params  # noqa: E402

def generate_random_3cnf(num_vars=20, num_clauses=80, seed=42):
    rng = random.Random(seed)
    clauses = []
    for _ in range(num_clauses):
        # Sample 3 distinct variables
        vars = rng.sample(range(num_vars), 3)
        # Sample signs (+1 or -1)
        signs = [rng.choice([-1, 1]) for _ in range(3)]
        clauses.append(list(zip(vars, signs)))
    return num_vars, clauses

def build_sat_factor_graph(num_vars, clauses, xor=False):
    num_clauses = len(clauses)
    num_nodes = num_vars + num_clauses
    
    edges = []
    c_3, u_3, v_3, t_tau = [], [], [], []
    
    # Motif type map
    def get_motif_type(s1, s2, s3, r, s):
        # 8 possible sign combinations, 3 possible pairs => 24 types
        s_tuple = tuple((1 if s > 0 else 0) for s in [s1, s2, s3])
        sign_idx = s_tuple[0] * 4 + s_tuple[1] * 2 + s_tuple[2]
        pair_idx = {(0,1): 0, (0,2): 1, (1,2): 2}[(r, s)]
        return sign_idx * 3 + pair_idx

    var_deg_pos = [0] * num_vars
    var_deg_neg = [0] * num_vars
    
    for a, clause in enumerate(clauses):
        c_idx = num_vars + a
        for r, (v_idx, sign) in enumerate(clause):
            edges.append((v_idx, c_idx)) # undirected handled by GET logic
            if sign > 0:
                var_deg_pos[v_idx] += 1
            else:
                var_deg_neg[v_idx] += 1
        
        # 3 wedges per clause
        s1, s2, s3 = [s for v, s in clause]
        for r, s in [(0,1), (0,2), (1,2)]:
            u = clause[r][0]
            v = clause[s][0]
            c_3.append(c_idx)
            u_3.append(u)
            v_3.append(v)
            t_tau.append(get_motif_type(s1, s2, s3, r, s))

    x = torch.zeros(num_nodes, 4)
    # Vars: type 0
    for i in range(num_vars):
        x[i, 0] = 0.0 # type
        x[i, 1] = var_deg_pos[i]
        x[i, 2] = var_deg_neg[i]
        x[i, 3] = 0.0
    
    # Clauses: type 1
    for a, clause in enumerate(clauses):
        c_idx = num_vars + a
        pos_cnt = sum(1 for v, s in clause if s > 0)
        neg_cnt = sum(1 for v, s in clause if s < 0)
        x[c_idx, 0] = 1.0 # type
        x[c_idx, 1] = 3.0 # fixed degree
        x[c_idx, 2] = pos_cnt
        x[c_idx, 3] = neg_cnt

    # Store clause definitions in the graph dict for loss computation
    clause_tensor = torch.zeros(num_clauses, 3, 2, dtype=torch.long)
    for a, clause in enumerate(clauses):
        for r, (v, s) in enumerate(clause):
            clause_tensor[a, r, 0] = v
            clause_tensor[a, r, 1] = 1 if s > 0 else 0

    parity_bits = torch.randint(0, 2, (num_clauses,), dtype=torch.float32) if xor else torch.zeros(num_clauses)
            
    return {
        "x": x,
        "edges": edges,
        "c_3": torch.tensor(c_3, dtype=torch.long),
        "u_3": torch.tensor(u_3, dtype=torch.long),
        "v_3": torch.tensor(v_3, dtype=torch.long),
        "t_tau": torch.tensor(t_tau, dtype=torch.long),
        "clauses": clause_tensor,
        "xor_parity": parity_bits,
        "num_vars": num_vars,
    }

def generate_dataset(num_graphs, num_vars=20, num_clauses=80, xor=False, seed=42):
    rng = random.Random(seed)
    dataset = []
    for i in range(num_graphs):
        nv, clauses = generate_random_3cnf(num_vars, num_clauses, seed=rng.randint(0, 1000000))
        g = build_sat_factor_graph(nv, clauses, xor=xor)
        dataset.append(g)
    return dataset

def compute_loss(logits, batch, xor=False, eps=1e-5):
    total_loss = 0.0
    for g_idx in range(len(batch.ptr) - 1):
        n_vars = batch.num_vars[g_idx]
        start = batch.ptr[g_idx]
        var_logits = logits[start : start + n_vars].view(-1)
        p = torch.sigmoid(var_logits)
        c_tensor = batch.clauses[g_idx] 
        v_idx = c_tensor[:, :, 0].to(device=logits.device)
        s_val = c_tensor[:, :, 1].to(device=logits.device)
        p_v = p[v_idx]
        pi = torch.where(s_val == 1, p_v, 1.0 - p_v)
        if xor:
            m = 2 * p_v - 1
            parity = batch.xor_parity[g_idx].to(device=logits.device)
            sign_prod = torch.prod(torch.where(s_val == 1, m, -m), dim=1)
            parity_sign = torch.where(parity == 1.0, -1.0, 1.0)
            s_hat = 0.5 * (1.0 + parity_sign * sign_prod)
        else:
            s_hat = 1.0 - torch.prod(1.0 - pi, dim=1)
        loss = -torch.log(s_hat + eps).mean()
        total_loss += loss
    return total_loss / max(1, len(batch.ptr) - 1)

def custom_collate(graph_list):
    batch = collate_get_batch(graph_list, max_motifs=None)
    batch.clauses = [g['clauses'] for g in graph_list]
    batch.xor_parity = [g['xor_parity'] for g in graph_list]
    batch.num_vars = [g['num_vars'] for g in graph_list]
    return batch

def compute_metrics(logits, batch, xor=False):
    satisfied_ratios = []
    solved_counts = 0
    for g_idx in range(len(batch.ptr) - 1):
        n_vars = batch.num_vars[g_idx]
        start = batch.ptr[g_idx]
        var_logits = logits[start : start + n_vars].view(-1)
        m = (var_logits >= 0).float() * 2.0 - 1.0 
        p = (var_logits >= 0).float() 
        c_tensor = batch.clauses[g_idx] 
        v_idx = c_tensor[:, :, 0] 
        s_val = c_tensor[:, :, 1] 
        p_v = p[v_idx]
        lit = torch.where(s_val == 1, p_v, 1.0 - p_v)
        if xor:
            m_v = m[v_idx]
            parity = batch.xor_parity[g_idx]
            sign_prod = torch.prod(torch.where(s_val == 1, m_v, -m_v), dim=1)
            parity_sign = torch.where(parity == 1.0, -1.0, 1.0)
            satisfied = (parity_sign * sign_prod > 0).float()
        else:
            satisfied = (lit.sum(dim=1) > 0).float()
        ratio = satisfied.mean().item()
        satisfied_ratios.append(ratio)
        if ratio >= 1.0 - 1e-9:
            solved_counts += 1
    return np.mean(satisfied_ratios), solved_counts / max(1, len(batch.ptr) - 1)

def train(model, train_loader, val_loader, test_loader, epochs, device, model_name=None, xor=False):
    import time
    model = model.to(device)
    model_name = model_name or model.__class__.__name__
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    print("-" * 50)
    print(f"EXPERIMENT: {model_name} on {'3-XORSAT' if xor else 'Max-3-SAT'}")
    print(f"DEVICE:     {device}")
    print(f"PARAMS:     {get_num_params(model)}")
    if hasattr(model, 'get_layer'):
        layer = model.get_layer
        print(f"CONFIG:     d={layer.d}, H={layer.num_heads}, steps={model.num_steps}")
        if hasattr(layer, 'R'):
            print(f"            R={layer.R}, K={layer.K}")
    print("-" * 50)

    best_val_loss = float('inf')
    best_test_metrics = (float('inf'), 0.0, 0.0) 
    param_cnt = get_num_params(model)
    pbar = tqdm(range(epochs), desc=f"Train {model_name} [{param_cnt}]", bar_format='{l_bar}{bar:20}{r_bar}', leave=False)
    
    for epoch in pbar:
        t0 = time.time()
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits, _ = model(batch, task_level="node")
            loss = compute_loss(logits, batch, xor=xor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits, _ = model(batch, task_level="node")
                loss = compute_loss(logits, batch, xor=xor)
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))
        epoch_time = time.time() - t0
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            test_loss, all_logits, all_batches = 0, [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    logits, _ = model(batch, task_level="node")
                    test_loss += compute_loss(logits, batch, xor=xor).item()
                    all_logits.append(logits.cpu())
                    all_batches.append(batch)
            test_loss /= max(1, len(test_loader))
            ratios, solved_list = [], []
            for l_val, b in zip(all_logits, all_batches):
                r, s = compute_metrics(l_val, b, xor=xor)
                ratios.append(r)
                solved_list.append(s)
            best_test_metrics = (test_loss, np.mean(ratios), np.mean(solved_list))
        metrics_str = f"L: {train_loss/max(1, len(train_loader)):.4f} | V: {val_loss:.4f} | B: {best_test_metrics[0]:.4f} | {epoch_time:.1f}s/ep"
        pbar.set_postfix_str(metrics_str)
    return best_test_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_graphs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--xor", action="store_true")
    args = parser.parse_args()
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = generate_dataset(args.num_graphs, num_vars=20, num_clauses=80, xor=args.xor)
    split1, split2 = int(0.6 * args.num_graphs), int(0.8 * args.num_graphs)
    train_ds, val_ds, test_ds = dataset[:split1], dataset[split1:split2], dataset[split2:]
    from torch.utils.data import DataLoader
    loader_kwargs = build_dataloader_kwargs(device)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=custom_collate, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=custom_collate, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=custom_collate, **loader_kwargs)
    models = {
        "PairwiseGET": PairwiseGET(4, int(args.hidden_dim * 1.73), 1, num_steps=8),
        "FullGET": FullGET(4, args.hidden_dim, 1, num_steps=8, R=2, lambda_3=1.0, lambda_m=1.0, num_motif_types=24),
    }
    for name, model in models.items():
        test_loss, clause_ratio, solved_ratio = train(model, train_loader, val_loader, test_loader, args.epochs, device, model_name=name, xor=args.xor)
        print(f"{name} Test Loss: {test_loss:.4f} | Satisfied Clause Ratio: {clause_ratio:.4f} | Solved Accuracy: {solved_ratio:.4f}")

if __name__ == "__main__":
    main()
