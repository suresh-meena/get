from __future__ import annotations

import argparse
from get.data import RealWorldGraphDataset, build_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Precompute and cache graph preprocessing artifacts")
    p.add_argument("--dataset_root", type=str, default="data")
    p.add_argument("--mode", type=str, default="real_world", choices=["real_world", "protocol"])
    p.add_argument("--dataset_name", type=str, default="MUTAG")
    p.add_argument("--task", type=str, default="stage4_tu_classification")
    p.add_argument("--in_dim", type=int, default=32)
    p.add_argument("--max_motifs_per_anchor", type=int, default=8)
    p.add_argument("--task_type", type=str, default="auto")
    p.add_argument("--ego_hops", type=int, default=1)
    p.add_argument("--max_graphs", type=int, default=0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--min_nodes", type=int, default=10)
    p.add_argument("--max_nodes", type=int, default=20)
    p.add_argument("--edge_prob", type=float, default=0.2)
    p.add_argument("--tu_name", type=str, default="MUTAG")
    p.add_argument("--brec_file", type=str, default="")
    p.add_argument("--pos_k", type=int, default=0)
    return p


def run_preprocessing(args: argparse.Namespace) -> None:
    if args.mode == "real_world":
        ds = RealWorldGraphDataset(
            name=args.dataset_name,
            root=args.dataset_root,
            in_dim=args.in_dim,
            max_motifs_per_anchor=args.max_motifs_per_anchor,
            pos_k=args.pos_k,
            task_type=args.task_type,
            cache_enabled=True,
        )
        print(f"Cached real-world dataset {args.dataset_name}: {len(ds)} graphs")
        return

    res = build_dataset(args.task, args)
    if isinstance(res, tuple):
        data, num_classes = res
        if isinstance(data, dict):
            total = sum(len(v) for v in data.values())
        else:
            total = len(data)
        print(f"Cached protocol task {args.task}: {total} graphs, num_classes={num_classes}")
    else:
        print(f"Cached protocol task {args.task}")


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    run_preprocessing(args)
