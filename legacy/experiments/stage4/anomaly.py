from __future__ import annotations

import argparse
import statistics

from tqdm.auto import tqdm

from experiments.shared.common import (
    build_anomaly_protocol_split,
    build_dataloader_kwargs,
    build_ego_graph_dataset,
    set_seed,
)
from .shared import (
    _build_graph_anomaly_models,
    _make_anomaly_batch_samplers,
    _mean_std,
    _maybe_cache_ego_dataset,
    _normalized_name,
    _prepare_get_cached_dataset,
    _resolve_stage4_num_workers,
    _recommend_anomaly_batch_size,
    _recommend_anomaly_motif_cap,
    _load_anomaly_graph,
    _train_graph_binary_with_val,
    make_synth_anomaly_dataset,
)


def run_graph_anomaly(args: argparse.Namespace) -> dict:
    get_pe_k = max(0, int(args.get_pe_k))
    cache_pe_k = max(get_pe_k, max(0, int(args.et_pe_k)))
    prebuilt_ds: list[dict] | None = None
    if _normalized_name(args.dataset) != "synth":
        base_graph = _load_anomaly_graph(args.dataset, data_root=args.data_root)
        limit = args.ego_limit if args.ego_limit > 0 else None
        print(
            f"Preparing anomaly dataset '{args.dataset}' with ego_limit="
            f"{limit if limit is not None else 'all'} and ego_hops={args.ego_hops}"
        )
        if limit is None and int(base_graph.num_nodes) > 200000:
            raise RuntimeError(
                f"Dataset '{args.dataset}' resolved to a very large base graph "
                f"({int(base_graph.num_nodes)} nodes). "
                "Set --ego_limit to a reasonable value (e.g. 5000-50000), "
                "or provide the intended YelpChi binary anomaly source."
            )
        use_cache = bool(args.cache_processed or limit is not None or _normalized_name(args.dataset) != "synth")
        if use_cache:
            prebuilt_ds = _maybe_cache_ego_dataset(
                base_graph=base_graph,
                dataset_name=args.dataset,
                num_hops=args.ego_hops,
                limit=limit,
                cache_dir=args.cache_dir,
                num_workers=_resolve_stage4_num_workers(args.device, int(args.num_workers)),
                max_nodes=int(args.ego_node_cap) if int(args.ego_node_cap) > 0 else None,
            )
        else:
            prebuilt_ds = build_ego_graph_dataset(
                base_graph,
                num_hops=args.ego_hops,
                limit=limit,
                num_workers=_resolve_stage4_num_workers(args.device, int(args.num_workers)),
                max_nodes=int(args.ego_node_cap) if int(args.ego_node_cap) > 0 else None,
            )
        effective_max_motifs = _recommend_anomaly_motif_cap(
            prebuilt_ds,
            int(args.max_motifs),
            int(args.anomaly_motif_budget),
            int(args.anomaly_motif_cap),
        )
        if effective_max_motifs != int(args.max_motifs):
            max_nodes = max(int(item["x"].size(0)) for item in prebuilt_ds) if len(prebuilt_ds) else 0
            print(
                f"Using anomaly max_motifs={effective_max_motifs} "
                f"(requested {args.max_motifs}, max_nodes={max_nodes})"
            )
        prebuilt_ds = _prepare_get_cached_dataset(
            dataset=prebuilt_ds,
            name=f"stage4_anom_{args.dataset}_h{args.ego_hops}",
            cache_dir=args.cache_dir,
            max_motifs=effective_max_motifs if effective_max_motifs > 0 else None,
            pe_k=cache_pe_k,
            rwse_k=args.rwse_k,
            enable_cache=use_cache,
        )

    by_rate: dict[str, list[dict]] = {}
    for rate in tqdm(args.anomaly_label_rates, desc="Label rates"):
        runs = []
        for seed in tqdm(args.seeds, desc=f"Seeds[{rate}]", leave=False):
            set_seed(int(seed))
            if _normalized_name(args.dataset) == "synth":
                ds = make_synth_anomaly_dataset(args.num_graphs, args.in_dim, seed=int(seed))
            else:
                ds = prebuilt_ds
                if ds is None:
                    raise RuntimeError("Failed to prepare anomaly dataset.")

            effective_batch_size = _recommend_anomaly_batch_size(
                ds,
                args.batch_size,
                int(args.anomaly_node_budget),
                int(args.anomaly_batch_cap),
            )
            if effective_batch_size != int(args.batch_size):
                max_nodes = max(int(item["x"].size(0)) for item in ds) if len(ds) else 0
                print(
                    f"Using anomaly batch_size={effective_batch_size} "
                    f"(requested {args.batch_size}, max_nodes={max_nodes})"
                )

            split = build_anomaly_protocol_split(
                ds,
                seed=int(seed),
                labeled_rate=float(rate),
                val_ratio=1,
                test_ratio=2,
            )
            train_ds, val_ds, test_ds = split["train"], split["val"], split["test"]
            in_dim = int(train_ds[0]["x"].size(1))
            make_train_batch_sampler, make_eval_batch_sampler = _make_anomaly_batch_samplers(
                train_ds,
                seed=int(seed),
                node_budget=int(args.anomaly_node_budget),
                hard_cap=int(args.anomaly_batch_cap),
            )
            pairwise, fullget, et_faithful = _build_graph_anomaly_models(
                in_dim=in_dim,
                args=args,
                get_pe_k=get_pe_k,
            )
            loader_num_workers = _resolve_stage4_num_workers(args.device, int(args.num_workers))
            loader_kwargs = build_dataloader_kwargs(
                args.device,
                num_workers=loader_num_workers,
                prefetch_factor=args.prefetch_factor,
            )
            pair_res = _train_graph_binary_with_val(
                pairwise,
                train_ds,
                val_ds,
                test_ds,
                args.epochs,
                effective_batch_size,
                args.device,
                loader_kwargs=loader_kwargs,
                use_weighted_bce=args.weighted_bce,
                eval_batch_sampler=make_eval_batch_sampler,
                train_batch_sampler=make_train_batch_sampler(),
                model_name="PairwiseGET",
                use_amp=getattr(args, "use_amp", None),
                amp_dtype=getattr(args, "amp_dtype", None),
            )
            full_res = _train_graph_binary_with_val(
                fullget,
                train_ds,
                val_ds,
                test_ds,
                args.epochs,
                effective_batch_size,
                args.device,
                loader_kwargs=loader_kwargs,
                use_weighted_bce=args.weighted_bce,
                eval_batch_sampler=make_eval_batch_sampler,
                train_batch_sampler=make_train_batch_sampler(),
                model_name="FullGET",
                use_amp=getattr(args, "use_amp", None),
                amp_dtype=getattr(args, "amp_dtype", None),
            )
            et_res = _train_graph_binary_with_val(
                et_faithful,
                train_ds,
                val_ds,
                test_ds,
                args.epochs,
                effective_batch_size,
                args.device,
                loader_kwargs=loader_kwargs,
                use_weighted_bce=args.weighted_bce,
                eval_batch_sampler=make_eval_batch_sampler,
                train_batch_sampler=make_train_batch_sampler(),
                model_name="ETFaithful",
                use_amp=getattr(args, "use_amp", None),
                amp_dtype=getattr(args, "amp_dtype", None),
            )
            runs.append(
                {
                    "seed": int(seed),
                    "pairwise_auc": float(pair_res.metric),
                    "pairwise_f1": float(pair_res.extra["best_test_f1"]),
                    "fullget_auc": float(full_res.metric),
                    "fullget_f1": float(full_res.extra["best_test_f1"]),
                    "et_faithful_auc": float(et_res.metric),
                    "et_faithful_f1": float(et_res.extra["best_test_f1"]),
                    "histories": {
                        "pairwise": pair_res.history,
                        "fullget": full_res.history,
                        "et_faithful": et_res.history,
                    },
                }
            )
        by_rate[str(float(rate))] = runs

    summary = {}
    for rate_key, runs in by_rate.items():
        pair_auc_mean, pair_auc_std = _mean_std([r["pairwise_auc"] for r in runs])
        pair_f1_mean, pair_f1_std = _mean_std([r["pairwise_f1"] for r in runs])
        fg_auc_mean, fg_auc_std = _mean_std([r["fullget_auc"] for r in runs])
        fg_f1_mean, fg_f1_std = _mean_std([r["fullget_f1"] for r in runs])
        et_auc_mean, et_auc_std = _mean_std([r["et_faithful_auc"] for r in runs])
        et_f1_mean, et_f1_std = _mean_std([r["et_faithful_f1"] for r in runs])
        summary[rate_key] = {
            "pairwise_mean": pair_auc_mean,
            "pairwise_std": pair_auc_std,
            "pairwise_f1_mean": pair_f1_mean,
            "pairwise_f1_std": pair_f1_std,
            "fullget_mean": fg_auc_mean,
            "fullget_std": fg_auc_std,
            "fullget_f1_mean": fg_f1_mean,
            "fullget_f1_std": fg_f1_std,
            "et_faithful_mean": et_auc_mean,
            "et_faithful_std": et_auc_std,
            "et_faithful_f1_mean": et_f1_mean,
            "et_faithful_f1_std": et_f1_std,
        }
    return {"task": "graph_anomaly", "dataset": args.dataset, "summary": summary, "runs": by_rate}
