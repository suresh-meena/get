from __future__ import annotations

import argparse
import statistics

from tqdm.auto import tqdm

from experiments.shared.common import add_cached_structural_features, build_dataloader_kwargs, load_tu_dataset, set_seed
from experiments.shared.model_config import instantiate_models_from_catalog

from .shared import (
    TU8_DATASETS,
    _build_classification_folds,
    _build_graph_classification_models,
    _mean_std,
    _normalized_name,
    _prepare_get_cached_dataset,
    _resolve_stage4_num_workers,
    _split_train_val_indices,
    _train_graph_classification,
    make_synth_classification_dataset,
)


def run_graph_classification(args: argparse.Namespace) -> dict:
    get_pe_k = max(0, int(args.get_pe_k))
    cache_pe_k = max(get_pe_k, max(0, int(args.et_pe_k)))
    if _normalized_name(args.dataset) in {"tu8", "alltu"}:
        target_datasets = list(TU8_DATASETS)
    else:
        target_datasets = [args.dataset]

    all_payloads = {}
    skipped_datasets: list[str] = []
    for dataset_name in tqdm(target_datasets, desc="Datasets"):
        runs = []
        for seed in tqdm(args.seeds, desc=f"Seeds[{dataset_name}]", leave=False):
            set_seed(int(seed))
            if _normalized_name(dataset_name) == "synth":
                raw_ds = make_synth_classification_dataset(args.num_graphs, args.in_dim, seed=int(seed))
            else:
                try:
                    raw_ds = load_tu_dataset(dataset_name, limit=args.limit_graphs if args.limit_graphs > 0 else None)
                except Exception as exc:
                    print(f"Skipping classification dataset '{dataset_name}': {exc}")
                    skipped_datasets.append(dataset_name)
                    runs = []
                    break
                if len(raw_ds) == 0:
                    print(f"Skipping classification dataset '{dataset_name}': no samples loaded.")
                    skipped_datasets.append(dataset_name)
                    runs = []
                    break

            proc_ds = _prepare_get_cached_dataset(
                dataset=raw_ds,
                name=f"stage4_cls_{dataset_name}",
                cache_dir=args.cache_dir,
                max_motifs=args.max_motifs if args.max_motifs > 0 else None,
                pe_k=cache_pe_k,
                rwse_k=args.rwse_k,
                # Classification reuses the same graphs across many epochs and folds,
                # so caching the processed motif-heavy representation is almost always
                # cheaper than redoing it inside every collate call.
                enable_cache=True,
            )
            labels = [int(g["y"].view(-1)[0].item()) for g in proc_ds]
            unique_labels = sorted(set(labels))
            num_classes = int(max(unique_labels)) + 1 if unique_labels else 2
            in_dim = int(proc_ds[0]["x"].size(1))
            split_indices = _build_classification_folds(labels, int(args.cv_folds), int(seed))

            fold_runs = []
            loader_num_workers = _resolve_stage4_num_workers(args.device, int(args.num_workers))
            loader_kwargs = build_dataloader_kwargs(
                args.device,
                num_workers=loader_num_workers,
                prefetch_factor=args.prefetch_factor,
            )
            for fold_id, (train_idx, test_idx) in enumerate(split_indices):
                train_sub_idx, val_sub_idx = _split_train_val_indices(list(train_idx), labels, int(seed), int(fold_id))
                test_idx = list(test_idx)
                train_ds = [proc_ds[i] for i in train_sub_idx]
                val_ds = [proc_ds[i] for i in val_sub_idx]
                test_ds = [proc_ds[i] for i in test_idx]
                train_ds_gin = add_cached_structural_features(train_ds)
                val_ds_gin = add_cached_structural_features(val_ds)
                test_ds_gin = add_cached_structural_features(test_ds)
                pairwise, fullget, et_faithful = _build_graph_classification_models(
                    in_dim=in_dim,
                    num_classes=num_classes,
                    args=args,
                    get_pe_k=get_pe_k,
                )
                gin = None
                try:
                    gin_context = {
                        "gin_in_dim": int(train_ds_gin[0]["x"].size(1)),
                        "hidden_dim": int(args.hidden_dim),
                        "num_classes": num_classes,
                    }
                    gin = instantiate_models_from_catalog(
                        args.model_config,
                        context=gin_context,
                        names=["GIN"],
                    )["GIN"]
                except Exception:
                    gin = None

                pair_res = _train_graph_classification(
                    pairwise,
                    train_ds,
                    val_ds,
                    test_ds,
                    args.epochs,
                    args.batch_size,
                    args.device,
                    loader_kwargs=loader_kwargs,
                    model_name="PairwiseGET",
                    use_amp=getattr(args, "use_amp", None),
                    amp_dtype=getattr(args, "amp_dtype", None),
                )
                full_res = _train_graph_classification(
                    fullget,
                    train_ds,
                    val_ds,
                    test_ds,
                    args.epochs,
                    args.batch_size,
                    args.device,
                    loader_kwargs=loader_kwargs,
                    model_name="FullGET",
                    use_amp=getattr(args, "use_amp", None),
                    amp_dtype=getattr(args, "amp_dtype", None),
                )
                et_res = _train_graph_classification(
                    et_faithful,
                    train_ds,
                    val_ds,
                    test_ds,
                    args.epochs,
                    args.batch_size,
                    args.device,
                    loader_kwargs=loader_kwargs,
                    model_name="ETFaithful",
                    use_amp=getattr(args, "use_amp", None),
                    amp_dtype=getattr(args, "amp_dtype", None),
                )

                fold_run = {
                    "fold": int(fold_id),
                    "pairwise_acc": pair_res.metric,
                    "fullget_acc": full_res.metric,
                    "et_faithful_acc": et_res.metric,
                    "histories": {
                        "pairwise": pair_res.history,
                        "fullget": full_res.history,
                        "et_faithful": et_res.history,
                    },
                    "energy_traces": {
                        "pairwise": pair_res.extra.get("energy_trace"),
                        "fullget": full_res.extra.get("energy_trace"),
                        "et_faithful": et_res.extra.get("energy_trace"),
                    },
                }
                if gin is not None:
                    gin_res = _train_graph_classification(
                        gin,
                        train_ds_gin,
                        val_ds_gin,
                        test_ds_gin,
                        args.epochs,
                        args.batch_size,
                        args.device,
                        loader_kwargs=loader_kwargs,
                        model_name="GIN",
                        use_amp=getattr(args, "use_amp", None),
                        amp_dtype=getattr(args, "amp_dtype", None),
                    )
                    fold_run["gin_struct_acc"] = gin_res.metric
                    fold_run["histories"]["gin_struct"] = gin_res.history
                    fold_run["energy_traces"]["gin_struct"] = gin_res.extra.get("energy_trace")
                fold_runs.append(fold_run)

            if not fold_runs:
                runs = []
                break

            runs.append(
                {
                    "seed": int(seed),
                    "folds": fold_runs,
                    "pairwise_acc": float(statistics.fmean([fr["pairwise_acc"] for fr in fold_runs])),
                    "fullget_acc": float(statistics.fmean([fr["fullget_acc"] for fr in fold_runs])),
                    "et_faithful_acc": float(statistics.fmean([fr["et_faithful_acc"] for fr in fold_runs])),
                }
            )

        pair_mean, pair_std = _mean_std([r["pairwise_acc"] for r in runs])
        full_mean, full_std = _mean_std([r["fullget_acc"] for r in runs])
        et_mean, et_std = _mean_std([r["et_faithful_acc"] for r in runs])
        all_payloads[dataset_name] = {
            "dataset": dataset_name,
            "summary": {
                "pairwise_mean": pair_mean,
                "pairwise_std": pair_std,
                "fullget_mean": full_mean,
                "fullget_std": full_std,
                "et_faithful_mean": et_mean,
                "et_faithful_std": et_std,
            },
            "runs": runs,
        }

    if len(all_payloads) == 0:
        skipped = ", ".join(sorted(set(skipped_datasets))) if skipped_datasets else "all requested datasets"
        raise RuntimeError(f"No TU classification datasets could be loaded ({skipped}).")

    if skipped_datasets:
        print(f"Skipped unavailable TU datasets: {', '.join(sorted(set(skipped_datasets)))}")

    if len(all_payloads) == 1:
        only = next(iter(all_payloads.values()))
        return {"task": "graph_classification", **only}
    return {"task": "graph_classification", "dataset": "tu8", "per_dataset": all_payloads}
