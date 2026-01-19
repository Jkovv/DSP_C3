#!/usr/bin/env python3
"""
CBS News-to-Report Linkage — Error Analysis + Dataset Comparison (Ranking-first)

Compares 3 datasets:
- final_trainset.csv (legacy)
- "final_basic_trainset_fixed.csv" (fixed baseline)
- final_hybrid_sbert_trainset_100pct.csv (hybrid, 100% dataset)

Outputs (all written to repo root):
- error_analysis_results.csv   : model x dataset comparison table (ranking + diagnostics)
- error_cases.csv     : top-1 error cases for the BEST model per dataset (explainable tags)

Console output:
- a clear table of the most important, explainable results
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# Utilities
def infer_id_col(df: pd.DataFrame) -> str:
    for c in ["child_id", "id_child", "id", "c"]:
        if c in df.columns:
            return c
    # fallback
    num_matches = int(df["match"].sum()) if "match" in df.columns else 0
    avg_block = int(len(df) / num_matches) if num_matches > 0 else 50
    df["child_id"] = np.arange(len(df)) // avg_block
    return "child_id"


def build_legacy_score(df: pd.DataFrame) -> pd.Series:
    if "jac_total" in df.columns:
        return df["jac_total"]
    sim_cols = [c for c in ["sbert_sim", "title_similarity", "tax_matches", "num_matches"] if c in df.columns]
    if not sim_cols:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return df[sim_cols].sum(axis=1)


def ranking_hits_total(df, score_col: str, k: int, id_col: str):
    grouped = df.groupby(id_col)
    total_relevant = 0
    hits = 0
    for _, g in grouped:
        if g["match"].sum() > 0:
            total_relevant += 1
            top_k = g.sort_values(score_col, ascending=False).head(k)
            if top_k["match"].sum() > 0:
                hits += 1
    return hits, total_relevant

def rank_of_true_match(group: pd.DataFrame, score_col: str) -> Optional[int]:
    if group["match"].sum() <= 0:
        return None
    sorted_g = group.sort_values(score_col, ascending=False).reset_index(drop=True)
    true_idx = sorted_g.index[sorted_g["match"] == 1]
    if len(true_idx) == 0:
        return None
    return int(true_idx[0] + 1)


def top1_margin(group: pd.DataFrame, score_col: str) -> float:
    scores = group.sort_values(score_col, ascending=False)[score_col].values
    if len(scores) < 2:
        return 0.0
    return float(scores[0] - scores[1])


# Error taxonomy (explainable heuristics)
@dataclass
class ErrorTagConfig:
    near_miss_margin: float = 0.03
    low_semantic: float = 0.25
    high_entity: int = 2
    high_numeric: int = 2
    high_tax: int = 2


def tag_error_case(row: pd.Series, cfg: ErrorTagConfig) -> List[str]:
    tags: List[str] = []

    margin = row.get("top1_margin", np.nan)
    if pd.notna(margin) and margin <= cfg.near_miss_margin:
        tags.append("near_miss_ambiguity")

    sbert = row.get("sbert_sim", np.nan)
    if pd.notna(sbert) and sbert < cfg.low_semantic:
        tags.append("low_sbert_signal")

    ner = row.get("ner_overlap", np.nan)
    if pd.notna(ner) and ner >= cfg.high_entity:
        tags.append("entity_overlap_ambiguity")

    nums = row.get("num_matches", np.nan)
    if pd.notna(nums) and nums >= cfg.high_numeric:
        tags.append("numeric_alignment_conflict")

    tax = row.get("tax_matches", np.nan)
    if pd.notna(tax) and tax >= cfg.high_tax:
        tags.append("high_taxonomy_overlap")

    if not tags:
        tags.append("unclassified")

    return tags


# Model factory
def resolve_models(model_list: List[str]) -> List[str]:
    out: List[str] = []
    for m in model_list:
        ml = m.lower().strip()
        if ml in ["rf", "randomforest", "random_forest"]:
            out.append("rf")
        elif ml == "catboost" and HAS_CATBOOST:
            out.append("catboost")
        elif ml == "lightgbm" and HAS_LGBM:
            out.append("lightgbm")
        elif ml == "xgboost" and HAS_XGB:
            out.append("xgboost")

    # de-dupe preserving order
    seen = set()
    final = []
    for m in out:
        if m not in seen:
            final.append(m)
            seen.add(m)
    if not final:
        final = ["rf"]
    return final


def make_model(name: str, seed: int):
    name = name.lower()
    if name == "catboost":
        return CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            random_seed=seed,
            verbose=0,
        )
    if name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=300,
            random_state=seed,
            verbose=-1,
        )
    if name == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=300,
            random_state=seed,
            eval_metric="logloss",
        )
    return RandomForestClassifier(
        n_estimators=400,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )


# Fit + evaluate one model
def fit_eval_one_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    id_col: str,
    k: int,
    seed: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    y_train = train_df["match"].astype(int)
    y_test = test_df["match"].astype(int)

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())

    model = make_model(model_name, seed=seed)
    model.fit(train_df[features], y_train)

    probs = model.predict_proba(test_df[features])[:, 1]
    preds = (probs > 0.5).astype(int)

    scored = test_df.copy()
    scored["probs"] = probs

    hits1, total = ranking_hits_total(scored, "probs", 1, id_col)
    hitsk, _     = ranking_hits_total(scored, "probs", k, id_col)

    success_at_1 = hits1 / total if total else 0.0
    success_at_k = hitsk / total if total else 0.0

    top1_errors = total - hits1
    topk_errors = total - hitsk


    # classification diagnostics (secondary)
    n_groups = total  # IMPORTANT: relevant groups only (groups that contain a positive)

    met: Dict[str, float] = {
        "success_at_1": float(success_at_1),
        f"success_at_{k}": float(success_at_k),
        "top1_errors": float(top1_errors),
        f"top{k}_errors": float(topk_errors),

        # secondary diagnostics (classification on rows)
        "test_accuracy": float(accuracy_score(y_test, preds)),
        "test_precision": float(precision_score(y_test, preds, zero_division=0)),
        "test_recall": float(recall_score(y_test, preds, zero_division=0)),
        "test_f1": float(f1_score(y_test, preds, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) == 2 else float("nan"),

        # IMPORTANT: relevant-group count, not total unique child_id
        "n_test_groups": float(n_groups),
        "n_test_rows": float(len(scored)),
}


    return met, scored


# Extract top-1 error cases (best model per dataset)
    def extract_top1_error_cases(scored: pd.DataFrame, id_col: str, k: int) -> pd.DataFrame:
    rows = []
    for cid, g in scored.groupby(id_col):
        if g["match"].sum() <= 0:
            continue
        true_rank = rank_of_true_match(g, "probs")
        if true_rank is None or true_rank <= 1:
            continue  # not a top-1 error

        margin = top1_margin(g, "probs")
        true_row = g[g["match"] == 1].iloc[0].to_dict()

        rows.append({
            id_col: cid,
            "true_rank": int(true_rank),
            "top1_margin": float(margin),
            # explainable hybrid features if present
            "sbert_sim": true_row.get("sbert_sim", np.nan),
            "spacy_sim": true_row.get("spacy_sim", np.nan),
            "tax_matches": true_row.get("tax_matches", np.nan),
            "ner_overlap": true_row.get("ner_overlap", np.nan),
            "num_matches": true_row.get("num_matches", np.nan),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    cfg = ErrorTagConfig()
    df["error_tags"] = df.apply(lambda r: "|".join(tag_error_case(r, cfg)), axis=1)
    df[f"is_top{k}_error"] = df["true_rank"] > k
    return df


# Main: compare datasets
def run(
    datasets: List[str],
    models: List[str],
    k: int,
    test_size: float,
    seed: int,
    out_results_csv: str,
    out_errors_csv: str,
) -> None:
    models = resolve_models(models)

    all_results: List[Dict[str, float]] = []
    all_error_cases: List[pd.DataFrame] = []

    for dataset_path in datasets:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        df = pd.read_csv(dataset_path).fillna(0)
        if "match" not in df.columns:
            raise ValueError(f"{dataset_path} must contain a 'match' column (0/1).")

        id_col = infer_id_col(df)
        df["legacy_score"] = build_legacy_score(df)

        # Feature selection similar to your audit style:
        exclude = {"match", "child_id", "parent_id", "id", "id_child", "legacy_score", "c", "p"}
        features = [c for c in df.columns if c not in exclude]

        # Group-aware split
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(df, groups=df[id_col]))
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        # legacy baseline ranking on test

        # evaluate all requested models on this dataset
        dataset_model_rows = []
        dataset_scored_by_model: Dict[str, pd.DataFrame] = {}

        for m in models:
            met, scored = fit_eval_one_model(
                model_name=m,
                train_df=train_df,
                test_df=test_df,
                features=features,
                id_col=id_col,
                k=k,
                seed=seed,
            )
            row = {
                "dataset": os.path.basename(dataset_path),
                "model": m,
                "id_col": id_col,
                **met,
            }
            dataset_model_rows.append(row)
            dataset_scored_by_model[m] = scored

        # pick best model for error extraction: maximize Success@1, tie-break Success@K, then higher ROC-AUC
        dataset_model_rows_sorted = sorted(
            dataset_model_rows,
            key=lambda r: (r["success_at_1"], r.get(f"success_at_{k}", 0.0), r.get("test_roc_auc", 0.0)),
            reverse=True,
        )
        best = dataset_model_rows_sorted[0]
        best_model = best["model"]

        # error cases for best model on this dataset
        for model_name, scored_df in dataset_scored_by_model.items():
            err_df = extract_top1_error_cases(scored_df, id_col=id_col, k=k)
            if err_df.empty:
                continue

            err_df.insert(0, "dataset", os.path.basename(dataset_path))
            err_df.insert(1, "model", model_name)
            all_error_cases.append(err_df)


        all_results.extend(dataset_model_rows)

    # Write model comparison results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_results_csv, index=False, sep=";")

    # Write error cases (best model per dataset)
    if all_error_cases:
        errors_df = pd.concat(all_error_cases, ignore_index=True)
    else:
        errors_df = pd.DataFrame(columns=["dataset", "best_model", "child_id", "true_rank", "top1_margin", "error_tags"])
    errors_df.to_csv(out_errors_csv, index=False, sep=";")

    # Console: explainable summary table
    display_cols = [
        "dataset", "model",
        "success_at_1", f"success_at_{k}",
        "top1_errors", f"top{k}_errors",
        "test_roc_auc",
    ]
    for c in display_cols:
        if c not in results_df.columns:
            results_df[c] = np.nan

    shown = results_df[display_cols].copy()
    for c in ["success_at_1", f"success_at_{k}", "test_roc_auc"]:
        shown[c] = shown[c].astype(float).round(4)
    for c in ["top1_errors", f"top{k}_errors"]:
        shown[c] = shown[c].astype(float).round(0).astype(int)

    # Sort: best models first within each dataset
    shown = shown.sort_values(["dataset", "success_at_1", f"success_at_{k}", "test_roc_auc"], ascending=[True, False, False, False])

    print("\nDATASET + MODEL COMPARISON (RANKING-FIRST)")
    print(shown.to_string(index=False))

    print("\nSaved:")
    print(f" - {out_results_csv}")
    print(f" - {out_errors_csv}")

    # Quick tag counts
    if not errors_df.empty and "error_tags" in errors_df.columns:
        tag_counts = errors_df["error_tags"].str.split(r"\|").explode().value_counts()
        if len(tag_counts) > 0:
            print("\nTOP-1 ERROR TAG COUNTS (BEST MODEL PER DATASET)")
            print(tag_counts.to_string())
        else:
            print("\nNo top-1 errors found (best models).")
    else:
        print("\nNo top-1 errors found (best models).")


def parse_args():
    p = argparse.ArgumentParser(description="CBS linkage error analysis across 3 datasets (no JSON/MD, root outputs).")
    p.add_argument("--k", type=int, default=3, help="K for Success@K.")
    p.add_argument("--test-size", type=float, default=0.3, help="GroupShuffleSplit test size.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--datasets", nargs="+", default=[
        "final_trainset.csv",
        "final_basic_trainset_fixed.csv",
        "final_hybrid_sbert_trainset_100pct.csv",
    ], help="Datasets to compare.")
    p.add_argument("--models", nargs="+", default=["catboost", "lightgbm", "xgboost", "rf"],
                   help="Models to compare. Unavailable ones are skipped automatically.")
    p.add_argument("--out-results", type=str, default="error_analysis_results.csv", help="Output CSV for model comparison.")
    p.add_argument("--out-errors", type=str, default="error_cases.csv", help="Output CSV for top-1 errors (best model per dataset).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        datasets=args.datasets,
        models=args.models,
        k=args.k,
        test_size=args.test_size,
        seed=args.seed,
        out_results_csv=args.out_results,
        out_errors_csv=args.out_errors,
    )
