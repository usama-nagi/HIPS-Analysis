#!/usr/bin/env python3
"""
audit_therapeutic_tiers.py
=================================
master_df["is_therapeutic"] is True for a match at any SI tier (100/99/
95-98), reflecting the best available tier per structure, while the
paper's Methods text defines exact therapeutic status as a 100% SI match
specifically. si_tier itself is retained correctly per row, so this script
works at the analysis layer: every number below is grouped by si_tier
directly, rather than by the collapsed is_therapeutic flag.

Four things this script computes against real data:
  (1) Per-tier row counts and per-tier unique-VH/VL-pair counts (counting
      per unique antibody, not per row), reusing the identical dedup-key
      logic from audit_exclusion_funnel.py so the two cannot diverge.
  (2) One-to-many INN<->structure mapping, checked directly: does any
      single INN map to multiple distinct (pdb,h,l) structures, and does
      any single structure match more than one INN.
  (3) Shorter-CDR-H3 claim: bootstrap CI on the unadjusted mean
      difference, plus a species + format-proxy + germline-adjusted
      version (OLS: cdr3_len ~ is_therapeutic_100pct + heavy_species +
      has_light_chain + heavy_subclass) so the reported effect isn't just
      the raw, potentially confounded difference.
  (4) A pointer, not a recompute: the post-dedup therapeutic-enrichment
      uncertainty is produced by audit_dedup_sensitivity.py's 200-draw
      sweep (therapeutic_fraction key), and reused directly here rather
      than recomputed, so the two scripts cannot disagree.

Usage
-----
    python scripts/audit_therapeutic_tiers.py --config configs/config.yaml \
        --dedup_circularity_json tables/dedup_circularity_check.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_config, require_path, log

SENTINEL = "__MISSING__"


def _key(df, cols):
    return list(zip(*[df[c].fillna(SENTINEL).astype(str) for c in cols]))


def bootstrap_mean_diff_ci(group_a, group_b, n_boot=10000, seed=42):
    """Bootstrap CI on mean(group_a) - mean(group_b), resampling each group
    independently with replacement."""
    rng = np.random.default_rng(seed)
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        a_s = rng.choice(a, size=len(a), replace=True)
        b_s = rng.choice(b, size=len(b), replace=True)
        diffs[i] = a_s.mean() - b_s.mean()
    return {
        "point_estimate": float(a.mean() - b.mean()),
        "ci_2.5": float(np.percentile(diffs, 2.5)),
        "ci_97.5": float(np.percentile(diffs, 97.5)),
        "n_a": len(a), "n_b": len(b),
    }


def ols_adjusted_effect(df, y_col, treatment_col, covariate_cols):
    """Manual OLS via numpy least squares (no statsmodels dependency
    required) -- design matrix = [intercept, treatment, one-hot covariates],
    returns the treatment coefficient with a normal-approximation 95% CI
    from the residual-based standard error. Drops rows with any missing
    covariate rather than silently imputing."""
    work = df[[y_col, treatment_col] + covariate_cols].copy()
    work = work.dropna()
    y = work[y_col].astype(float).values
    X_parts = [np.ones(len(work)), work[treatment_col].astype(float).values]
    col_names = ["intercept", treatment_col]
    for c in covariate_cols:
        dummies = pd.get_dummies(work[c], prefix=c, drop_first=True)
        for dc in dummies.columns:
            X_parts.append(dummies[dc].astype(float).values)
            col_names.append(dc)
    X = np.column_stack(X_parts)
    beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    n, p = X.shape
    y_hat = X @ beta
    resid = y - y_hat
    dof = max(n - p, 1)
    sigma2 = float((resid ** 2).sum() / dof)
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    treat_idx = col_names.index(treatment_col)
    coef = float(beta[treat_idx])
    coef_se = float(se[treat_idx])
    return {
        "adjusted_coefficient": coef,
        "ci_2.5": coef - 1.96 * coef_se,
        "ci_97.5": coef + 1.96 * coef_se,
        "n_rows_used": n,
        "covariates": covariate_cols,
        "note": "Manual OLS (numpy lstsq), normal-approximation 95% CI from "
                "residual variance. Rows with missing covariates dropped.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--dedup_circularity_json", default="tables/dedup_circularity_check.json",
                         help="Path (relative to work_dir, or absolute) to fr8b's output, "
                              "reused for the post-dedup enrichment sensitivity rather than "
                              "recomputed here.")
    parser.add_argument("--n_boot", type=int, default=10000)
    args = parser.parse_args()
    cfg = load_config(args.config)
    work_dir = cfg["paths"]["work_dir"]
    master_path = os.path.join(work_dir, "tables", "master_antibodies.csv")
    require_path(master_path, "master_antibodies.csv")

    df = pd.read_csv(master_path, low_memory=False)
    log(f"Loaded {len(df)} rows")

    required = ["si_tier", "is_therapeutic", "therapeutic_name", "heavy_seq", "light_seq",
                "cdr3_len_actual", "heavy_species"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[SCHEMA MISMATCH] missing {missing}. Found: {list(df.columns)}")

    # ── (1) Per-tier row counts AND per-unique-VH/VL-pair counts ──
    log("=" * 70)
    log("(1) Tier-stratified counts")
    df["_vh_vl_key"] = _key(df, ["heavy_seq", "light_seq"])
    n_unique_antibodies_total = df["_vh_vl_key"].nunique()

    tier_report = {}
    for tier in ["100", "99", "95-98"]:
        tier_rows = df[df["si_tier"] == tier]
        n_rows = len(tier_rows)
        n_unique = tier_rows["_vh_vl_key"].nunique()
        tier_report[tier] = {
            "n_rows": int(n_rows),
            "pct_of_corpus_rows": round(100 * n_rows / len(df), 3),
            "n_unique_antibodies": int(n_unique),
            "pct_of_corpus_unique_antibodies": round(100 * n_unique / n_unique_antibodies_total, 3),
        }
        log(f"  tier {tier}: {n_rows} rows ({100*n_rows/len(df):.3f}%), "
            f"{n_unique} unique antibodies ({100*n_unique/n_unique_antibodies_total:.3f}%)")

    n_any_tier_rows = int(df["is_therapeutic"].sum())
    n_any_tier_unique = df.loc[df["is_therapeutic"] == True, "_vh_vl_key"].nunique()
    log(f"  ANY tier (current is_therapeutic definition): {n_any_tier_rows} rows, "
        f"{n_any_tier_unique} unique antibodies")

    # ── (2) One-to-many INN<->structure mapping, verified ──
    log("=" * 70)
    log("(2) One-to-many INN<->structure mapping (verified against real data)")
    matched = df[df["is_therapeutic"] == True]
    n_unique_inn = matched["therapeutic_name"].nunique()
    inn_to_structures = matched.groupby("therapeutic_name")["pdb_id"].nunique()
    n_inn_with_multiple_structures = int((inn_to_structures > 1).sum())
    max_structures_per_inn = int(inn_to_structures.max()) if len(inn_to_structures) else 0

    structure_key = _key(matched, ["pdb_id", "h_chain", "l_chain"]) if "h_chain" in matched.columns else None
    n_structures_with_multiple_inn = None
    if structure_key is not None:
        matched = matched.copy()
        matched["_struct_key"] = structure_key
        struct_to_inn = matched.groupby("_struct_key")["therapeutic_name"].nunique()
        n_structures_with_multiple_inn = int((struct_to_inn > 1).sum())

    log(f"  {n_unique_inn} unique INNs matched; {n_inn_with_multiple_structures} map to >1 "
        f"distinct PDB structure (max {max_structures_per_inn} structures for one INN)")
    log(f"  Structures matching >1 INN (should be rare/zero per code comment): "
        f"{n_structures_with_multiple_inn}")

    # ── (3) Shorter-CDR-H3 claim: bootstrap CI + adjusted comparison ──
    log("=" * 70)
    log("(3) Shorter-CDR-H3 claim -- bootstrap CI + species/format/germline-adjusted")
    # Restrict to the 100%-tier definition Methods actually claims, per C10
    df["is_therapeutic_100pct"] = df["si_tier"] == "100"
    ther_lens = df.loc[df["is_therapeutic_100pct"], "cdr3_len_actual"].dropna()
    nonther_lens = df.loc[~df["is_therapeutic_100pct"], "cdr3_len_actual"].dropna()
    unadjusted = bootstrap_mean_diff_ci(ther_lens, nonther_lens, n_boot=args.n_boot)
    log(f"  Unadjusted (100%-tier only): therapeutic mean={ther_lens.mean():.2f}, "
        f"non-therapeutic mean={nonther_lens.mean():.2f}, {unadjusted}")

    df["has_light_chain"] = df["light_seq"].notna() & (df["light_seq"].astype(str).str.len() > 0)
    df["_is_ther_int"] = df["is_therapeutic_100pct"].astype(int)
    try:
        adjusted = ols_adjusted_effect(
            df, y_col="cdr3_len_actual", treatment_col="_is_ther_int",
            covariate_cols=["heavy_species", "has_light_chain", "heavy_subclass"],
        )
        log(f"  Adjusted (species + light-chain-presence + germline family): {adjusted}")
    except Exception as e:
        adjusted = {"error": str(e)}
        log(f"  WARNING: adjusted regression failed: {e}")

    # ── (4) Post-dedup enrichment: reuse FR8b's sweep, don't recompute ──
    log("=" * 70)
    log("(4) Post-dedup therapeutic-enrichment uncertainty (from FR8b, not recomputed)")
    dedup_circularity_path = args.dedup_circularity_json if os.path.isabs(args.dedup_circularity_json) else os.path.join(work_dir, args.dedup_circularity_json)
    dedup_circularity_sweep = None
    if os.path.exists(dedup_circularity_path):
        with open(dedup_circularity_path) as f:
            dedup_circularity_data = json.load(f)
        dedup_circularity_sweep = dedup_circularity_data.get("random_sweep_sensitivity", {}).get("therapeutic_fraction")
        log(f"  Loaded from {dedup_circularity_path}: {dedup_circularity_sweep}")
    else:
        log(f"  WARNING: {dedup_circularity_path} not found -- run audit_dedup_sensitivity.py first "
            f"if this section is needed.")

    report = {
        "tier_stratified_counts": tier_report,
        "any_tier_current_definition": {
            "n_rows": n_any_tier_rows, "n_unique_antibodies": int(n_any_tier_unique),
        },
        "n_unique_antibodies_total_corpus": int(n_unique_antibodies_total),
        "inn_mapping": {
            "n_unique_inn": int(n_unique_inn),
            "n_inn_with_multiple_structures": n_inn_with_multiple_structures,
            "max_structures_per_inn": max_structures_per_inn,
            "n_structures_with_multiple_inn": n_structures_with_multiple_inn,
        },
        "shorter_h3_claim": {
            "unadjusted_bootstrap_ci": unadjusted,
            "species_format_germline_adjusted": adjusted,
        },
        "post_dedup_enrichment_sensitivity_from_dedup_circularity_check": dedup_circularity_sweep,
    }
    out_path = os.path.join(work_dir, "tables", "therapeutic_tier_audit.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
