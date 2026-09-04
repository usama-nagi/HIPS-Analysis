#!/usr/bin/env python3
"""
audit_sars_cov2_sensitivity.py
================================
How much of the antigen-concentration finding (2,292 antigen clusters,
Gini=0.715; antigen_class 40.1% viral) is driven by the SARS-CoV-2
structural surge specifically?

Two complementary cuts, each computed against the full corpus and reported
side by side:
  (1) Exclude all SARS-CoV-2-antigen entries (via antigen_species).
  (2) Restrict to pre-2020 depositions (before the pandemic surge began).

Deliberately reuses the pipeline's own shannon_entropy/gini_coefficient
(scripts/common.py) and classify_antigen (scripts/rq1_sequence_structural_bias.py)
rather than reimplementing them, so results are guaranteed methodologically
identical to the numbers already in the paper -- only the input population
differs.

Schema-dump-first: before applying any SARS-CoV-2 filter, this script prints
every distinct antigen_species value that matches "sars" case-insensitively,
so the exact filter string is confirmed against real data rather than
assumed. It aborts loudly if that filter matches zero rows.

Reads:
  <work_dir>/tables/master_antibodies.csv

Writes:
  <work_dir>/tables/audit_sars_cov2_sensitivity.json

Usage
-----
    python scripts/audit_sars_cov2_sensitivity.py --config configs/config.yaml
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_config, shannon_entropy, gini_coefficient, require_path, log
from rq1_sequence_structural_bias import classify_antigen, year_from_date

TOP_N = 10


def diversity_snapshot(df: pd.DataFrame, label: str) -> dict:
    """Antigen-cluster and antigen-class diversity metrics for one
    (sub)population, restricted to has_antigen==True throughout, mirroring
    rq3_redundancy_and_recommendations.py's compute_diversity_snapshot for
    the antigen_cluster_id block and rq1_sequence_structural_bias.py's
    Section C for the antigen_class block."""
    ag = df.loc[df["has_antigen"] == True].copy()
    out = {
        "label": label,
        "n_total": int(len(df)),
        "n_antigen_bound": int(len(ag)),
    }

    # --- antigen_cluster_id block ---
    cl = ag["antigen_cluster_id"].dropna()
    if len(cl) > 0:
        vc = cl.value_counts()
        top_n_frac = float(vc.head(TOP_N).sum() / vc.sum())
        out["antigen_cluster_n_unique"] = int(vc.shape[0])
        out["antigen_cluster_entropy_bits"] = shannon_entropy(vc.values)
        out["antigen_cluster_gini"] = gini_coefficient(vc.values)
        out[f"antigen_cluster_top{TOP_N}_fraction"] = top_n_frac
    else:
        out["antigen_cluster_n_unique"] = 0

    # --- antigen_class block (reuses the pipeline's own classifier) ---
    if len(ag) > 0:
        ag["antigen_class"] = ag.apply(
            lambda r: classify_antigen(r.get("antigen_name", ""), r.get("antigen_type", "")),
            axis=1,
        )
        class_counts = ag["antigen_class"].value_counts()
        out["antigen_class_counts"] = class_counts.to_dict()
        out["antigen_class_entropy_bits"] = shannon_entropy(class_counts.values)
        out["antigen_class_gini"] = gini_coefficient(class_counts.values)
        out["viral_fraction"] = float(class_counts.get("viral", 0) / class_counts.sum())

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    work_dir = cfg["paths"]["work_dir"]
    master_path = os.path.join(work_dir, "tables", "master_antibodies.csv")
    out_path = os.path.join(work_dir, "tables", "audit_sars_cov2_sensitivity.json")

    require_path(master_path, "master_antibodies.csv")
    df = pd.read_csv(master_path)
    log(f"Loaded {len(df)} entries")

    for col in ["has_antigen", "antigen_cluster_id", "antigen_species", "antigen_name",
                "antigen_type", "date"]:
        if col not in df.columns:
            sys.exit(f"ABORT: required column '{col}' not found in {master_path}")

    # ── Schema-dump first: don't guess a substring, look at what's actually there ──
    # Filtering antigen_species for just the substring "sars" would match only
    # "sars coronavirus" (the original 2003 SARS / SARS-CoV-1) and miss
    # SARS-CoV-2 entirely wherever the species field stores the spelled-out
    # name ("severe acute respiratory syndrome coronavirus 2"), which contains
    # no literal "sars" substring. Dumping raw top-N frequency sidesteps
    # guessing the string at all.
    log(f"Top {2*TOP_N} antigen_species values by raw frequency (unfiltered):")
    top_species = df["antigen_species"].value_counts().head(2 * TOP_N)
    for val, n in top_species.items():
        log(f"    {val!r}: {n}")

    species_lower = df["antigen_species"].astype(str).str.lower()
    # "coronavirus2" (no space) is the real string, confirmed by the top-N
    # dump above: 'severe acute respiratory syndrome coronavirus2'. A spaced
    # "coronavirus 2" term is deliberately not used here: it would
    # false-positive match 'human betacoronavirus 2c emc/2012' (MERS-CoV's
    # species designation, n=15), a different virus entirely.
    broad_terms = ["sars-cov-2", "sars cov 2", "coronavirus2", "cov-2", "cov2", "covid"]
    sars_mask_species = species_lower.apply(
        lambda s: any(t in s for t in broad_terms)
    )
    log("Distinct antigen_species values matching any of "
        f"{broad_terms} (case-insensitive):")
    for val, n in df.loc[sars_mask_species, "antigen_species"].value_counts().items():
        log(f"    {val!r}: {n}")

    if sars_mask_species.sum() == 0:
        sys.exit("ABORT: zero rows matched any broad SARS-CoV-2 term in antigen_species. "
                  "Inspect the top-N dump above, find the real string by eye, and hardcode "
                  "it into broad_terms. Do not trust any number below until this matches.")

    # Cross-check against antigen_name, in case some SARS-CoV-2 entries have an
    # unpopulated/blank antigen_species but a name that clearly indicates it.
    name_lower = df["antigen_name"].astype(str).str.lower()
    sars_mask_name_only = name_lower.str.contains("sars-cov-2|sars cov 2|covid", na=False) & ~sars_mask_species
    if sars_mask_name_only.sum() > 0:
        log(f"NOTE: {sars_mask_name_only.sum()} additional rows match a SARS-CoV-2-like "
            f"antigen_name but were NOT caught by the antigen_species filter -- inspect "
            f"these before deciding whether to fold them in:")
        for val, n in df.loc[sars_mask_name_only, "antigen_name"].value_counts().head(20).items():
            log(f"    {val!r}: {n}")

    sars_mask = sars_mask_species | sars_mask_name_only
    log(f"SARS-CoV-2-antigen entries (species-or-name union): {int(sars_mask.sum())} / {len(df)}")
    if abs(int(sars_mask.sum()) - 2541) > 100:
        log(f"WARNING: this count ({int(sars_mask.sum())}) is far from the paper's stated "
            f"2,541 SARS-CoV-2 antigen_species figure (Sec 4.1). Do not proceed to trust "
            f"the excluding_sars_cov2_antigen numbers below without reconciling this gap "
            f"first -- check the top-N dump above for the real string.")

    # ── Cut 1: exclude SARS-CoV-2-antigen entries ──
    full = diversity_snapshot(df, "full_corpus")
    excl_sars = diversity_snapshot(df.loc[~sars_mask], "excluding_sars_cov2_antigen")

    # ── Cut 2: restrict to pre-2020 depositions ──
    years = df["date"].apply(year_from_date)
    pre2020_mask = years.notna() & (years < 2020)
    log(f"Pre-2020-deposition entries: {int(pre2020_mask.sum())} / {len(df)} "
        f"({df['date'].notna().sum()} have a parseable date)")
    pre2020 = diversity_snapshot(df.loc[pre2020_mask], "pre_2020_depositions")

    summary = {
        "sars_cov2_antigen_species_values": {
            str(k): int(v) for k, v in
            df.loc[sars_mask_species, "antigen_species"].value_counts().items()
        },
        "full_corpus": full,
        "excluding_sars_cov2_antigen": excl_sars,
        "pre_2020_depositions": pre2020,
    }

    os.makedirs(os.path.join(work_dir, "tables"), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"Wrote {out_path}")

    log("=" * 70)
    log("SUMMARY (antigen-bound entries only)")
    for snap in (full, excl_sars, pre2020):
        log(f"  {snap['label']:28s} n_antigen_bound={snap.get('n_antigen_bound'):>6} "
            f"n_clusters={snap.get('antigen_cluster_n_unique'):>6} "
            f"entropy={snap.get('antigen_cluster_entropy_bits', float('nan')):.3f} "
            f"gini={snap.get('antigen_cluster_gini', float('nan')):.3f} "
            f"top{TOP_N}_frac={snap.get(f'antigen_cluster_top{TOP_N}_fraction', float('nan')):.3f} "
            f"viral_frac={snap.get('viral_fraction', float('nan')):.3f}")


if __name__ == "__main__":
    main()
