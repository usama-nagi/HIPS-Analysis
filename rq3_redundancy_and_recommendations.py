#!/usr/bin/env python3
"""
rq3_redundancy_and_recommendations.py
========================================
RQ3: What redundancy and imbalance exists in SAbDab, and what benchmark-
design recommendations follow from it?

Two sections:

  SECTION A — before/after deduplication
      Recomputes every diversity metric from RQ1 twice: once on raw
      SAbDab, once after CDR-H3 cluster-aware deduplication. This
      before/after framing is the paper's strongest single result: a
      falsifiable, quantified claim ("applying the field-standard fix
      changes apparent diversity by X%"), not a static bias inventory.

      Dedup rule (simple and auditable on purpose): one representative
      per CDR-H3 MMseqs2 cluster (from RQ1 Section B's output), preferring
      has_antigen=True, deterministic pdb_id tie-break.

      Requires rq1_sequence_structural_bias.py (Section B) to have been
      run first — needs rq1_cdrh3_clusters.tsv.

  SECTION B — benchmark design checklist
      Turns the numbers already computed across RQ1/RQ2/Section A into a
      concrete, numbered checklist, each item citing the specific
      statistic that motivates it. Makes NO new measurements — if an
      upstream file is missing, the corresponding item is marked
      "INSUFFICIENT DATA" rather than guessed.

Output (all under outputs/tables/)
-----------------------------------
rq3_before_after_dedup.json            (Section A)
rq3_deduplicated_master.csv            (the deduplicated subset itself —
                                         reusable directly as a benchmark split)

rq3_benchmark_checklist.json           (Section B)
rq3_benchmark_checklist.md             (human-readable, drop into paper)

Usage
-----
    python scripts/rq3_redundancy_and_recommendations.py --config configs/config.yaml
    python scripts/rq3_redundancy_and_recommendations.py --config configs/config.yaml --only A
"""

import os
import sys
import json
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    load_config, require_path, shannon_entropy, normalized_entropy,
    gini_coefficient, log,
)


# ═════════════════════════════════════════════════════════════════════════
# SECTION A — before/after deduplication
# ═════════════════════════════════════════════════════════════════════════

def pick_cluster_representatives(master_df: pd.DataFrame, clusters_df: pd.DataFrame) -> pd.DataFrame:
    """For each CDR-H3 cluster, pick one representative row from master_df.
    Preference: has_antigen=True, then lexicographically smallest pdb_id
    (deterministic tie-break, not random)."""
    merged = master_df.merge(
        clusters_df[["filename_stem", "cluster_rep"]],
        on="filename_stem", how="inner",
    )
    if len(merged) < len(master_df):
        log(f"WARNING: {len(master_df) - len(merged)} master_df rows had no "
            f"matching CDR-H3 cluster assignment (not present in "
            f"rq1_cdrh3_clusters.tsv) — excluded from dedup analysis.")

    merged = merged.sort_values(
        by=["cluster_rep", "has_antigen", "pdb_id"],
        ascending=[True, False, True],
    )
    dedup = merged.drop_duplicates(subset=["cluster_rep"], keep="first")
    return dedup


def compute_diversity_snapshot(df: pd.DataFrame) -> dict:
    """Recomputes the same core diversity metrics used in RQ1 on whatever
    dataframe is passed in, so it can be called once on the full set and
    once on the deduplicated set with guaranteed identical logic."""
    out = {"n_antibodies": int(len(df))}

    if "cdr3_len_actual" in df.columns:
        lens = df["cdr3_len_actual"].dropna().astype(int)
        len_counts = lens.value_counts()
        out["length_entropy_bits"] = shannon_entropy(len_counts.values)
        out["length_normalized_entropy"] = normalized_entropy(len_counts.values)
        out["length_gini"] = gini_coefficient(len_counts.values)
        out["length_mean"] = float(lens.mean()) if len(lens) else None

    if "heavy_subclass" in df.columns:
        vc = df["heavy_subclass"].fillna("UNKNOWN").value_counts()
        out["heavy_germline_entropy_bits"] = shannon_entropy(vc.values)
        out["heavy_germline_gini"] = gini_coefficient(vc.values)
        out["heavy_germline_n_unique"] = int(vc.shape[0])

    if "antigen_cluster_id" in df.columns:
        ag = df.loc[df["has_antigen"] == True, "antigen_cluster_id"].dropna()
        if len(ag) > 0:
            vc = ag.value_counts()
            out["antigen_cluster_entropy_bits"] = shannon_entropy(vc.values)
            out["antigen_cluster_gini"] = gini_coefficient(vc.values)
            out["antigen_cluster_n_unique"] = int(vc.shape[0])

    if "is_therapeutic" in df.columns:
        out["therapeutic_fraction"] = float(df["is_therapeutic"].mean())

    return out


def run_section_a(work_dir: str, cfg: dict) -> dict:
    log("=" * 70)
    log("SECTION A: before/after deduplication")

    master_csv = os.path.join(work_dir, "tables", "master_antibodies.csv")
    clusters_tsv = os.path.join(work_dir, "tables", "rq1_cdrh3_clusters.tsv")
    require_path(master_csv, "master_antibodies.csv (run 00_build_dataset.py first)")
    require_path(clusters_tsv,
                 "rq1_cdrh3_clusters.tsv (run rq1_sequence_structural_bias.py "
                 "Section B first, without --skip_mmseqs)")

    master_df = pd.read_csv(master_csv)
    clusters_df = pd.read_csv(clusters_tsv, sep="\t")

    # master_antibodies.csv may already carry its own cluster_rep column
    # (e.g. from migrate_master_csv.py --only cluster_rep, run at some
    # earlier point against a possibly-different rq1_cdrh3_clusters.tsv).
    # If it isn't dropped here, the merge below produces cluster_rep_x /
    # cluster_rep_y instead of cluster_rep, and every downstream reference
    # to "cluster_rep" raises a KeyError. clusters_tsv is always the single
    # source of truth for cluster_rep within this function.
    if "cluster_rep" in master_df.columns:
        log("master_antibodies.csv already has a cluster_rep column -- "
            "dropping it before merge so the fresh assignment from "
            f"{clusters_tsv} is used, not a possibly-stale prior value.")
        master_df = master_df.drop(columns=["cluster_rep"])

    log(f"Loaded {len(master_df)} antibodies, {clusters_df['cluster_rep'].nunique()} CDR-H3 clusters")

    before = compute_diversity_snapshot(master_df)
    log(f"BEFORE dedup: {before}")

    dedup_df = pick_cluster_representatives(master_df, clusters_df)
    after = compute_diversity_snapshot(dedup_df)
    log(f"AFTER dedup:  {after}")

    delta = {}
    for k in before:
        if k in after and isinstance(before[k], (int, float)) and isinstance(after[k], (int, float)):
            if k == "n_antibodies":
                delta[k] = {"before": before[k], "after": after[k],
                            "reduction_fraction": 1 - (after[k] / max(before[k], 1))}
            else:
                abs_delta = after[k] - before[k]
                rel_delta = abs_delta / abs(before[k]) if before[k] not in (0, None) else None
                delta[k] = {"before": before[k], "after": after[k],
                            "absolute_delta": abs_delta, "relative_delta": rel_delta}

    summary = {
        "dedup_method": "one representative per CDR-H3 MMseqs2 cluster "
                         f"(min_seq_id={cfg['mmseqs2']['cdrh3_min_seq_id']}), "
                         "preferring has_antigen=True, deterministic pdb_id tie-break",
        "before": before,
        "after": after,
        "delta": delta,
    }

    out_path = os.path.join(work_dir, "tables", "rq3_before_after_dedup.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"Wrote {out_path}")

    dedup_csv = os.path.join(work_dir, "tables", "rq3_deduplicated_master.csv")
    dedup_df.to_csv(dedup_csv, index=False)
    log(f"Wrote {dedup_csv} ({len(dedup_df)} rows) — reusable directly as a "
        f"cluster-deduplicated benchmark split, not just an analysis byproduct.")
    return summary


# ═════════════════════════════════════════════════════════════════════════
# SECTION B — benchmark design checklist
# ═════════════════════════════════════════════════════════════════════════

def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def run_section_b(work_dir: str, cfg: dict) -> dict:
    log("=" * 70)
    log("SECTION B: benchmark design checklist")

    tables = os.path.join(work_dir, "tables")
    rq1_seq = _load_json(os.path.join(tables, "rq1_sequence_metadata_bias.json"))
    rq1_struct = _load_json(os.path.join(tables, "rq1_structural_redundancy_paratope.json"))
    rq2 = _load_json(os.path.join(tables, "rq2_oas_comparison_summary.json"))
    rq3a = _load_json(os.path.join(tables, "rq3_before_after_dedup.json"))

    recs = []

    # Rec 1: sequence-identity threshold for splitting
    if rq1_struct and "cdrh3_redundancy" in rq1_struct:
        r = rq1_struct["cdrh3_redundancy"]
        recs.append({
            "id": 1,
            "title": "Cluster-aware CDR-H3 splitting",
            "recommendation": (
                f"Split train/test by CDR-H3 sequence cluster, not by PDB ID. "
                f"At {r['min_seq_id_threshold']*100:.0f}% identity threshold, "
                f"{r['n_sequences']} SAbDab entries collapse into only "
                f"{r['n_unique_clusters']} clusters (compression "
                f"{r['compression_ratio']:.2f}x; "
                f"{r['singleton_fraction']*100:.1f}% are singletons). "
                f"A PDB-ID-only split allows near-duplicate CDR-H3 loops into "
                f"both train and test."
            ),
            "supporting_stat_source": "rq1_structural_redundancy_paratope.json:cdrh3_redundancy",
        })
    else:
        recs.append({"id": 1, "title": "Cluster-aware CDR-H3 splitting",
                      "recommendation": "INSUFFICIENT DATA — run rq1_sequence_structural_bias.py Section B first."})

    # Rec 2: antigen-balanced splitting, reusing existing antigen_cluster_id
    if rq1_struct and "antigen_redundancy_reused_from_existing_pipeline" in rq1_struct:
        r = rq1_struct["antigen_redundancy_reused_from_existing_pipeline"]
        recs.append({
            "id": 2,
            "title": "Antigen-cluster-balanced splitting",
            "recommendation": (
                f"Use antigen_cluster_id (already computed in the existing "
                f"preprocessing pipeline via MMseqs2) as the split key for "
                f"antigen-disjoint evaluation. The top 10 antigen clusters "
                f"account for {r['top10_cluster_dominance_fraction']*100:.1f}% "
                f"of antigen-bound complexes (Gini={r['gini']:.2f} across "
                f"{r['n_unique_antigen_clusters']} clusters) — a random split "
                f"risks leaking near-identical antigens across train/test."
            ),
            "supporting_stat_source": "rq1_structural_redundancy_paratope.json:antigen_redundancy_reused_from_existing_pipeline",
        })
    else:
        recs.append({"id": 2, "title": "Antigen-cluster-balanced splitting",
                      "recommendation": "INSUFFICIENT DATA — run rq1_sequence_structural_bias.py Section B first."})

    # Rec 3: report diversity metrics alongside any benchmark
    if rq1_seq and "length" in rq1_seq:
        l = rq1_seq["length"]
        recs.append({
            "id": 3,
            "title": "Report diversity metrics, not just size",
            "recommendation": (
                f"Any SAbDab-derived benchmark should report CDR-H3 length "
                f"entropy and Gini alongside sample count. This dataset's "
                f"full (non-deduplicated) length distribution has entropy="
                f"{l['entropy_bits']:.2f} bits (normalized={l['normalized_entropy']:.2f}) "
                f"and Gini={l['gini']:.2f} — a benchmark subset reporting "
                f"substantially lower entropy than this is measurably less diverse "
                f"than the source dataset, and that should be disclosed."
            ),
            "supporting_stat_source": "rq1_sequence_metadata_bias.json:length",
        })
    else:
        recs.append({"id": 3, "title": "Report diversity metrics, not just size",
                      "recommendation": "INSUFFICIENT DATA — run rq1_sequence_structural_bias.py Section A first."})

    # Rec 4: germline-family balance, with explicit resolution caveat
    if rq1_seq and "heavy_germline_family" in rq1_seq:
        g = rq1_seq["heavy_germline_family"]
        recs.append({
            "id": 4,
            "title": "Disclose germline-family balance (at available resolution)",
            "recommendation": (
                f"The top 5 heavy-chain germline FAMILIES (IGHV-level only — "
                f"allele/J-gene not available in SAbDab's summary export) "
                f"account for {g['top5_fraction_of_total']*100:.1f}% of entries "
                f"(Gini={g['gini']:.2f} across {g['n_unique_families']} families). "
                f"Benchmarks should report this family-level breakdown explicitly "
                f"and disclose that finer-grained allele bias is NOT captured by "
                f"this statistic."
            ),
            "supporting_stat_source": "rq1_sequence_metadata_bias.json:heavy_germline_family",
            "explicit_limitation": cfg["scope_notes"]["germline_resolution"],
        })
    else:
        recs.append({"id": 4, "title": "Disclose germline-family balance (at available resolution)",
                      "recommendation": "INSUFFICIENT DATA — run rq1_sequence_structural_bias.py Section A first."})

    # Rec 5: deduplication's measured effect on diversity
    if rq3a:
        delta = rq3a.get("delta", {})
        n_delta = delta.get("n_antibodies", {})
        recs.append({
            "id": 5,
            "title": "Apply cluster-aware deduplication before computing benchmark statistics",
            "recommendation": (
                f"Cluster-aware deduplication (one representative per CDR-H3 "
                f"cluster) removed {n_delta.get('reduction_fraction', 0)*100:.1f}% "
                f"of entries in this dataset and changed measured diversity "
                f"metrics by the amounts in rq3_before_after_dedup.json. "
                f"Any benchmark statistic (entropy, Gini, class balance) computed "
                f"BEFORE this dedup step is measuring redundancy, not diversity, "
                f"to a quantifiable degree — report both, or report only the "
                f"deduplicated numbers."
            ),
            "supporting_stat_source": "rq3_before_after_dedup.json:delta",
        })
    else:
        recs.append({"id": 5, "title": "Apply cluster-aware deduplication before computing benchmark statistics",
                      "recommendation": "INSUFFICIENT DATA — run Section A first."})

    # Rec 6: species scope discipline when comparing to repertoire data
    if rq2:
        recs.append({
            "id": 6,
            "title": "State species scope explicitly when benchmarking against repertoire data (e.g. OAS)",
            "recommendation": (
                f"When using OAS or similar repertoire data as a diversity "
                f"reference, state the species scope explicitly. "
                f"{cfg['scope_notes']['oas_species_scope']} "
                f"The length-distribution divergence measured here "
                f"(JS={rq2.get('length_jsd_sabdab_vs_oas_heavy', 'N/A')}) is only "
                f"valid under that scope restriction — re-introducing other "
                f"species on the SAbDab side without a matching OAS comparison "
                f"would silently break the comparison's validity."
            ),
            "supporting_stat_source": "rq2_oas_comparison_summary.json",
        })
    else:
        recs.append({"id": 6, "title": "State species scope explicitly when benchmarking against repertoire data (e.g. OAS)",
                      "recommendation": "INSUFFICIENT DATA — run rq2_oas_comparison.py first."})

    checklist = {"n_recommendations": len(recs), "recommendations": recs}
    json_path = os.path.join(tables, "rq3_benchmark_checklist.json")
    with open(json_path, "w") as f:
        json.dump(checklist, f, indent=2, default=str)
    log(f"Wrote {json_path}")

    md_lines = ["# Benchmark Design Checklist", "",
                "Generated from quantitative analysis — each item cites its "
                "supporting statistic file.", ""]
    for r in recs:
        md_lines.append(f"## {r['id']}. {r['title']}")
        md_lines.append("")
        md_lines.append(r["recommendation"])
        if "supporting_stat_source" in r:
            md_lines.append("")
            md_lines.append(f"*Source: `{r['supporting_stat_source']}`*")
        if "explicit_limitation" in r:
            md_lines.append("")
            md_lines.append(f"**Limitation:** {r['explicit_limitation']}")
        md_lines.append("")

    md_path = os.path.join(tables, "rq3_benchmark_checklist.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    log(f"Wrote {md_path}")
    return checklist


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--only", choices=["A", "B"], default=None,
                         help="Run only one section (A=before/after dedup, "
                              "B=benchmark checklist). Default: run both, A then B.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    work_dir = cfg["paths"]["work_dir"]
    os.makedirs(os.path.join(work_dir, "tables"), exist_ok=True)

    sections_to_run = [args.only] if args.only else ["A", "B"]

    if "A" in sections_to_run:
        run_section_a(work_dir, cfg)
    if "B" in sections_to_run:
        run_section_b(work_dir, cfg)

    log("RQ3 analysis complete.")


if __name__ == "__main__":
    main()