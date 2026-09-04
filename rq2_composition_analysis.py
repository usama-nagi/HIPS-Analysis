#!/usr/bin/env python3
"""
rq2_composition_analysis.py
============================
Rescopes the CDR-H3 amino-acid composition comparison (RQ2 Section C)
to human-only SAbDab entries, adds cluster-weighted alongside entry-weighted
estimates, and replaces per-residue z-tests with cluster-level bootstrap 95% CIs.

Reads:
  <work_dir>/tables/master_antibodies.csv     human SAbDab entries + h3_seq
  <work_dir>/tables/rq1_cdrh3_clusters.tsv    cluster_rep mapping (filename_stem -> cluster_rep)
  <work_dir>/tables/rq2*.json                 existing OAS-heavy composition proportions

Writes:
  <work_dir>/tables/composition_by_weighting_scheme.json

Usage
-----
    python scripts/rq2_composition_analysis.py --config configs/config.yaml
"""
import json, glob, sys, os, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config

AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
N_BOOTSTRAP = 2000
SEED = 42

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def composition_from_seqs(seqs):
    """Returns (frac_dict, total_residues) over AA_ALPHABET."""
    comp = Counter()
    for s in seqs:
        for aa in str(s):
            if aa in AA_ALPHABET:
                comp[aa] += 1
    total = sum(comp.values())
    if total == 0:
        return {aa: 0.0 for aa in AA_ALPHABET}, 0
    return {aa: comp.get(aa, 0) / total for aa in AA_ALPHABET}, total

# ── 1. Load human SAbDab entries ─────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None)
args = parser.parse_args()
cfg = load_config(args.config)
work_dir = cfg["paths"]["work_dir"]

MASTER = os.path.join(work_dir, "tables", "master_antibodies.csv")
CLUSTERS_TSV = os.path.join(work_dir, "tables", "rq1_cdrh3_clusters.tsv")

for p in [MASTER, CLUSTERS_TSV]:
    if not os.path.exists(p):
        sys.exit(f"ABORT: required file not found: {p}")

df = pd.read_csv(MASTER)
log(f"Loaded master: {len(df)} entries")

human = df[
    (df["heavy_species"].fillna("").str.lower() == "homo sapiens")
    & df["h3_seq"].notna()
].copy()
log(f"Human entries with h3_seq: {len(human)}")

# ── 2. Join cluster membership ────────────────────────────────────────────
clusters_df = pd.read_csv(CLUSTERS_TSV, sep="\t")
log(f"Cluster TSV: {len(clusters_df)} rows, columns: {list(clusters_df.columns)}")

# Expect columns: filename_stem, cluster_rep
if "filename_stem" not in clusters_df.columns or "cluster_rep" not in clusters_df.columns:
    sys.exit(f"ABORT: rq1_cdrh3_clusters.tsv must have filename_stem and cluster_rep columns. "
             f"Found: {list(clusters_df.columns)}")

human = human.merge(
    clusters_df[["filename_stem", "cluster_rep"]],
    on="filename_stem", how="left"
)
n_missing_cluster = human["cluster_rep"].isna().sum()
if n_missing_cluster > 0:
    log(f"WARNING: {n_missing_cluster} human entries have no cluster assignment -- "
        f"these will be excluded from cluster-weighted analysis.")
human_clust = human.dropna(subset=["cluster_rep"])
log(f"Human entries with cluster assignment: {len(human_clust)}")

# ── 3. Entry-weighted composition (human SAbDab) ──────────────────────────
entry_frac, entry_total = composition_from_seqs(human_clust["h3_seq"])
log(f"Entry-weighted: {entry_total:,} residues, {len(human_clust):,} sequences")

# ── 4. Cluster-weighted composition (equal weight per CDR-H3 cluster) ─────
# For each cluster: compute its mean composition profile (proportions).
# Overall cluster-weighted estimate = mean of per-cluster profiles.
groups = human_clust.groupby("cluster_rep")
cluster_ids = list(groups.groups.keys())
n_clusters = len(cluster_ids)

per_cluster_profiles = {}
for cid, grp in groups:
    frac, _ = composition_from_seqs(grp["h3_seq"].tolist())
    per_cluster_profiles[cid] = frac

def mean_profile(profiles):
    return {aa: float(np.mean([p[aa] for p in profiles])) for aa in AA_ALPHABET}

cluster_frac = mean_profile(list(per_cluster_profiles.values()))
log(f"Cluster-weighted: {n_clusters:,} unique clusters")

# ── 5. Bootstrap CIs (cluster-level resampling, 2000 iterations) ──────────
rng = np.random.default_rng(SEED)
cid_arr = np.array(cluster_ids, dtype=object)

boot_entry = {aa: [] for aa in AA_ALPHABET}
boot_cluster = {aa: [] for aa in AA_ALPHABET}

log(f"Running {N_BOOTSTRAP} bootstrap iterations (cluster-level resampling)...")
for b in range(N_BOOTSTRAP):
    sampled = rng.choice(cid_arr, size=n_clusters, replace=True)

    # Entry-weighted bootstrap: pool all sequences in sampled clusters
    all_seqs = []
    profiles_b = []
    for cid in sampled:
        all_seqs.extend(groups.get_group(cid)["h3_seq"].dropna().tolist())
        profiles_b.append(per_cluster_profiles[cid])

    ef, _ = composition_from_seqs(all_seqs)
    for aa in AA_ALPHABET:
        boot_entry[aa].append(ef[aa])
        boot_cluster[aa].append(np.mean([p[aa] for p in profiles_b]))

    if (b + 1) % 400 == 0:
        log(f"  Bootstrap: {b+1}/{N_BOOTSTRAP}")

log("Bootstrap complete.")

# ── 6. Load OAS-heavy proportions from existing rq2 output ───────────────
oas_frac = None
oas_total_residues = None
oas_source_file = None

rq2_candidates = sorted(glob.glob(os.path.join(work_dir, "tables", "rq2*.json")))
log(f"rq2 JSON candidates: {rq2_candidates}")

for path in rq2_candidates:
    try:
        with open(path) as f:
            rq2 = json.load(f)
        # Navigate through possible nesting structures
        comp_block = (
            rq2.get("oas_vs_sabdab_h3_aa_composition")
            or rq2.get("C")
            or rq2.get("composition")
        )
        if comp_block and "per_aa_comparison" in comp_block:
            oas_frac = {r["aa"]: r["oas_frac"] for r in comp_block["per_aa_comparison"]}
            oas_total_residues = comp_block.get("oas_total_residues")
            oas_source_file = path
            log(f"Loaded OAS proportions from {path} "
                f"(oas_total_residues={oas_total_residues})")
            break
    except Exception as e:
        log(f"  Could not parse {path}: {e}")

if oas_frac is None:
    log("WARNING: OAS proportions not found. "
        "Re-run: python3 scripts/rq2_oas_comparison.py --config configs/config.yaml --only C")
    oas_frac = {aa: None for aa in AA_ALPHABET}

# ── 7. Assemble per-aa results ────────────────────────────────────────────
per_aa = []
for aa in AA_ALPHABET:
    ew     = entry_frac[aa]
    cw     = cluster_frac[aa]
    ew_lo  = float(np.percentile(boot_entry[aa], 2.5))
    ew_hi  = float(np.percentile(boot_entry[aa], 97.5))
    cw_lo  = float(np.percentile(boot_cluster[aa], 2.5))
    cw_hi  = float(np.percentile(boot_cluster[aa], 97.5))
    of     = oas_frac.get(aa)

    def fold(num, denom):
        return round(num / denom, 4) if denom else None

    # CI excludes 1 → significant under cluster-bootstrap inference
    entry_sig   = (ew_lo > of or ew_hi < of) if of else None
    cluster_sig = (cw_lo > of or cw_hi < of) if of else None

    per_aa.append({
        "aa": aa,
        "entry_frac":          round(ew, 6),
        "entry_ci_lo":         round(ew_lo, 6),
        "entry_ci_hi":         round(ew_hi, 6),
        "cluster_frac":        round(cw, 6),
        "cluster_ci_lo":       round(cw_lo, 6),
        "cluster_ci_hi":       round(cw_hi, 6),
        "oas_frac":            round(of, 6) if of else None,
        "fold_entry":          fold(ew, of),
        "fold_entry_ci_lo":    fold(ew_lo, of),
        "fold_entry_ci_hi":    fold(ew_hi, of),
        "fold_cluster":        fold(cw, of),
        "fold_cluster_ci_lo":  fold(cw_lo, of),
        "fold_cluster_ci_hi":  fold(cw_hi, of),
        "entry_sig_95":        entry_sig,
        "cluster_sig_95":      cluster_sig,
    })

# Sort by entry-weighted fold-change descending
per_aa_sorted = sorted(per_aa, key=lambda x: x["fold_entry"] or 0, reverse=True)

out = {
    "summary": {
        "n_human_entries":      int(len(human_clust)),
        "entry_total_residues": int(entry_total),
        "n_clusters_human":     int(n_clusters),
        "n_bootstrap":          N_BOOTSTRAP,
        "seed":                 SEED,
        "oas_source_file":      oas_source_file,
        "oas_total_residues":   int(oas_total_residues) if oas_total_residues else None,
    },
    "per_aa": per_aa_sorted,
}

out_path = os.path.join(work_dir, "tables", "composition_by_weighting_scheme.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
log(f"Wrote {out_path}")

# ── 8. Pretty-print table for copy-paste ─────────────────────────────────
print()
print(f"{'AA':>3}  {'EW':>7}  {'EW 95% CI':>18}  {'CW':>7}  {'CW 95% CI':>18}  "
      f"{'OAS':>7}  {'FC_EW':>7}  {'FC_CW':>7}  {'Sig_EW':>7}  {'Sig_CW':>7}")
print("-" * 105)
for r in per_aa_sorted:
    of = r["oas_frac"]
    print(
        f"{r['aa']:>3}  "
        f"{r['entry_frac']:>7.4f}  "
        f"[{r['entry_ci_lo']:.4f},{r['entry_ci_hi']:.4f}]  "
        f"{r['cluster_frac']:>7.4f}  "
        f"[{r['cluster_ci_lo']:.4f},{r['cluster_ci_hi']:.4f}]  "
        f"{of:>7.4f}  "
        f"{r['fold_entry']:>7.3f}  "
        f"{r['fold_cluster']:>7.3f}  "
        f"{'YES' if r['entry_sig_95'] else 'no':>7}  "
        f"{'YES' if r['cluster_sig_95'] else 'no':>7}"
    )
