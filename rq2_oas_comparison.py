#!/usr/bin/env python3
"""
rq2_oas_comparison.py
======================
RQ2: How representative is SAbDab compared to large-scale natural antibody
repertoires?

Three sections, mirroring RQ1's lettered-section convention:

  SECTION A — CDR-H3 length & clonotype diversity   [ORIGINAL, UNCHANGED]
      Compares SAbDab against natural repertoire diversity using OAS,
      restricted to what's actually possible given the preprocessed OAS
      mmap's real schema (confirmed from its meta.json -- see scope_notes
      in config.yaml):
        - OAS mmap is human-only (species is hardcoded "human" by the
          loader, not a stored per-sample field) and stores NO germline/
          V-gene calls (v_gene_check in hard_filters was a preprocessing
          FILTER, not a retained annotation).
        - Therefore: SAbDab side is restricted to heavy_species ==
          'homo sapiens' for a fair comparison, and the comparison covers
          CDR-H3 LENGTH and CLONOTYPE DIVERSITY only -- not germline family.

  SECTION B — SAbDab-side clonotype-equivalent diversity (PROXY)
      (folded in from the standalone rq2_extended_diversity.py, Part A)
      OAS's clonotype definition (germline V + germline J + CDR3) is NOT
      computable for SAbDab -- the summary table has no J-gene column at
      all, and germline is family-level only. This section reports a
      clearly-labeled, NON-STANDARD proxy instead: (heavy_subclass family,
      CDR-H3 cluster_rep) as a grouping key. Reported side by side with
      Section A's oas_heavy_clonotype_diversity for context ONLY, never as
      a directly comparable number -- see the "label" field in the output
      JSON, which restates this every time the number is read.
      REQUIRES master_antibodies.csv to have a cluster_rep column (merged
      in from rq1_cdrh3_clusters.tsv -- see migrate_master_csv.py
      --only cluster_rep if it doesn't).

  SECTION C — OAS vs SAbDab CDR-H3 amino acid composition
      (folded in from the standalone rq2_extended_diversity.py, Part B)
      Extends Section A's length-only comparison to full amino-acid
      composition. Reuses the same direct-memmap-read approach as Section
      A (tokens.bin / lengths.bin / sources.bin / cdr_full_mask.bin), so
      this section has the SAME no-dependency property as the rest of the
      script -- no import of oas_loader.py.
      REQUIRES master_antibodies.csv to have an h3_seq column (isolated
      CDR-H3 sequence per row -- see migrate_master_csv.py --only h3_seq
      if it doesn't).

This script reads the OAS mmap arrays DIRECTLY (np.memmap + meta.json)
using its own independent code path -- it does NOT import
data/loaders/oas_loader.py, per the "clean new files, no dependency on
existing code" requirement. The binary layout it depends on is documented
inline below and should be re-verified against oas_loader.py if that
loader's mmap format ever changes.

Output
------
outputs/tables/rq2_oas_comparison_summary.json     (Section A)
outputs/tables/rq2_length_distributions.csv        (Section A)
outputs/tables/rq2_extended_diversity.json         (Sections B + C)

Usage
-----
    python scripts/rq2_oas_comparison.py --config configs/config.yaml
    python scripts/rq2_oas_comparison.py --config configs/config.yaml --only A
    python scripts/rq2_oas_comparison.py --config configs/config.yaml --only B
    python scripts/rq2_oas_comparison.py --config configs/config.yaml --only C --oas_sample_n 2000000

NOTE ON RUN TIME: Section C samples up to --oas_sample_n (default 2,000,000)
OAS sequences and isolates CDR-H3 per sequence in a Python loop. Sample
indices are sorted before iteration to get near-sequential memmap access
(random-order access was confirmed to make this dramatically slower with
no progress visibility). Progress is logged every ~5% of the sample.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    load_config, require_path, validate_oas_meta, shannon_entropy,
    normalized_entropy, simpson_diversity, jensen_shannon_divergence,
    align_distributions_on_support, setup_rng, log,
)

AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")

DEFAULT_OAS_VOCAB = {
    0: "<pad>", 1: "<unk>", 2: "<s>", 3: "</s>", 4: "<mask>",
    5: "A", 6: "C", 7: "D", 8: "E", 9: "F", 10: "G", 11: "H", 12: "I",
    13: "K", 14: "L", 15: "M", 16: "N", 17: "P", 18: "Q", 19: "R",
    20: "S", 21: "T", 22: "V", 23: "W", 24: "Y",
}
# Verified against the real tokenizer's build_vocab(): toks = [PAD, UNK,
# START, END, MASK] + list("ACDEFGHIKLMNPQRSTVWY"), i.e. 5 special tokens
# before the amino acid block starts at ID 5. Any count-based sanity check
# on this mapping should treat a common residue (e.g. alanine) coming out
# at exactly 0% as a strong signal the vocab offset is wrong: an
# off-by-one error here still lets every count sum to oas_total_residues
# correctly, since it just shifts each amino acid onto its alphabetical
# neighbor's token, so totals alone will not catch it.


# ═════════════════════════════════════════════════════════════════════════
# SECTION A — CDR-H3 length & clonotype diversity
# ═════════════════════════════════════════════════════════════════════════

def find_oas_split_dirs(oas_root: str) -> list:
    """
    Returns a list of directories whose union constitutes the full OAS pool
    this project's paper describes (~355.8M samples). Preference order:

      1. oas_root/meta.json directly, if present -- a single combined pool.
         Returns [oas_root].
      2. oas_root/train/meta.json + oas_root/test/meta.json, if both are
         present. The combined root-level files were removed for storage
         after this train/test partition was created; train+test together
         losslessly reconstitute the original population, since this is
         the same data split in two, not two independent subsets. Reading
         both and combining is therefore equivalent to reading the
         original root pool, not an approximation of it. Returns
         [train_dir, test_dir].
      3. oas_root/train/meta.json alone, if test/ is unavailable. This is a
         genuine subset (~99.5% of the pool, ~0.5% held out, per
         train/meta.json's total_samples against the root's) -- last
         resort only, with an explicit warning.

    Every caller loops over (dir, meta) pairs and combines results via
    proportional allocation (see _proportional_split_allocation below),
    rather than reading one directory and treating it as the whole
    population. Both Section A and Section C call this one function, so
    they cannot diverge from each other.
    """
    if os.path.exists(os.path.join(oas_root, "meta.json")):
        return [oas_root]

    train_dir = os.path.join(oas_root, "train")
    test_dir = os.path.join(oas_root, "val")
    train_has_meta = os.path.exists(os.path.join(train_dir, "meta.json"))
    test_has_meta = os.path.exists(os.path.join(test_dir, "meta.json"))

    if train_has_meta and test_has_meta:
        log(f"No combined meta.json at {oas_root}; reading {train_dir} + "
            f"{test_dir} together as the full pool (a lossless train/test "
            f"partition of the original combined files, not independent "
            f"subsets).")
        return [train_dir, test_dir]

    if train_has_meta:
        log(f"WARNING: no meta.json found directly in {oas_root}, and no "
            f"test/ split alongside train/; falling back to {train_dir} "
            f"ALONE, which is a genuine SUBSET (~99.5%) of the full OAS "
            f"pool, not the complete population. Verify this is "
            f"intentional before trusting downstream statistics.")
        return [train_dir]

    raise FileNotFoundError(
        f"No meta.json found in {oas_root}, {train_dir}, or {test_dir}. "
        f"Check paths.oas_mmap_root in config.yaml."
    )


def _load_split_metas(split_dirs: list) -> list:
    """Load meta.json for each split dir and cross-validate schema
    consistency -- if source_map or max_len ever disagree between splits,
    combining their arrays index-for-index would silently produce nonsense,
    so this raises loudly instead."""
    metas = []
    for d in split_dirs:
        with open(os.path.join(d, "meta.json")) as f:
            metas.append(json.load(f))
    if len(metas) > 1:
        ref_source_map, ref_max_len = metas[0].get("source_map"), metas[0].get("max_len")
        for d, m in zip(split_dirs[1:], metas[1:]):
            if m.get("source_map") != ref_source_map:
                raise ValueError(
                    f"source_map mismatch between {split_dirs[0]} and {d}: "
                    f"{ref_source_map} vs {m.get('source_map')} -- cannot "
                    f"safely combine splits with different source-id schemas."
                )
            if m.get("max_len") != ref_max_len:
                raise ValueError(
                    f"max_len mismatch between {split_dirs[0]} and {d}: "
                    f"{ref_max_len} vs {m.get('max_len')}"
                )
    return metas


def _proportional_split_allocation(sizes: list, n_requested: int) -> list:
    """Given each split's pool size and a total sample size requested
    across all splits combined, allocate how many to draw from each split
    proportional to its share of the combined pool -- this is what makes
    per-split sampling statistically equivalent to simple random sampling
    from the true union, without ever materializing a concatenated array
    of the underlying (300M+ row) binary files. Largest-remainder rounding
    so the allocated total exactly equals n_requested (pool size permitting)."""
    total = sum(sizes)
    if total == 0 or n_requested <= 0:
        return [0] * len(sizes)
    raw = [n_requested * s / total for s in sizes]
    alloc = [int(np.floor(r)) for r in raw]
    remainder = n_requested - sum(alloc)
    order = sorted(range(len(sizes)), key=lambda i: raw[i] - alloc[i], reverse=True)
    for i in order[:remainder]:
        alloc[i] += 1
    return [min(a, s) for a, s in zip(alloc, sizes)]


def load_oas_cdr3_lengths(split_dirs: list, metas: list, rng: np.random.Generator,
                           n_heavy_sample: int, n_paired_sample: int) -> dict:
    """Accepts a list of split directories/metas and draws a
    proportionally-stratified sample across all of them (see
    _proportional_split_allocation) -- statistically equivalent to simple
    random sampling from the true union of every split's pool."""
    per_split_heavy_idx, per_split_paired_idx, per_split_cdr3len = [], [], []

    for d, m in zip(split_dirs, metas):
        total = m["total_samples"]
        cdr3len_path = os.path.join(d, "cdr3_len.bin")
        sources_path = os.path.join(d, "sources.bin")
        require_path(cdr3len_path, f"OAS cdr3_len.bin ({d})")
        require_path(sources_path, f"OAS sources.bin ({d})")

        cdr3_lens = np.memmap(cdr3len_path, dtype=np.uint8, mode="r", shape=(total,))
        sources = np.memmap(sources_path, dtype=np.uint8, mode="r", shape=(total,))
        source_map = m["source_map"]
        heavy_id, paired_id = source_map.get("heavy"), source_map.get("paired")

        heavy_idx = np.flatnonzero(sources == heavy_id) if heavy_id is not None else np.array([], dtype=np.int64)
        paired_idx = np.flatnonzero(sources == paired_id) if paired_id is not None else np.array([], dtype=np.int64)

        per_split_heavy_idx.append(heavy_idx)
        per_split_paired_idx.append(paired_idx)
        per_split_cdr3len.append(cdr3_lens)

    n_heavy_total = sum(len(x) for x in per_split_heavy_idx)
    n_paired_total = sum(len(x) for x in per_split_paired_idx)
    log(f"OAS pool sizes (combined across {len(split_dirs)} split(s)) — "
        f"heavy: {n_heavy_total:,} | paired: {n_paired_total:,}")

    heavy_alloc = _proportional_split_allocation([len(x) for x in per_split_heavy_idx], n_heavy_sample)
    paired_alloc = _proportional_split_allocation([len(x) for x in per_split_paired_idx], n_paired_sample)

    heavy_lens_parts, paired_lens_parts = [], []
    n_heavy_sampled = n_paired_sampled = 0

    for heavy_idx, paired_idx, cdr3_lens, n_h, n_p in zip(
        per_split_heavy_idx, per_split_paired_idx, per_split_cdr3len, heavy_alloc, paired_alloc
    ):
        if n_h > 0:
            h_idx = rng.choice(heavy_idx, size=n_h, replace=False)
            heavy_lens_parts.append(np.array(cdr3_lens[h_idx], dtype=np.int64))
            n_heavy_sampled += n_h
        if n_p > 0:
            p_idx = rng.choice(paired_idx, size=n_p, replace=False)
            paired_lens_parts.append(np.array(cdr3_lens[p_idx], dtype=np.int64))
            n_paired_sampled += n_p

    heavy_lens = np.concatenate(heavy_lens_parts) if heavy_lens_parts else np.array([], dtype=np.int64)
    paired_lens = np.concatenate(paired_lens_parts) if paired_lens_parts else np.array([], dtype=np.int64)

    return {
        "heavy_lens": heavy_lens, "paired_lens": paired_lens,
        "n_heavy_sampled": n_heavy_sampled, "n_paired_sampled": n_paired_sampled,
        "n_heavy_total": n_heavy_total, "n_paired_total": n_paired_total,
    }


def load_oas_clonotype_diversity(split_dirs: list, metas: list,
                                  per_split_sample_idx: list) -> dict:
    """per_split_sample_idx is a list, same order as split_dirs, of the
    index arrays already drawn for each split (reuses whatever heavy
    sample was allocated upstream rather than resampling independently, so
    this describes the same sequences Section A's length comparison used).
    Uniqueness is computed on the combined array of hashes, not summed
    per-split then added -- a clonotype could in principle appear in both
    train and test, and checking per-split then summing would double-count
    any hash that does."""
    parts = []
    for d, m, idx in zip(split_dirs, metas, per_split_sample_idx):
        total = m["total_samples"]
        clono_path = os.path.join(d, "clono_hash.bin")
        if not os.path.exists(clono_path):
            log(f"WARNING: clono_hash.bin not found in {d} — skipping clonotype diversity for OAS")
            return None
        if len(idx) == 0:
            continue
        clono = np.memmap(clono_path, dtype=np.uint32, mode="r", shape=(total,))
        parts.append(np.array(clono[idx], dtype=np.uint32))

    if not parts:
        return None
    combined = np.concatenate(parts)
    n_unique = len(np.unique(combined))
    return {
        "n_sampled": len(combined),
        "n_unique_clonotypes": int(n_unique),
        "unique_fraction": float(n_unique / max(len(combined), 1)),
    }


def run_section_a(work_dir: str, cfg: dict) -> dict:
    log("=" * 70)
    log("SECTION A: CDR-H3 length & clonotype diversity")

    cdr3_min = cfg["filters"]["cdr3_min_len"]
    cdr3_max = cfg["filters"]["cdr3_max_len"]

    master_csv = os.path.join(work_dir, "tables", "master_antibodies.csv")
    require_path(master_csv, "master_antibodies.csv (run 00_build_dataset.py first)")
    df = pd.read_csv(master_csv)

    df_human = df[df["heavy_species"].fillna("").str.strip().str.lower() == "homo sapiens"]
    df_human = df_human[(df_human["cdr3_len_actual"] >= cdr3_min) &
                        (df_human["cdr3_len_actual"] <= cdr3_max)]
    log(f"SAbDab human-only, length-filtered: {len(df_human)}/{len(df)} entries")

    sabdab_len_counts = (df_human["cdr3_len_actual"].astype(int)
                         .value_counts().sort_index())

    oas_root = cfg["paths"]["oas_mmap_root"]
    require_path(oas_root, "OAS mmap root")
    oas_split_dirs = find_oas_split_dirs(oas_root)
    oas_metas = _load_split_metas(oas_split_dirs)
    total_samples_combined = sum(m["total_samples"] for m in oas_metas)
    for d, m in zip(oas_split_dirs, oas_metas):
        validate_oas_meta(m, os.path.join(d, "meta.json"))
    log(f"OAS meta: {total_samples_combined:,} total samples across {len(oas_split_dirs)} split(s)")

    rng = setup_rng(cfg["sampling"]["oas_subsample_seed"])
    n_heavy = cfg["sampling"]["oas_subsample_n_heavy"]
    n_paired = cfg["sampling"]["oas_subsample_n_paired"]

    oas_data = load_oas_cdr3_lengths(oas_split_dirs, oas_metas, rng, n_heavy, n_paired)

    def _filtered_counts(lens):
        lens = lens[(lens >= cdr3_min) & (lens <= cdr3_max)]
        return pd.Series(lens).value_counts().sort_index() if len(lens) else pd.Series(dtype=int)

    oas_heavy_counts = _filtered_counts(oas_data["heavy_lens"])
    oas_paired_counts = _filtered_counts(oas_data["paired_lens"])

    summary = {
        "scope_note": cfg["scope_notes"]["oas_germline_scope"],
        "species_scope_note": cfg["scope_notes"]["oas_species_scope"],
        "sabdab_human_n": int(len(df_human)),
        "oas_heavy_sampled_n": oas_data["n_heavy_sampled"],
        "oas_paired_sampled_n": oas_data["n_paired_sampled"],
        "oas_heavy_total_pool": oas_data["n_heavy_total"],
        "oas_paired_total_pool": oas_data["n_paired_total"],
    }

    for label, oas_counts in [("oas_heavy", oas_heavy_counts), ("oas_paired", oas_paired_counts)]:
        if len(oas_counts) == 0:
            continue
        support, a, b = align_distributions_on_support(
            sabdab_len_counts.to_dict(), oas_counts.to_dict())
        jsd = jensen_shannon_divergence(a, b)
        summary[f"length_jsd_sabdab_vs_{label}"] = jsd
        summary[f"{label}_length_mean"] = float(np.average(
            [int(k) for k in oas_counts.index], weights=oas_counts.values))
        summary[f"{label}_length_entropy_bits"] = shannon_entropy(oas_counts.values)

    summary["sabdab_human_length_mean"] = float(sabdab_len_counts.index.to_series()
                                                  .repeat(sabdab_len_counts.values).mean())
    summary["sabdab_human_length_entropy_bits"] = shannon_entropy(sabdab_len_counts.values)

    all_support = sorted(set(sabdab_len_counts.index) | set(oas_heavy_counts.index)
                          | set(oas_paired_counts.index))
    out_df = pd.DataFrame({
        "cdr3_length": all_support,
        "sabdab_human_count": [int(sabdab_len_counts.get(k, 0)) for k in all_support],
        "oas_heavy_count": [int(oas_heavy_counts.get(k, 0)) for k in all_support],
        "oas_paired_count": [int(oas_paired_counts.get(k, 0)) for k in all_support],
    })
    out_df.to_csv(os.path.join(work_dir, "tables", "rq2_length_distributions.csv"), index=False)

    rng2 = setup_rng(cfg["sampling"]["oas_subsample_seed"])
    per_split_heavy_idx = []
    for d, m in zip(oas_split_dirs, oas_metas):
        total = m["total_samples"]
        sources = np.memmap(os.path.join(d, "sources.bin"), dtype=np.uint8, mode="r", shape=(total,))
        heavy_id = m["source_map"].get("heavy")
        h_idx = np.flatnonzero(sources == heavy_id) if heavy_id is not None else np.array([], dtype=np.int64)
        per_split_heavy_idx.append(h_idx)
    heavy_alloc = _proportional_split_allocation([len(x) for x in per_split_heavy_idx], n_heavy)
    per_split_sample_idx = [
        rng2.choice(h_idx, size=n_h, replace=False) if n_h > 0 else np.array([], dtype=np.int64)
        for h_idx, n_h in zip(per_split_heavy_idx, heavy_alloc)
    ]
    clono_result = load_oas_clonotype_diversity(oas_split_dirs, oas_metas, per_split_sample_idx)
    if clono_result:
        summary["oas_heavy_clonotype_diversity"] = clono_result

    out_path = os.path.join(work_dir, "tables", "rq2_oas_comparison_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"Wrote {out_path}")
    return summary


# ═════════════════════════════════════════════════════════════════════════
# SECTION B — SAbDab-side clonotype-equivalent diversity (PROXY)
#             [FOLDED IN FROM rq2_extended_diversity.py, Part A]
# ═════════════════════════════════════════════════════════════════════════

def run_section_b(work_dir: str, master_csv_override: str = None) -> dict:
    log("=" * 70)
    log("SECTION B: SAbDab-side clonotype-equivalent diversity (NON-STANDARD PROXY)")

    master_path = master_csv_override or os.path.join(work_dir, "tables", "master_antibodies.csv")
    require_path(master_path, "master_antibodies.csv")
    df = pd.read_csv(master_path)

    required = {"heavy_subclass", "cluster_rep"}
    missing = required - set(df.columns)
    if missing:
        result = {
            "status": "SKIPPED",
            "reason": (
                f"master_antibodies.csv missing columns {missing}. cluster_rep must be "
                f"merged onto the full master table from rq1_cdrh3_clusters.tsv -- see "
                f"migrate_master_csv.py --only cluster_rep. Do not point this at "
                f"rq3_deduplicated_master.csv instead: that file already has cluster_rep "
                f"but describes the deduplicated cluster-representative subset, not the "
                f"full master table, and using it here would change what this statistic "
                f"means. Check the current row counts directly rather than assuming a "
                f"specific figure, since they change as the pipeline is rerun."
            ),
        }
    else:
        proxy_key = df["heavy_subclass"].astype(str) + "||" + df["cluster_rep"].astype(str)
        counts = proxy_key.value_counts()
        result = {
            "status": "OK",
            "label": "NON-STANDARD PROXY -- NOT a real clonotype statistic. "
                     "Grouping key = (heavy_subclass family, CDR-H3 90%-identity cluster_rep). "
                     "Not comparable to OAS's germline-V + germline-J + CDR3 clonotype "
                     "definition; OAS has no J-gene data and SAbDab germline is family-level "
                     "only, so no real clonotype statistic is computable for either side on "
                     "a shared definition. Report side by side with Section A's "
                     "oas_heavy_clonotype_diversity for context only, never as a directly "
                     "comparable number.",
            "n_entries": len(df),
            "n_unique_proxy_groups": int(counts.shape[0]),
            "unique_fraction": float(counts.shape[0] / len(df)),
            "top10_largest_groups": counts.head(10).to_dict(),
        }

    log(f"Section B: {result['status']}")
    if result["status"] == "OK":
        log(f"  unique_fraction = {result['unique_fraction']:.3f} "
            f"({result['n_unique_proxy_groups']} groups / {result['n_entries']} entries)")
    else:
        log(f"  {result['reason']}")
    return result


# ═════════════════════════════════════════════════════════════════════════
# SECTION C — OAS vs SAbDab CDR-H3 amino acid composition
#             [FOLDED IN FROM rq2_extended_diversity.py, Part B]
# ═════════════════════════════════════════════════════════════════════════

def run_section_c(work_dir: str, cfg: dict, oas_sample_n: int, seed: int,
                   master_csv_override: str = None) -> dict:
    log("=" * 70)
    log(f"SECTION C: OAS vs SAbDab CDR-H3 amino acid composition "
        f"(requesting {oas_sample_n:,} OAS sequences)")

    master_path = master_csv_override or os.path.join(work_dir, "tables", "master_antibodies.csv")
    require_path(master_path, "master_antibodies.csv")
    df = pd.read_csv(master_path)

    if "h3_seq" not in df.columns or "cdr3_len_actual" not in df.columns:
        return {
            "status": "SKIPPED",
            "reason": (
                "No h3_seq column in master_antibodies.csv. This section needs an "
                "isolated CDR-H3-only sequence per row to build a SAbDab-side AA "
                "composition comparable to OAS H3 -- see migrate_master_csv.py "
                "--only h3_seq."
            ),
        }

    sabdab_h3_seqs = df["h3_seq"].dropna().tolist()
    sabdab_composition = Counter()
    for seq in sabdab_h3_seqs:
        for aa in seq:
            if aa in AA_ALPHABET:
                sabdab_composition[aa] += 1

    # Uses the same find_oas_split_dirs() function Section A uses, so both
    # sections are structurally guaranteed to read the same OAS population
    # rather than resolving the directory independently and risking drift
    # between sections. The function returns a list of split directories,
    # combined here the same proportionally-stratified way Section A does.
    oas_mmap_root_cfg = cfg["paths"]["oas_mmap_root"]
    try:
        oas_split_dirs = find_oas_split_dirs(oas_mmap_root_cfg)
    except FileNotFoundError as e:
        return {"status": "SKIPPED", "reason": str(e)}
    oas_metas = _load_split_metas(oas_split_dirs)

    total_combined = sum(m["total_samples"] for m in oas_metas)
    max_len = oas_metas[0]["max_len"]
    log(f"OAS mmap metadata: total_samples={total_combined:,} "
        f"(across {len(oas_split_dirs)} split(s)), max_len={max_len}")
    source_map = oas_metas[0]["source_map"]
    heavy_source_id = source_map["heavy"]

    per_split_arrays = []
    per_split_heavy_idx = []
    for d, m in zip(oas_split_dirs, oas_metas):
        total = m["total_samples"]
        tokens = np.memmap(os.path.join(d, "tokens.bin"), dtype=np.int16,
                            mode="r", shape=(total, max_len))
        lengths = np.memmap(os.path.join(d, "lengths.bin"), dtype=np.int16,
                             mode="r", shape=(total,))
        sources = np.memmap(os.path.join(d, "sources.bin"), dtype=np.uint8,
                             mode="r", shape=(total,))
        cdr_full_mask = np.memmap(os.path.join(d, "cdr_full_mask.bin"), dtype=np.uint8,
                                   mode="r", shape=(total, max_len))
        per_split_arrays.append((tokens, lengths, cdr_full_mask))
        heavy_idx = np.where(sources == heavy_source_id)[0]
        per_split_heavy_idx.append(heavy_idx)

    rng = np.random.default_rng(seed)
    heavy_sizes = [len(x) for x in per_split_heavy_idx]
    alloc = _proportional_split_allocation(heavy_sizes, oas_sample_n)
    log(f"OAS heavy pool: {sum(heavy_sizes):,} sequences available (combined "
        f"across {len(oas_split_dirs)} split(s)), sampling {sum(alloc):,}")

    oas_composition = Counter()
    n_isolation_failures = 0
    n_processed = 0
    n_total_sampled = sum(alloc)
    report_every = max(1, n_total_sampled // 20)

    for split_i, ((tokens, lengths, cdr_full_mask), heavy_idx, n_this_split) in enumerate(
        zip(per_split_arrays, per_split_heavy_idx, alloc)
    ):
        if n_this_split == 0:
            continue
        sampled_idx = rng.choice(heavy_idx, size=n_this_split, replace=False)
        sampled_idx = np.sort(sampled_idx)
        log(f"Sorted sample indices for split {split_i} ({oas_split_dirs[split_i]}) "
            f"for sequential memmap access (index range: {sampled_idx[0]:,} to {sampled_idx[-1]:,})")

        for i in sampled_idx:
            seq_len = int(lengths[i])
            full_mask = cdr_full_mask[i, :seq_len]
            cdr_idx = np.where(full_mask)[0]
            if len(cdr_idx) == 0:
                n_isolation_failures += 1
                n_processed += 1
                continue
            blocks = []
            start = cdr_idx[0]
            prev = cdr_idx[0]
            for idx in cdr_idx[1:]:
                if idx != prev + 1:
                    blocks.append((start, prev))
                    start = idx
                prev = idx
            blocks.append((start, prev))
            h3_start, h3_end = blocks[-1]

            token_ids = tokens[i, h3_start:h3_end + 1]
            for tid in token_ids:
                aa = DEFAULT_OAS_VOCAB.get(int(tid))
                if aa in AA_ALPHABET:
                    oas_composition[aa] += 1

            n_processed += 1
            if n_processed % report_every == 0:
                log(f"  ...{n_processed:,}/{n_total_sampled:,} OAS sequences processed")

    sabdab_total = sum(sabdab_composition.values())
    oas_total = sum(oas_composition.values())

    comparison = []
    for aa in AA_ALPHABET:
        sc = sabdab_composition.get(aa, 0)
        oc = oas_composition.get(aa, 0)
        sf = sc / sabdab_total if sabdab_total else 0
        of = oc / oas_total if oas_total else 0
        comparison.append({
            "aa": aa,
            "sabdab_count": sc, "sabdab_frac": sf,
            "oas_count": oc, "oas_frac": of,
            "fold_sabdab_over_oas": (sf / of) if of > 0 else None,
        })

    return {
        "status": "OK",
        "vocab_used": "DEFAULT_OAS_VOCAB, verified against the real tokenizer's "
                       "build_vocab() (5 special tokens [PAD,UNK,START,END,MASK] at IDs "
                       "0-4, amino acids A-Y at IDs 5-24). If this tokenizer's vocab is "
                       "ever rebuilt with a different special-token set, re-verify this "
                       "mapping before trusting this section again: an incorrect offset "
                       "produces a plausible-looking but wrong result, so a common amino "
                       "acid such as alanine coming out at exactly 0% is a reliable sign "
                       "the vocab mapping is wrong.",
        "n_sabdab_h3_sequences": len(sabdab_h3_seqs),
        "n_oas_sampled": int(n_total_sampled),
        "n_oas_isolation_failures": n_isolation_failures,
        "sabdab_total_residues": sabdab_total,
        "oas_total_residues": oas_total,
        "per_aa_comparison": comparison,
    }


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--only", choices=["A", "B", "C"], default=None,
                         help="Run only one section (A=length/clonotype, B=SAbDab "
                              "clonotype proxy, C=AA composition). Default: A,B,C.")
    parser.add_argument("--master_csv", default=None,
                         help="Override path to master_antibodies.csv for Sections B/C. "
                              "Default: <work_dir>/tables/master_antibodies.csv")
    parser.add_argument("--oas_sample_n", type=int, default=2_000_000,
                         help="Section C only: number of OAS heavy sequences to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Section C only.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    work_dir = cfg["paths"]["work_dir"]
    os.makedirs(os.path.join(work_dir, "tables"), exist_ok=True)

    sections_to_run = [args.only] if args.only else ["A", "B", "C"]

    results = {}
    if "A" in sections_to_run:
        results["A"] = run_section_a(work_dir, cfg)
    if "B" in sections_to_run:
        results["B"] = run_section_b(work_dir, args.master_csv)
    if "C" in sections_to_run:
        results["C"] = run_section_c(work_dir, cfg, args.oas_sample_n, args.seed, args.master_csv)

    if "B" in results or "C" in results:
        # Loads any existing rq2_extended_diversity.json and merges into
        # it, rather than starting from an empty dict, so that running
        # Section B and Section C as two separate invocations (--only B,
        # then later --only C) does not have one section's output
        # overwrite the other's.
        out_path = os.path.join(work_dir, "tables", "rq2_extended_diversity.json")
        extended = {}
        if os.path.exists(out_path):
            with open(out_path) as f:
                extended = json.load(f)
            log(f"Loaded existing {out_path} to merge into (preserves any "
                f"section not recomputed this run).")
        if "B" in results:
            extended["sabdab_clonotype_proxy"] = results["B"]
        if "C" in results:
            extended["oas_vs_sabdab_h3_aa_composition"] = results["C"]
        with open(out_path, "w") as f:
            json.dump(extended, f, indent=2, default=str)
        log(f"Wrote {out_path}")

    log("RQ2 analysis complete.")


if __name__ == "__main__":
    main()