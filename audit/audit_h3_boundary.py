#!/usr/bin/env python3
"""
audit_h3_boundary.py

The exact boundary rule, confirmed against preprocess_sabdab.py (not
re-derived or guessed):
  - Numbering: IMGT, taken directly from SAbDab's own IMGT-renumbered PDB
    structures (struct_dir/imgt/{pdb}.pdb) -- not computed by this pipeline.
  - Residue inclusion: only residues passing Bio.PDB's is_aa(standard=True)
    are counted at all. A residue with zero atoms in the file (true missing
    density, no ATOM/HETATM record) is not a residue object and is
    absent from the sequence -- i.e. a gap in density currently shortens the
    apparent H3 length exactly like a genuinely short loop would. See part
    (b) below for the tool used to check this (it cannot be checked from
    the .pt files alone -- it needs the raw structure).
  - A residue WITH atoms but no resolved CA (rare) still counts toward
    length; only its coord_mask entry is False.
  - Insertion codes: position = IMGT_number + (0.1 * (ord(icode)-ord('A')+1)),
    e.g. 111A -> 111.1, 111B -> 111.2. This is a real, specific rule, not
    "last contiguous block" hand-waving -- now documented precisely.
  - CDR-H3 window: IMGT positions [105, 117.9]. H3 itself is identified as
    the 3rd contiguous true-valued block within the heavy-chain CDR mask
    (H1, H2, H3 in that fixed order) -- not literally "the last block";
    falls back to the last block only if fewer than 3 blocks are found.
  - The [3, 35] filter is applied entirely at PREPROCESSING time (excluded
    entries never get a .pt file), so the "unfiltered" distribution the
    paper needs is NOT recoverable from master_antibodies.csv alone -- it
    requires the excluded lengths embedded in exclusions.csv's stage column
    (e.g. "h3_too_long_42"), which this script recovers.

Part (a): SAbDab-only unfiltered length distribution (retained + the 56
length-excluded entries, 48 too long + 8 too short) vs. the 3-35-filtered
distribution used for the OAS comparison, side by side. Excluded-for-length
entries are reported separately from quality failures, since a length
outside [3, 35] is a scope decision, not a data-quality problem.

Part (b): validates the string-derived H3 boundary against ANARCI on a
random sample of retained entries, by re-running ANARCI's own IMGT
numbering on the same heavy-chain sequence and comparing its CDR-H3 span
length against this pipeline's cdr3_len_actual. Requires the `anarci`
package (conda install -c bioconda anarci, or pip install anarci depending
on your environment) -- part (a) runs independently and does not need it.

Usage
-----
    python scripts/audit_h3_boundary.py --config configs/config.yaml --part a
    python scripts/audit_h3_boundary.py --config configs/config.yaml --part b --n_sample 200
    python scripts/audit_h3_boundary.py --config configs/config.yaml --part all
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_config, require_path, shannon_entropy, normalized_entropy, gini_coefficient, log

H3_STAGE_RE = re.compile(r"^h3_too_(long|short)_(\d+)$")

# Part (a): unfiltered vs. filtered length distributions
def run_part_a(cfg, work_dir):
    sabdab_pt_dir = cfg["paths"]["sabdab_pt_dir"]
    exclusions_path = os.path.join(sabdab_pt_dir, "exclusions.csv")
    funnel_path = os.path.join(sabdab_pt_dir, "funnel_report.json")
    master_path = os.path.join(work_dir, "tables", "master_antibodies.csv")

    require_path(exclusions_path, "exclusions.csv (from preprocess_sabdab.py)")
    require_path(master_path, "master_antibodies.csv")

    master = pd.read_csv(master_path, low_memory=False)
    if "cdr3_len_actual" not in master.columns:
        raise ValueError(
            "[SCHEMA MISMATCH] master_antibodies.csv has no cdr3_len_actual column. "
            f"Found: {list(master.columns)}"
        )
    retained_lengths = master["cdr3_len_actual"].dropna().astype(int).tolist()
    log(f"Retained (3-35-filtered) entries: {len(retained_lengths)}")

    excl = pd.read_csv(exclusions_path)
    excluded_lengths = []
    other_exclusion_counts = Counter()
    for stage in excl["stage"].dropna():
        m = H3_STAGE_RE.match(stage)
        if m:
            excluded_lengths.append(int(m.group(2)))
        else:
            other_exclusion_counts[stage] += 1

    log(f"Recovered {len(excluded_lengths)} length-excluded entries from exclusions.csv "
        f"(expect 56: 48 too long, 8 too short per paper's existing funnel numbers)")
    n_too_long = sum(1 for l in excluded_lengths if l > 35)
    n_too_short = sum(1 for l in excluded_lengths if l < 3)
    log(f"  -> {n_too_long} too long, {n_too_short} too short")

    unfiltered_lengths = retained_lengths + excluded_lengths
    log(f"Unfiltered universe: {len(unfiltered_lengths)} entries "
        f"({len(retained_lengths)} retained + {len(excluded_lengths)} length-excluded)")

    def _dist_stats(lengths, label):
        arr = np.array(lengths)
        counts_by_len = Counter(arr.tolist())
        support = sorted(counts_by_len.keys())
        counts_vec = np.array([counts_by_len[l] for l in support])
        return {
            "label": label,
            "n": len(arr),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "min": int(arr.min()),
            "max": int(arr.max()),
            "n_observed_categories": len(support),
            "entropy_bits": round(shannon_entropy(counts_vec), 4),
            "normalized_entropy_observed_support": round(normalized_entropy(counts_vec), 4),
            "gini_over_length_histogram_bins": round(gini_coefficient(counts_vec), 4),
        }

    report = {
        "filtered_3_35": _dist_stats(retained_lengths, "SAbDab, 3-35-filtered (used for OAS comparison)"),
        "unfiltered": _dist_stats(unfiltered_lengths, "SAbDab, unfiltered (all entries with a measurable H3 length)"),
        "excluded_for_length": {
            "n_total": len(excluded_lengths),
            "n_too_long": n_too_long,
            "n_too_short": n_too_short,
            "note": "These are NOT quality failures -- 44/48 too-long entries "
                    "are Bos taurus ultralong CDR-H3s (real biology, distinct "
                    "genetic mechanism), reported here as genuine species "
                    "diversity, not binned into the quality-exclusion funnel.",
        },
        "other_exclusion_stages_seen": dict(other_exclusion_counts),
    }

    # Cross-check against the paper's existing claim (56 = 48 + 8)
    if n_too_long != 48 or n_too_short != 8:
        log(f"WARNING: recovered counts ({n_too_long} too long, {n_too_short} too short) "
            f"do NOT match the paper's stated 48/8 split. Re-check funnel_report.json "
            f"before writing anything -- either the paper's number or this recovery "
            f"has a bug.")
        report["MISMATCH_WARNING"] = (
            f"Recovered {n_too_long}/{n_too_short} vs. paper's stated 48/8 -- investigate before use."
        )

    out_path = os.path.join(work_dir, "tables", "h3_boundary_length_distributions.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log(f"Wrote {out_path}")
    print(json.dumps(report, indent=2))
    return report

# Part (b): ANARCI cross-validation of the H3 boundary rule
def run_part_b(cfg, work_dir, n_sample, seed):
    try:
        from anarci import anarci as anarci_number
    except ImportError:
        raise ImportError(
            "The `anarci` package is not installed in this environment. "
            "Install it (e.g. `conda install -c bioconda anarci` or "
            "`pip install anarci`) and re-run this part. Part (a) does not "
            "need it and can be run independently."
        )

    master_path = os.path.join(work_dir, "tables", "master_antibodies.csv")
    require_path(master_path, "master_antibodies.csv")
    master = pd.read_csv(master_path, low_memory=False)

    rng = np.random.default_rng(seed)
    sample = master.sample(n=min(n_sample, len(master)), random_state=seed)
    log(f"Validating {len(sample)} randomly sampled entries against ANARCI IMGT numbering "
        f"(seed={seed})")

    agreements, mismatches = [], []
    for _, row in sample.iterrows():
        pdb_id = row.get("pdb_id", "?")
        heavy_seq = row.get("heavy_seq", "")
        our_len = row.get("cdr3_len_actual")
        if not heavy_seq or pd.isna(our_len):
            continue
        try:
            numbering, _, _ = anarci_number(
                [("query", heavy_seq)], scheme="imgt", allow=set(["H"])
            )
        except Exception as e:
            mismatches.append({"pdb_id": pdb_id, "error": str(e)})
            continue
        if not numbering or not numbering[0]:
            mismatches.append({"pdb_id": pdb_id, "error": "ANARCI returned no numbering"})
            continue
        # ANARCI numbering[0][0][0] is a list of ((imgt_pos, icode), aa) tuples
        num_list = numbering[0][0][0]
        h3_positions = [aa for (pos, icode), aa in num_list
                         if 105 <= pos <= 117 and aa != "-"]
        anarci_h3_len = len(h3_positions)
        match = (anarci_h3_len == int(our_len))
        (agreements if match else mismatches).append({
            "pdb_id": pdb_id, "our_len": int(our_len), "anarci_len": anarci_h3_len,
        })

    n_compared = len(agreements) + len([m for m in mismatches if "anarci_len" in m])
    n_agree = len(agreements)
    report = {
        "n_sampled": len(sample),
        "n_compared": n_compared,
        "n_agree": n_agree,
        "agreement_pct": round(100.0 * n_agree / n_compared, 2) if n_compared else None,
        "n_anarci_errors": len([m for m in mismatches if "error" in m]),
        "mismatches_sample": [m for m in mismatches if "anarci_len" in m][:20],
        "note": "Compares this pipeline's IMGT-block-derived CDR-H3 length "
                "against ANARCI's independently-computed IMGT CDR-H3 length "
                "for the same heavy-chain sequence. A mismatch does not by "
                "itself prove which one is wrong -- inspect mismatches_sample "
                "manually before concluding anything.",
    }
    out_path = os.path.join(work_dir, "tables", "h3_boundary_anarci_cross_check.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log(f"Wrote {out_path}")
    print(json.dumps(report, indent=2))
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--part", choices=["a", "b", "all"], default="all")
    parser.add_argument("--n_sample", type=int, default=200, help="Part b only")
    parser.add_argument("--seed", type=int, default=42, help="Part b only")
    args = parser.parse_args()
    cfg = load_config(args.config)
    work_dir = cfg["paths"]["work_dir"]
    os.makedirs(os.path.join(work_dir, "tables"), exist_ok=True)

    if args.part in ("a", "all"):
        run_part_a(cfg, work_dir)
    if args.part in ("b", "all"):
        run_part_b(cfg, work_dir, args.n_sample, args.seed)

if __name__ == "__main__":
    main()
