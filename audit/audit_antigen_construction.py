#!/usr/bin/env python3
"""
audit_antigen_construction.py

Four checks against real master_antibodies.csv / raw structure data:
  (1) How many retained entries have a multi-chain antigen (pipe-delimited
      antigen_chain_raw), and what does the distribution of chain-counts
      look like?
  (2) Among multi-chain antigens, how many involve near-identical or
      identical constituent chains (symmetric assembly signature)?
      IMPORTANT: the .pt schema stores no per-residue chain-id field, so
      antigen_seq (the concatenated string) cannot be split back into its
      constituent chains after the fact. This check instead RE-DERIVES
      each requested antigen chain's own sequence directly from the raw
      structure file, reusing antigen_residue_letter() and
      ANTIGEN_EXCLUDED_RESNAMES imported from the real preprocessing
      module rather than reimplementing the selection logic separately
      (which could silently drift from the actual construction rule).
  (3) How many entries contain a non-standard "[XXX]" bracketed residue
      tag in antigen_seq, and which residue names are most common --
      confirms whether glycans are actually present and how they show up.
  (4) Sanity distribution of antigen_seq length.

Usage
-----
    python scripts/audit_antigen_construction.py --config configs/config.yaml \
        --struct_dir <raw_data>/all_structures
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_config, require_path, log

# Reuse the REAL construction logic rather than reimplementing it --
# avoids silently drifting from what actually built antigen_seq.
from preprocess_sabdab import (
    antigen_residue_letter, ANTIGEN_EXCLUDED_RESNAMES,
)
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings("ignore")

BRACKET_RE = re.compile(r"\[([A-Za-z0-9]+)\]")


def sample_antigen_seqs_from_pt(ag_df, n_sample, seed):
    """antigen_seq is not a column in master_antibodies.csv (only h3_seq and
    a few others were ever migrated in) -- it lives only inside the .pt
    files themselves. Rather than running a full-corpus CSV migration for
    a descriptive check, read it directly from a random sample of .pt
    files via torch.load, which is cheap (small files) compared to check
    2's raw-structure re-parsing."""
    if "pt_path" not in ag_df.columns:
        return []
    sample = ag_df.sample(n=min(n_sample, len(ag_df)), random_state=seed)
    seqs = []
    n_load_errors = 0
    for pt_path in sample["pt_path"]:
        if not isinstance(pt_path, str) or not os.path.isfile(pt_path):
            n_load_errors += 1
            continue
        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            seq = data.get("antigen_seq", "")
            if seq:
                seqs.append(seq)
        except Exception:
            n_load_errors += 1
    if n_load_errors:
        log(f"  ({n_load_errors} .pt load errors during antigen_seq sampling, skipped)")
    return seqs


def get_chain_sequences(struct_dir, pdb_id, model_id_raw, requested_chains, ab_chain_ids):
    """Re-derives each requested antigen chain's OWN sequence directly from
    the raw structure file, using the exact same residue-selection and
    translation rule as the real preprocessing script. Returns
    {chain_id: sequence} or None if the file/model can't be loaded."""
    pdb_file = os.path.join(struct_dir, "imgt", f"{pdb_id.lower()}.pdb")
    if not os.path.isfile(pdb_file):
        return None
    try:
        structure = PDBParser(QUIET=True).get_structure(pdb_id, pdb_file)
        models = list(structure.get_models())
        model_idx = int(model_id_raw)
        model = models[min(model_idx, len(models) - 1)]
    except Exception:
        return None

    out = {}
    for chain_id in requested_chains:
        if chain_id in ab_chain_ids:
            continue
        matches = [c for c in model.get_chains() if str(c.id).strip().upper() == chain_id]
        if not matches:
            continue
        chain = matches[0]
        residues = [r for r in chain.get_residues()
                    if r.get_resname().strip() not in ANTIGEN_EXCLUDED_RESNAMES]
        if not residues:
            continue
        out[chain_id] = "".join(antigen_residue_letter(r.get_resname()) for r in residues)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--struct_dir", required=True)
    parser.add_argument("--n_sample", type=int, default=300,
                         help="How many multi-chain entries to re-derive from raw "
                              "structure files for check (2) -- re-parsing PDB files "
                              "is slow, this is a bounded random sample.")
    parser.add_argument("--n_antigen_seq_sample", type=int, default=3000,
                         help="How many .pt files to sample for checks (3)/(4) "
                              "(bracket tags, length stats) -- cheap torch.load, "
                              "can afford a larger sample than check (2).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg = load_config(args.config)
    work_dir = cfg["paths"]["work_dir"]
    master_path = os.path.join(work_dir, "tables", "master_antibodies.csv")
    require_path(master_path, "master_antibodies.csv")

    df = pd.read_csv(master_path, low_memory=False)
    log(f"Loaded {len(df)} rows")

    required = ["antigen_chain_raw", "pdb_id", "h_chain", "l_chain"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[SCHEMA MISMATCH] missing {missing}. Found: {list(df.columns)}")

    model_col = "model_id_raw" if "model_id_raw" in df.columns else "model_id"
    has_ag = df["has_antigen"] == True if "has_antigen" in df.columns else df["antigen_chain_raw"].notna()
    ag_df = df[has_ag].copy()
    log(f"Antigen-bound entries: {len(ag_df)}")

    # --- Check 1: multi-chain prevalence ---
    def n_chains(raw):
        if pd.isna(raw) or str(raw).strip() == "":
            return 0
        return len([c for c in str(raw).split("|") if c.strip()])

    ag_df["_n_ag_chains"] = ag_df["antigen_chain_raw"].apply(n_chains)
    chain_count_dist = ag_df["_n_ag_chains"].value_counts().sort_index().to_dict()
    n_multichain = int((ag_df["_n_ag_chains"] > 1).sum())
    log(f"Chain-count distribution among antigen-bound entries: {chain_count_dist}")
    log(f"Multi-chain antigen entries: {n_multichain} ({100*n_multichain/len(ag_df):.2f}%)")

    # --- Check 3 & 4: bracketed tags and length stats (antigen_seq sampled from .pt files) ---
    log(f"Sampling antigen_seq from up to {args.n_antigen_seq_sample} .pt files "
        f"(not a CSV column -- see docstring)...")
    sampled_seqs = sample_antigen_seqs_from_pt(ag_df, args.n_antigen_seq_sample, args.seed)
    log(f"Successfully read antigen_seq from {len(sampled_seqs)} .pt files")

    bracket_tally = Counter()
    length_stats = {}
    if sampled_seqs:
        for seq in sampled_seqs:
            for tag in BRACKET_RE.findall(str(seq)):
                bracket_tally[tag] += 1
        log(f"Top 20 bracketed non-standard residue tags: {bracket_tally.most_common(20)}")

        GLYCAN_TAGS = {"NAG", "BMA", "MAN", "FUC", "GAL", "SIA", "NDG", "BGC", "FUL", "GLC"}
        n_with_glycan = sum(1 for s in sampled_seqs if any(f"[{t}]" in s for t in GLYCAN_TAGS))
        log(f"Sampled entries with at least one recognizable glycan tag: {n_with_glycan}/{len(sampled_seqs)} "
            f"({100*n_with_glycan/len(sampled_seqs):.2f}%)")

        lengths = np.array([len(s) for s in sampled_seqs])
        length_stats = {
            "n_sampled": len(lengths), "mean": float(lengths.mean()), "median": float(np.median(lengths)),
            "max": int(lengths.max()), "p99": float(np.percentile(lengths, 99)),
            "n_with_glycan_tag": n_with_glycan,
            "pct_with_glycan_tag": round(100 * n_with_glycan / len(sampled_seqs), 2),
        }
        log(f"antigen_seq length stats (sampled): {length_stats}")
    else:
        log("WARNING: could not sample any antigen_seq from .pt files -- checks 3/4 empty.")

    # --- Check 2: identical/near-identical constituent chains, re-derived from raw structures ---
    multichain_df = ag_df[ag_df["_n_ag_chains"] > 1]
    sample = multichain_df.sample(n=min(args.n_sample, len(multichain_df)), random_state=args.seed) \
        if len(multichain_df) else multichain_df

    n_checked = 0
    n_identical_pair = 0
    examples = []
    for _, row in sample.iterrows():
        requested = {c.strip().upper() for c in str(row["antigen_chain_raw"]).split("|") if c.strip()}
        ab_ids = {str(row["h_chain"]).strip().upper()}
        if pd.notna(row.get("l_chain")) and str(row["l_chain"]).strip().upper() != "NA":
            ab_ids.add(str(row["l_chain"]).strip().upper())
        chain_seqs = get_chain_sequences(
            args.struct_dir, row["pdb_id"], row.get(model_col, 0), requested, ab_ids
        )
        if not chain_seqs or len(chain_seqs) < 2:
            continue
        n_checked += 1
        seqs = list(chain_seqs.values())
        found = any(seqs[i] == seqs[j] for i in range(len(seqs)) for j in range(i + 1, len(seqs))
                    if seqs[i])
        if found:
            n_identical_pair += 1
            if len(examples) < 10:
                examples.append({"pdb_id": row["pdb_id"],
                                  "chain_lengths": {k: len(v) for k, v in chain_seqs.items()}})

    log(f"Check 2: of {n_checked} multi-chain entries with successfully re-derived per-chain "
        f"sequences, {n_identical_pair} contain at least one pair of byte-identical "
        f"constituent chains (symmetric-assembly signature).")
    if examples:
        log(f"Examples: {examples}")

    report = {
        "n_antigen_bound_entries": len(ag_df),
        "chain_count_distribution": {str(k): v for k, v in chain_count_dist.items()},
        "n_multichain_entries": n_multichain,
        "pct_multichain": round(100 * n_multichain / len(ag_df), 2) if len(ag_df) else None,
        "n_antigen_seq_sampled": len(sampled_seqs),
        "bracket_tag_tally_top20": dict(bracket_tally.most_common(20)) if bracket_tally else {},
        "antigen_seq_length_stats": length_stats,
        "identical_chain_check": {
            "n_multichain_sampled": len(sample),
            "n_with_successful_rederivation": n_checked,
            "n_with_identical_pair": n_identical_pair,
            "examples": examples,
        },
    }
    out_path = os.path.join(work_dir, "tables", "antigen_construction_audit.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

