#!/usr/bin/env python3
"""
audit_paratope_h3.py
=======================================
Finer-grained re-check of the RQ1 paratope/full-loop contact comparison
(Section 4.2.2 of the paper), addressing two properties of that simpler
computation that this audit treats more carefully:

  (1) 10A is a proximity threshold, not a contact threshold. It is an
      any-atom-to-any-atom distance (compute_interface_mask uses
      Bio.PDB.NeighborSearch over all antigen atoms against all atoms of
      each antibody residue), which at 10A includes a great deal of
      near-but-not-contacting surface. This script recomputes the same
      any-atom distance definition at 4.5/6/8/10A by computing each H3
      residue's true minimum distance to the antigen once (via a KD-tree,
      not a repeated threshold search), then thresholding that one
      continuous value four ways -- so all four numbers come from a
      single, consistent computation, not four separate reruns that could
      drift from each other.

  (2) "Full H3 loop" and "paratope" are not independent samples -- the
      paratope pool is a subset of the full-loop pool (every paratope
      residue is also a full-loop residue), so a two-proportion z-test
      comparing them would violate its own independence assumption before
      any clustering/redundancy issue is even considered. This script
      instead partitions H3 residues into two disjoint, exhaustive sets --
      "contacting" and "non-contacting" -- at each threshold, and compares
      those.

Uncertainty is obtained by resampling CDR-H3 clusters (not residues) with
replacement, respecting the redundancy structure already established
elsewhere in this paper (same clusters as Section 4.2.1/4.2.2), rather
than treating residues as independent draws.

compute_interface_mask's antigen distance calculation uses full atomic
coordinates, while the .pt files only store per-residue C-alpha
coordinates for the antigen (see _ca_coords() in process_entry), so this
requires re-parsing raw structure files -- it cannot be reconstructed
from the .pt files alone. H3-residue identification and antigen-chain
resolution logic is reused from preprocess_sabdab.py (get_fv_residues,
_residue_aa, IMGT_CDR_H, ANTIGEN_EXCLUDED_RESNAMES) rather than
re-derived independently, to stay consistent with the established
H3-boundary and antigen-construction rules.

Usage
-----
    python scripts/audit_paratope_h3.py --config configs/config.yaml \
        --struct_dir <raw_data>/all_structures \
        --n_sample 2000 --n_boot 2000
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_config, require_path, log

from preprocess_sabdab import (
    get_fv_residues, _residue_aa, IMGT_CDR_H, ANTIGEN_EXCLUDED_RESNAMES,
)

THRESHOLDS = [4.5, 6.0, 8.0, 10.0]
AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")


def get_h3_residues_with_aa(chain):
    """Mirrors parse_chain()'s H3-isolation logic exactly (same
    get_fv_residues, same insertion-code fractional numbering, same IMGT_CDR_H
    range, same 3rd-contiguous-block-else-last-block H3 rule from
    process_entry), but returns (residue_object, amino_acid) pairs for the H3
    range specifically, instead of arrays -- residue objects are needed here
    for full-atom distance computation, which parse_chain doesn't expose."""
    residues = get_fv_residues(chain)
    if not residues:
        return []

    kept = []
    imgt_nums = []
    for res in residues:
        aa = _residue_aa(res)
        if aa is None:
            continue
        res_id = res.get_id()
        pos = res_id[1]
        icode = res_id[2].strip()
        if icode:
            pos = pos + (ord(icode.upper()) - ord('A') + 1) * 0.1
        kept.append((res, aa))
        imgt_nums.append(pos)

    if not kept:
        return []
    imgt_nums = np.array(imgt_nums)

    cdr_mask = np.zeros(len(kept), dtype=bool)
    for (start, end) in IMGT_CDR_H.values():
        cdr_mask |= (imgt_nums >= start) & (imgt_nums <= end)

    padded = np.concatenate(([False], cdr_mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    if len(starts) == 0:
        return []
    if len(starts) >= 3:
        h3_start, h3_end = starts[2], ends[2]
    else:
        h3_start, h3_end = starts[-1], ends[-1]

    return kept[h3_start:h3_end + 1]


def get_antigen_atoms(model, requested_chains, ab_chain_ids):
    ag_residues = []
    for chain_id in requested_chains:
        if chain_id in ab_chain_ids:
            continue
        matches = [c for c in model.get_chains() if str(c.id).strip().upper() == chain_id]
        if matches:
            ag_residues.extend([r for r in matches[0].get_residues()
                                 if r.get_resname().strip() not in ANTIGEN_EXCLUDED_RESNAMES])
    atoms = [a for r in ag_residues for a in r.get_atoms()]
    return atoms


def process_one_entry(struct_dir, row, cluster_lookup):
    pdb_id = row["pdb_id"]
    pdb_file = os.path.join(struct_dir, "imgt", f"{pdb_id.lower()}.pdb")
    if not os.path.isfile(pdb_file):
        return None
    try:
        structure = PDBParser(QUIET=True).get_structure(pdb_id, pdb_file)
        models = list(structure.get_models())
        model_idx = int(row.get("model_id_raw", 0))
        model = models[min(model_idx, len(models) - 1)]
    except Exception:
        return None

    h_chain_id = str(row["h_chain"]).strip().upper()
    if h_chain_id not in model:
        return None
    h3_res_aa = get_h3_residues_with_aa(model[h_chain_id])
    if not h3_res_aa:
        return None

    ab_chain_ids = {h_chain_id}
    if pd.notna(row.get("l_chain")) and str(row["l_chain"]).strip().upper() != "NA":
        ab_chain_ids.add(str(row["l_chain"]).strip().upper())
    requested = {c.strip().upper() for c in str(row["antigen_chain_raw"]).split("|") if c.strip()}
    ag_atoms = get_antigen_atoms(model, requested, ab_chain_ids)
    if not ag_atoms:
        return None

    ag_coords = np.array([a.get_coord() for a in ag_atoms])
    tree = cKDTree(ag_coords)

    cluster_rep = cluster_lookup.get(row["filename_stem"])
    out = []
    for res, aa in h3_res_aa:
        if aa not in AA_ALPHABET:
            continue
        res_coords = np.array([a.get_coord() for a in res.get_atoms()])
        if len(res_coords) == 0:
            continue
        dists, _ = tree.query(res_coords, k=1)
        min_dist = float(dists.min())
        out.append({"aa": aa, "min_dist": min_dist, "cluster_rep": cluster_rep, "pdb_id": pdb_id})
    return out


def composition_by_threshold(records, threshold):
    contacting = Counter()
    noncontacting = Counter()
    for r in records:
        if r["min_dist"] <= threshold:
            contacting[r["aa"]] += 1
        else:
            noncontacting[r["aa"]] += 1
    n_c, n_nc = sum(contacting.values()), sum(noncontacting.values())
    result = {}
    for aa in sorted(AA_ALPHABET):
        f_c = contacting[aa] / n_c if n_c else 0.0
        f_nc = noncontacting[aa] / n_nc if n_nc else 0.0
        result[aa] = {
            "contacting_count": contacting[aa], "noncontacting_count": noncontacting[aa],
            "contacting_freq": f_c, "noncontacting_freq": f_nc,
            "fold_change": (f_c / f_nc) if f_nc > 0 else None,
        }
    return result, n_c, n_nc


def cluster_bootstrap_ci(records_by_cluster, cluster_reps, threshold, n_boot, seed):
    rng = np.random.default_rng(seed)
    fold_changes = defaultdict(list)
    n_clusters = len(cluster_reps)
    for b in range(n_boot):
        drawn = rng.choice(cluster_reps, size=n_clusters, replace=True)
        pooled = []
        for c in drawn:
            pooled.extend(records_by_cluster[c])
        comp, _, _ = composition_by_threshold(pooled, threshold)
        for aa, d in comp.items():
            if d["fold_change"] is not None:
                fold_changes[aa].append(d["fold_change"])
    ci = {}
    for aa, vals in fold_changes.items():
        if vals:
            arr = np.array(vals)
            ci[aa] = {"n_boot_valid": len(arr), "ci_2.5": float(np.percentile(arr, 2.5)),
                       "ci_97.5": float(np.percentile(arr, 97.5)), "median": float(np.median(arr))}
    return ci


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--struct_dir", required=True)
    parser.add_argument("--n_sample", type=int, default=2000,
                         help="Random sample of antigen-bound entries -- re-parsing raw "
                              "structures + KD-tree queries for all 15,736 would be slow; "
                              "bump this up if you want full coverage.")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg = load_config(args.config)
    work_dir = cfg["paths"]["work_dir"]

    master_path = os.path.join(work_dir, "tables", "master_antibodies.csv")
    clusters_path = os.path.join(work_dir, "tables", "rq1_cdrh3_clusters.tsv")
    require_path(master_path, "master_antibodies.csv")
    require_path(clusters_path, "rq1_cdrh3_clusters.tsv")

    master_df = pd.read_csv(master_path, low_memory=False)
    clusters_df = pd.read_csv(clusters_path, sep="\t")
    cluster_lookup = clusters_df.set_index("filename_stem")["cluster_rep"].to_dict()

    ag_df = master_df[master_df["has_antigen"] == True]
    sample = ag_df.sample(n=min(args.n_sample, len(ag_df)), random_state=args.seed)
    log(f"Sampling {len(sample)} of {len(ag_df)} antigen-bound entries")

    all_records = []
    n_failures = 0
    for i, (_, row) in enumerate(sample.iterrows()):
        recs = process_one_entry(args.struct_dir, row, cluster_lookup)
        if recs:
            all_records.extend(recs)
        else:
            n_failures += 1
        if (i + 1) % 200 == 0:
            log(f"  ...{i + 1}/{len(sample)} entries processed "
                f"({len(all_records)} H3 residues collected so far)")

    log(f"Done: {len(all_records)} H3 residues from {len(sample) - n_failures} entries "
        f"({n_failures} entries failed to parse/resolve)")

    records_by_cluster = defaultdict(list)
    for r in all_records:
        if r["cluster_rep"]:
            records_by_cluster[r["cluster_rep"]].append(r)
    cluster_reps = list(records_by_cluster.keys())
    log(f"{len(cluster_reps)} distinct CDR-H3 clusters represented in the sample")

    report = {"n_entries_sampled": len(sample), "n_entries_failed": n_failures,
              "n_h3_residues_total": len(all_records), "n_clusters": len(cluster_reps),
              "by_threshold": {}}

    for T in THRESHOLDS:
        comp, n_c, n_nc = composition_by_threshold(all_records, T)
        frac_contacting = n_c / (n_c + n_nc) if (n_c + n_nc) else None
        log(f"Threshold {T}A: {n_c} contacting / {n_nc} non-contacting "
            f"({100*frac_contacting:.1f}% of residues labeled contacting)")
        ci = cluster_bootstrap_ci(records_by_cluster, cluster_reps, T, args.n_boot, args.seed)
        report["by_threshold"][str(T)] = {
            "fraction_of_h3_residues_contacting": frac_contacting,
            "n_contacting": n_c, "n_noncontacting": n_nc,
            "composition": comp, "cluster_bootstrap_ci_fold_change": ci,
        }

    out_path = os.path.join(work_dir, "tables", "paratope_contact_redefinition.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
