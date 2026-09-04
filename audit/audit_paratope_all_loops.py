#!/usr/bin/env python3
"""
audit_paratope_all_loops.py
=========================================
Extends audit_paratope_h3.py (H3-only) to all six CDR
loops (H1, H2, H3, L1, L2, L3), matching Table tab:paratope_all_loops.

IMPORTANT: this reuses the SAME per-loop block-selection convention as
isolate_cdr_loops() in rq1_sequence_structural_bias.py (Section E) -- the
function that produced the CURRENTLY PUBLISHED six-loop table -- not the
slightly different "3rd-block-else-last-block" heuristic used in the
H3-only script. Specifically: if a chain's CDR mask has more than 3
contiguous blocks, that chain's loops are skipped entirely for that entry
(matching isolate_cdr_loops' `if len(blocks) > 3: return None`), and loop
labels are assigned from the END of the label list
(labels[-len(blocks):]) so that e.g. 2 detected blocks on the heavy chain
are assumed to be H2+H3 (missing H1), not H1+H2. This choice is
deliberate: it keeps this analysis's loop identity consistent with what's
already published in Table tab:paratope_all_loops, rather than
introducing a second, silently different definition of "L2" alongside the
existing one. In the overwhelming majority of entries (exactly 3 detected
blocks per chain), this agrees exactly with the H3-only script's rule, so
already-delivered H3 numbers are not expected to be materially affected.

Antigen-side computation (antigen chain resolution, KD-tree over antigen
atoms, per-residue minimum any-atom distance) is identical to and reused
from the H3-only script's design -- built ONCE per entry and reused across
all six loops' residues, since the antigen doesn't change per loop.

Usage
-----
    python scripts/audit_paratope_all_loops.py --config configs/config.yaml \
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
    get_fv_residues, _residue_aa, IMGT_CDR_H, IMGT_CDR_L, ANTIGEN_EXCLUDED_RESNAMES,
)

THRESHOLDS = [4.5, 6.0, 8.0, 10.0]
AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")
LOOPS_H = ["H1", "H2", "H3"]
LOOPS_L = ["L1", "L2", "L3"]


def find_contiguous_blocks(mask):
    true_idx = np.where(mask)[0]
    if len(true_idx) == 0:
        return []
    blocks, start, prev = [], true_idx[0], true_idx[0]
    for idx in true_idx[1:]:
        if idx != prev + 1:
            blocks.append((start, prev))
            start = idx
        prev = idx
    blocks.append((start, prev))
    return blocks


def get_chain_loop_residues(chain, is_heavy):
    """Mirrors isolate_cdr_loops()'s block-selection convention exactly
    (see module docstring), but returns residue objects + amino acids per
    loop label, not index ranges into a stored sequence array."""
    residues = get_fv_residues(chain)
    if not residues:
        return {}

    kept, imgt_nums = [], []
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
        return {}
    imgt_nums = np.array(imgt_nums)

    cdr_ranges = IMGT_CDR_H if is_heavy else IMGT_CDR_L
    cdr_mask = np.zeros(len(kept), dtype=bool)
    for (start, end) in cdr_ranges.values():
        cdr_mask |= (imgt_nums >= start) & (imgt_nums <= end)

    blocks = find_contiguous_blocks(cdr_mask)
    if len(blocks) > 3:
        return {}
    labels = LOOPS_H if is_heavy else LOOPS_L
    result = {}
    for lbl, (s, e) in zip(labels[-len(blocks):], blocks):
        result[lbl] = kept[s:e + 1]
    return result


def get_antigen_atoms(model, requested_chains, ab_chain_ids):
    ag_residues = []
    for chain_id in requested_chains:
        if chain_id in ab_chain_ids:
            continue
        matches = [c for c in model.get_chains() if str(c.id).strip().upper() == chain_id]
        if matches:
            ag_residues.extend([r for r in matches[0].get_residues()
                                 if r.get_resname().strip() not in ANTIGEN_EXCLUDED_RESNAMES])
    return [a for r in ag_residues for a in r.get_atoms()]


def process_one_entry(struct_dir, row, cluster_lookup):
    """Returns {loop_label: [{"aa":..,"min_dist":..,"cluster_rep":..}, ...]}"""
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

    loops = {}
    loops.update(get_chain_loop_residues(model[h_chain_id], is_heavy=True))

    has_light = pd.notna(row.get("l_chain")) and str(row["l_chain"]).strip().upper() != "NA"
    l_chain_id = str(row["l_chain"]).strip().upper() if has_light else None
    if has_light and l_chain_id in model:
        loops.update(get_chain_loop_residues(model[l_chain_id], is_heavy=False))

    if not loops:
        return None

    ab_chain_ids = {h_chain_id}
    if l_chain_id:
        ab_chain_ids.add(l_chain_id)
    requested = {c.strip().upper() for c in str(row["antigen_chain_raw"]).split("|") if c.strip()}
    ag_atoms = get_antigen_atoms(model, requested, ab_chain_ids)
    if not ag_atoms:
        return None
    ag_coords = np.array([a.get_coord() for a in ag_atoms])
    tree = cKDTree(ag_coords)

    cluster_rep = cluster_lookup.get(row["filename_stem"])
    out = defaultdict(list)
    for lbl, res_aa_list in loops.items():
        for res, aa in res_aa_list:
            if aa not in AA_ALPHABET:
                continue
            res_coords = np.array([a.get_coord() for a in res.get_atoms()])
            if len(res_coords) == 0:
                continue
            dists, _ = tree.query(res_coords, k=1)
            out[lbl].append({"aa": aa, "min_dist": float(dists.min()),
                              "cluster_rep": cluster_rep, "pdb_id": pdb_id})
    return dict(out)


def composition_by_threshold(records, threshold):
    contacting, noncontacting = Counter(), Counter()
    for r in records:
        (contacting if r["min_dist"] <= threshold else noncontacting)[r["aa"]] += 1
    n_c, n_nc = sum(contacting.values()), sum(noncontacting.values())
    result = {}
    for aa in sorted(AA_ALPHABET):
        f_c = contacting[aa] / n_c if n_c else 0.0
        f_nc = noncontacting[aa] / n_nc if n_nc else 0.0
        result[aa] = {"contacting_count": contacting[aa], "noncontacting_count": noncontacting[aa],
                       "contacting_freq": f_c, "noncontacting_freq": f_nc,
                       "fold_change": (f_c / f_nc) if f_nc > 0 else None}
    return result, n_c, n_nc


def cluster_bootstrap_ci(records_by_cluster, cluster_reps, threshold, n_boot, seed):
    rng = np.random.default_rng(seed)
    fold_changes = defaultdict(list)
    n_clusters = len(cluster_reps)
    if n_clusters == 0:
        return {}
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
    parser.add_argument("--n_sample", type=int, default=2000)
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

    all_loops = {lbl: [] for lbl in LOOPS_H + LOOPS_L}
    n_failures = 0
    n_zero_loop_entries = {lbl: 0 for lbl in LOOPS_H + LOOPS_L}
    n_entries_with_loop = {lbl: 0 for lbl in LOOPS_H + LOOPS_L}

    for i, (_, row) in enumerate(sample.iterrows()):
        loops = process_one_entry(args.struct_dir, row, cluster_lookup)
        if not loops:
            n_failures += 1
            continue
        for lbl in LOOPS_H + LOOPS_L:
            if lbl in loops and loops[lbl]:
                all_loops[lbl].extend(loops[lbl])
                n_entries_with_loop[lbl] += 1
            else:
                n_zero_loop_entries[lbl] += 1
        if (i + 1) % 200 == 0:
            log(f"  ...{i + 1}/{len(sample)} entries processed")

    log(f"Done: {len(sample) - n_failures}/{len(sample)} entries successfully resolved")
    for lbl in LOOPS_H + LOOPS_L:
        log(f"  {lbl}: {len(all_loops[lbl])} residues from {n_entries_with_loop[lbl]} entries "
            f"({n_zero_loop_entries[lbl]} entries with no {lbl} detected)")

    report = {"n_entries_sampled": len(sample), "n_entries_failed": n_failures, "loops": {}}

    for lbl in LOOPS_H + LOOPS_L:
        records = all_loops[lbl]
        records_by_cluster = defaultdict(list)
        for r in records:
            if r["cluster_rep"]:
                records_by_cluster[r["cluster_rep"]].append(r)
        cluster_reps = list(records_by_cluster.keys())

        loop_report = {
            "n_residues": len(records), "n_entries_with_loop": n_entries_with_loop[lbl],
            "n_entries_without_loop": n_zero_loop_entries[lbl], "n_clusters": len(cluster_reps),
            "by_threshold": {},
        }
        for T in THRESHOLDS:
            comp, n_c, n_nc = composition_by_threshold(records, T)
            frac = n_c / (n_c + n_nc) if (n_c + n_nc) else None
            ci = cluster_bootstrap_ci(records_by_cluster, cluster_reps, T, args.n_boot, args.seed)
            loop_report["by_threshold"][str(T)] = {
                "fraction_contacting": frac, "n_contacting": n_c, "n_noncontacting": n_nc,
                "composition": comp, "cluster_bootstrap_ci_fold_change": ci,
            }
        report["loops"][lbl] = loop_report
        frac10 = loop_report["by_threshold"]["10.0"]["fraction_contacting"]
        frac_str = f"{frac10:.1%}" if frac10 is not None else "N/A (0 residues)"
        log(f"{lbl} @ 10A: {frac_str} contacting")

    out_path = os.path.join(work_dir, "tables", "paratope_all_loops_contact_redefinition.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
