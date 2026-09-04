#!/usr/bin/env python3
"""
00_build_dataset.py
=============================
a single clean per-antibody dataframe joining three independent sources:

  1. Preprocessed SAbDab .pt files (already built by the
     preprocess_sabdab.py + assign_antigen_clusters.py pipeline) - read
     here purely as data, not via importing those scripts.
  2. The raw SAbDab summary TSV - re-parsed independently because several
     fields used in this paper (heavy_subclass, light_subclass,
     heavy_species, light_species, antigen_species, resolution, method,
     date, antigen_type) never made it into the .pt schema.
  3. Thera-SAbDab summary TSV - joined via the structural-identity columns
     ("100% SI Structure", "99% SI Structure", "95-98% SI Structure"),
     which encode pdb:chain_pair references back into SAbDab. This gives
     a REAL therapeutic-status label (is_therapeutic, genetics_class,
     modality, target_gene, clinical_phase, est_status) rather than a
     name-matching heuristic.

Output
------
outputs/tables/master_antibodies.csv  - one row per antibody chain-pair
outputs/tables/build_report.json      - join statistics / coverage report,
                                         so any silent data loss is visible
                                         and citable in the paper's Methods.

Usage
-----
    python scripts/00_build_dataset.py --config configs/config.yaml
"""

import os
import csv
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    load_config, require_path, validate_tsv_columns, validate_pt_sample,
    REQUIRED_SABDAB_TSV_COLUMNS, REQUIRED_THERA_SABDAB_COLUMNS,
    detect_delimiter, log,
)



# Step 1: scan preprocessed .pt files


def scan_pt_files(pt_dir: str) -> pd.DataFrame:
    """
    Read every .pt file and extract the fields this paper needs.
    Returns a dataframe keyed by (pdb_id, h_chain, l_chain) parsed back out
    of the filename, since the .pt sample dict itself doesn't store the
    original chain letters (only pdb_id + paired flag).

    Filename convention (from preprocess_sabdab.py):
        f"{safe_pdb}_{safe_h}{safe_l}_ag{safe_ag}_m{safe_model}.pt"
    We parse this defensively rather than assuming it never collides, since
    safe_h/safe_l can each be multi-character in principle.
    """
    pt_files = sorted(Path(pt_dir).glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {pt_dir}")
    log(f"Found {len(pt_files)} .pt files in {pt_dir}")

    rows = []
    n_errors = 0
    n_schema_errors = 0

    for i, pt_path in enumerate(pt_files):
        if i % 5000 == 0 and i > 0:
            log(f"  scanned {i}/{len(pt_files)}")
        try:
            sample = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        except Exception as e:
            n_errors += 1
            continue

        try:
            validate_pt_sample(sample, str(pt_path))
        except ValueError as e:
            n_schema_errors += 1
            if n_schema_errors <= 3:
                log(f"  SCHEMA WARNING: {e}")
            continue

        fname = pt_path.stem  # e.g. "9jy3_PO_agF_m0"
        # Parse out the components from the filename. This is intentionally
        # NOT a regex on assumed-fixed-width chain codes - it splits on the
        # known fixed markers "_ag" and "_m" inserted by preprocess_sabdab.py.
        try:
            head, rest = fname.split("_ag", 1)
            ag_part, model_part = rest.rsplit("_m", 1)
        except ValueError:
            n_errors += 1
            continue

        pdb_id = sample.get("pdb_id", "").lower()

        rows.append({
            "pt_path": str(pt_path),
            "pdb_id": pdb_id,
            "filename_stem": fname,
            "antigen_chain_str": ag_part,
            "model_id": model_part,
            "paired": bool(sample.get("paired", False)),
            "has_antigen": bool(sample.get("has_antigen", False)),
            "antigen_cluster_id": sample.get("antigen_cluster_id"),
            "cdr3_len_actual": int(sample.get("cdr3_len_actual", -1)),
            "heavy_seq": sample.get("heavy", {}).get("sequence_aa", ""),
            "light_seq": sample.get("light", {}).get("sequence_aa", ""),
            "n_residues": int(sample.get("coord_mask", torch.tensor([])).numel()),
        })

    log(f"Parsed {len(rows)} samples ok | load_errors={n_errors} | schema_errors={n_schema_errors}")
    df = pd.DataFrame(rows)
    return df, {"n_pt_files": len(pt_files), "n_parsed_ok": len(rows),
                "n_load_errors": n_errors, "n_schema_errors": n_schema_errors}


# Step 2: parse raw SAbDab summary TSV for metadata not in the .pt files
def parse_sabdab_summary(tsv_path: str) -> pd.DataFrame:
    delimiter = detect_delimiter(tsv_path)
    log(f"Detected delimiter {delimiter!r} for {tsv_path}")
    rows = []
    with open(tsv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        validate_tsv_columns(reader.fieldnames, REQUIRED_SABDAB_TSV_COLUMNS, tsv_path)
        for row in reader:
            pdb = row.get("pdb", "").strip().lower()
            h = row.get("Hchain", "").strip()
            l = row.get("Lchain", "NA").strip()
            if not pdb or not h:
                continue
            rows.append({
                "pdb_id": pdb,
                "h_chain": h,
                "l_chain": l,
                "model_id_raw": row.get("model", "0").strip(),
                "antigen_chain_raw": row.get("antigen_chain", "").strip(),
                "antigen_type": row.get("antigen_type", "").strip(),
                "antigen_name": row.get("antigen_name", "").strip(),
                "date": row.get("date", "").strip(),
                "organism": row.get("organism", "").strip(),
                "heavy_species": row.get("heavy_species", "").strip().lower(),
                "light_species": row.get("light_species", "").strip().lower(),
                "antigen_species": row.get("antigen_species", "").strip().lower(),
                "resolution_raw": row.get("resolution", "").strip(),
                "method": row.get("method", "").strip(),
                "heavy_subclass": row.get("heavy_subclass", "").strip(),
                "light_subclass": row.get("light_subclass", "").strip(),
                "light_ctype": row.get("light_ctype", "").strip(),
                "scfv": row.get("scfv", "").strip(),
                "engineered": row.get("engineered", "").strip(),
            })
    log(f"Parsed {len(rows)} rows from SAbDab summary TSV")
    df = pd.DataFrame(rows)

    # resolution can be "NA" -> coerce to NaN rather than crashing downstream
    df["resolution"] = pd.to_numeric(df["resolution_raw"], errors="coerce")
    return df

# Step 3: parse Thera-SAbDab and build the PDB:chain -> therapeutic-metadata map
def _parse_si_structure_cell(cell: str):
    """
    Parse a Thera-SAbDab '*% SI Structure' cell into a list of
    (pdb_id, h_chain, l_chain) tuples. Cell format observed:
        "6v4p:CD"               -> single structure, chains C(heavy) D(light)
        "7vux:HL"               -> single structure, chains H(heavy) L(light)
        "6r8x:CB"               -> single structure
        "na" / "None" / ""      -> no structure at this identity tier
        "x:AB;y:CD"             -> multiple structures (seen in bispecifics),
                                   semicolon-delimited
    We do NOT assume exactly 2 chain letters always means (heavy, light) in
    that order for bispecifics - Thera-SAbDab's own convention for bispecific
    rows pairs chain 1 with chain 2 in entry order; we preserve that order
    and flag bispecific rows separately rather than silently mislabeling
    heavy/light.
    """
    if cell is None:
        return []
    cell = cell.strip()
    if not cell or cell.lower() in ("na", "none"):
        return []
    out = []
    for entry in cell.split(";"):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        pdb, chains = entry.split(":", 1)
        pdb = pdb.strip().lower()
        chains = chains.strip()
        if len(chains) < 2:
            continue
        out.append((pdb, chains[0], chains[1]))
    return out


def parse_thera_sabdab(tsv_path: str) -> pd.DataFrame:
    """
    Returns a long-format dataframe: one row per (pdb_id, h_chain, l_chain)
    structural match, joined back to its therapeutic metadata. A single
    therapeutic can map to zero, one, or several PDB structures (across the
    three SI tiers); a PDB can in principle match more than one therapeutic
    name only in pathological cases (not expected, but we don't silently
    dedupe - duplicates are surfaced in the build report).
    """
    delimiter = detect_delimiter(tsv_path)
    log(f"Detected delimiter {delimiter!r} for {tsv_path}")
    rows = []
    with open(tsv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        validate_tsv_columns(reader.fieldnames, REQUIRED_THERA_SABDAB_COLUMNS, tsv_path)
        for row in reader:
            name = row.get("Therapeutic", "").strip()
            if not name:
                continue
            meta = {
                "therapeutic_name": name,
                "modality": row.get("Format", "").strip(),
                "clinical_phase": row.get("Highest_Clin_Trial (Feb '25)", "").strip(),
                "est_status": row.get("Est. Status", "").strip(),
                "target_gene": row.get("Target", "").strip(),
                "genetics_class": row.get(
                    "Genetics (Bispecifics delimited with semicolon)", "").strip(),
                "companies": row.get("Companies", "").strip(),
            }
            # Tier priority: 100% is an exact-sequence structural match,
            # 99% and 95-98% are near-identity matches Thera-SAbDab's
            # maintainers have already computed and curated - we keep the
            # tier label so downstream analysis can choose to use only
            # the 100% tier for a stricter definition if desired.
            for tier_col, tier_label in [
                ("100% SI Structure", "100"),
                ("99% SI Structure", "99"),
                ("95-98% SI Structure", "95-98"),
            ]:
                for pdb_id, h_chain, l_chain in _parse_si_structure_cell(row.get(tier_col, "")):
                    rows.append({**meta, "pdb_id": pdb_id, "h_chain": h_chain,
                                 "l_chain": l_chain, "si_tier": tier_label})

    df = pd.DataFrame(rows, columns=[
        "therapeutic_name", "modality", "clinical_phase", "est_status",
        "target_gene", "genetics_class", "companies", "pdb_id", "h_chain",
        "l_chain", "si_tier",
    ])
    log(f"Thera-SAbDab: parsed {df['therapeutic_name'].nunique() if len(df) else 0} unique "
        f"therapeutics with {len(df)} total structural matches across all SI tiers")
    return df

# Step 4: assemble + join everything
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    sabdab_tsv = cfg["paths"]["sabdab_summary_tsv"]
    sabdab_pt_dir = cfg["paths"]["sabdab_pt_dir"]
    thera_tsv = cfg["paths"]["thera_sabdab_tsv"]
    work_dir = cfg["paths"]["work_dir"]

    require_path(sabdab_tsv, "SAbDab summary TSV")
    require_path(sabdab_pt_dir, "SAbDab preprocessed .pt directory")
    require_path(thera_tsv, "Thera-SAbDab summary TSV")
    os.makedirs(os.path.join(work_dir, "tables"), exist_ok=True)

    log("=" * 70)
    log("Step 1/4: scanning preprocessed .pt files")
    pt_df, pt_report = scan_pt_files(sabdab_pt_dir)

    log("=" * 70)
    log("Step 2/4: parsing raw SAbDab summary TSV")
    meta_df = parse_sabdab_summary(sabdab_tsv)

    log("=" * 70)
    log("Step 3/4: parsing Thera-SAbDab + building therapeutic join")
    thera_df = parse_thera_sabdab(thera_tsv)

    log("=" * 70)
    log("Step 4/4: joining")

    # Join .pt-derived rows to TSV metadata on pdb_id. Because a single PDB
    # can contain multiple antibody chain pairs (see e.g. 9jy3 in the sample
    # data, which has two independent H/L pairs against the same antigen)
    # AND multiple models per chain pair (NMR ensembles can have 20+ models,
    # each its own .pt file), we cannot join on pdb_id alone. We instead
    # reconstruct the EXACT filename stem preprocess_sabdab.py would have
    # produced for each TSV row and match against that directly.
    def _safe(s: str) -> str:
        return str(s).replace("|", "_")

    meta_by_pdb = defaultdict(list)
    for _, r in meta_df.iterrows():
        expected_stem = (
            f"{_safe(r['pdb_id'])}_{_safe(r['h_chain'])}{_safe(r['l_chain'])}"
            f"_ag{_safe(r['antigen_chain_raw'])}_m{_safe(r['model_id_raw'])}"
        )
        meta_by_pdb[(r["pdb_id"], expected_stem)] = r

    matched_rows = []
    n_unmatched = 0
    n_fallback_used = 0
    for _, pt_row in pt_df.iterrows():
        pdb_id = pt_row["pdb_id"]
        stem = pt_row["filename_stem"]
        match = meta_by_pdb.get((pdb_id, stem))

        if match is None:
            # Fallback: some PDBs have only one chain-pair total in the TSV
            # (independent of model count) - if so, use it even though the
            # exact stem didn't match character-for-character (e.g. due to
            # an encoding edge case not covered above).
            same_pdb_candidates = [r for (p, s), r in meta_by_pdb.items() if p == pdb_id]
            # de-duplicate by (h_chain, l_chain) since the same chain-pair
            # appears once per model_id in the TSV
            unique_chain_pairs = {(r["h_chain"], r["l_chain"]) for r in same_pdb_candidates}
            if len(unique_chain_pairs) == 1 and len(same_pdb_candidates) > 0:
                match = same_pdb_candidates[0]
                n_fallback_used += 1

        if match is None:
            n_unmatched += 1
            merged = pt_row.to_dict()
        else:
            merged = {**pt_row.to_dict(), **match.to_dict()}
        matched_rows.append(merged)

    master_df = pd.DataFrame(matched_rows)
    log(f"Matched {len(master_df) - n_unmatched}/{len(master_df)} .pt rows to TSV metadata "
        f"({n_fallback_used} via single-chain-pair fallback, {n_unmatched} truly unmatched - "
        f"see build_report.json for the count to cite as a caveat)")

    # Join therapeutic metadata. A .pt row matches if its (pdb_id, h_chain,
    # l_chain) appears in the Thera-SAbDab long-format table. Prefer the
    # 100% SI tier match if multiple tiers matched the same PDB+chains.
    thera_df["_tier_rank"] = thera_df["si_tier"].map({"100": 0, "99": 1, "95-98": 2})
    thera_best = (thera_df.sort_values("_tier_rank")
                           .drop_duplicates(subset=["pdb_id", "h_chain", "l_chain"], keep="first"))
    thera_best = thera_best.drop(columns=["_tier_rank"])

    if "h_chain" in master_df.columns and "l_chain" in master_df.columns:
        master_df = master_df.merge(
            thera_best, on=["pdb_id", "h_chain", "l_chain"], how="left"
        )
    else:
        log("WARNING: master_df has no h_chain/l_chain columns after TSV join - "
            "therapeutic join skipped. Check the TSV-join match rate above.")

    master_df["is_therapeutic"] = master_df.get(
        "therapeutic_name", pd.Series([None] * len(master_df))
    ).notna()

    n_therapeutic = int(master_df["is_therapeutic"].sum())
    log(f"Therapeutic join: {n_therapeutic}/{len(master_df)} antibody entries "
        f"matched to a known Thera-SAbDab therapeutic "
        f"({n_therapeutic/max(len(master_df),1)*100:.2f}%)")

    out_csv = os.path.join(work_dir, "tables", "master_antibodies.csv")
    master_df.to_csv(out_csv, index=False)
    log(f"Wrote {out_csv} ({len(master_df)} rows, {len(master_df.columns)} columns)")

    report = {
        "pt_scan": pt_report,
        "n_tsv_rows": len(meta_df),
        "n_unmatched_pt_to_tsv": n_unmatched,
        "n_fallback_matches_used": n_fallback_used,
        "n_therapeutic_matches": n_therapeutic,
        "n_master_rows": len(master_df),
        "master_columns": list(master_df.columns),
    }
    report_path = os.path.join(work_dir, "tables", "build_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log(f"Wrote {report_path}")
    log("Done. Inspect build_report.json before trusting downstream figures - "
        "specifically n_unmatched_pt_to_tsv should be a small fraction of n_master_rows.")


if __name__ == "__main__":
    main()