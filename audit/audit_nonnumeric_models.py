#!/usr/bin/env python3
"""
audit_nonnumeric_models.py

Three checks:
  (1) Full raw TSV rows for these 6 PDBs (every model value present,
      letter or numeric) -- shows whether the letter models are alongside
      normal numeric models for the same PDB, or the PDB's only entries.
  (2) Raw structure file MODEL record serial numbers, read directly from
      the PDB file (grep, not BioPython, since BioPython's model.id
      already silently coerces/interprets these and would hide exactly
      the thing we're trying to see).
  (3) Compositional comparison: species / method / antigen_type for all
      rows belonging to these 6 PDBs vs. the corpus-wide baseline, so a
      "compositionally concentrated, not just PDB-concentrated" claim (or
      its absence) is actually checked rather than assumed.

Usage
-----
    python scripts/audit_nonnumeric_models.py \
        --config configs/config.yaml \
        --struct_dir <raw_data>/all_structures \
        --pdb_ids 2kh2,2ltq,7ssh,7st3,7stg,7ums
"""

import os
import sys
import csv
import json
import argparse
import subprocess
from pathlib import Path
from collections import Counter

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_config, require_path, detect_delimiter, log


def load_full_tsv(tsv_path: str) -> pd.DataFrame:
    delimiter = detect_delimiter(tsv_path)
    with open(tsv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = list(reader)
    df = pd.DataFrame(rows)
    df.columns = [c.strip() for c in df.columns]
    return df


def check_1_raw_tsv_rows(df: pd.DataFrame, pdb_ids: list) -> dict:
    sub = df[df["pdb"].str.lower().isin(pdb_ids)]
    out = {}
    for pdb in pdb_ids:
        rows = sub[sub["pdb"].str.lower() == pdb]
        model_values = rows["model"].value_counts().to_dict()
        out[pdb] = {
            "n_rows_total": len(rows),
            "model_value_counts": model_values,
            "n_numeric_models": sum(1 for m in rows["model"] if str(m).strip().lstrip("-").isdigit()),
            "n_letter_models": sum(1 for m in rows["model"] if not str(m).strip().lstrip("-").isdigit()),
        }
    return out


def check_2_raw_model_records(struct_dir: str, pdb_ids: list) -> dict:
    out = {}
    for pdb in pdb_ids:
        pdb_file = os.path.join(struct_dir, "imgt", f"{pdb}.pdb")
        if not os.path.isfile(pdb_file):
            out[pdb] = {"status": "FILE_NOT_FOUND", "path": pdb_file}
            continue
        try:
            result = subprocess.run(
                ["grep", "-E", "^MODEL", pdb_file], capture_output=True, text=True
            )
            model_lines = [l.strip() for l in result.stdout.splitlines()]
        except Exception as e:
            out[pdb] = {"status": "GREP_FAILED", "error": str(e)}
            continue
        # Also check for REMARK lines that might explain alternate numbering
        # or multiple biological assemblies (common source of non-sequential
        # model labeling in curated/re-annotated files).
        try:
            remark_result = subprocess.run(
                ["grep", "-iE", "MODEL|ASSEMBLY|ENSEMBLE", pdb_file],
                capture_output=True, text=True,
            )
            context_lines = [l.strip() for l in remark_result.stdout.splitlines()][:15]
        except Exception:
            context_lines = []
        out[pdb] = {
            "status": "OK",
            "n_model_records": len(model_lines),
            "model_record_lines": model_lines[:20],
            "context_lines_sample": context_lines,
        }
    return out

def check_3_compositional_comparison(df: pd.DataFrame, pdb_ids: list) -> dict:
    df = df.copy()
    df["heavy_species_norm"] = df.get("heavy_species", pd.Series([""] * len(df))).str.strip().str.lower()
    df["method_norm"] = df.get("method", pd.Series([""] * len(df))).str.strip()
    df["antigen_type_norm"] = df.get("antigen_type", pd.Series([""] * len(df))).str.strip().str.lower()

    baseline = {
        "n_rows": len(df),
        "heavy_species_top5": df["heavy_species_norm"].value_counts().head(5).to_dict(),
        "method_top5": df["method_norm"].value_counts().head(5).to_dict(),
        "antigen_type_top5": df["antigen_type_norm"].value_counts().head(5).to_dict(),
    }

    sub = df[df["pdb"].str.lower().isin(pdb_ids)]
    affected = {
        "n_rows": len(sub),
        "heavy_species_counts": sub["heavy_species_norm"].value_counts().to_dict(),
        "method_counts": sub["method_norm"].value_counts().to_dict(),
        "antigen_type_counts": sub["antigen_type_norm"].value_counts().to_dict(),
        "organism_counts": sub.get("organism", pd.Series(dtype=str)).value_counts().to_dict(),
        "date_range": [sub.get("date", pd.Series(dtype=str)).min(), sub.get("date", pd.Series(dtype=str)).max()],
    }

    return {"corpus_baseline": baseline, "affected_6_pdbs": affected,
            "note": "147 rows across 6 PDBs is concentrated at the PDB level "
                    "by construction. This section checks whether it is ALSO "
                    "concentrated by species/method/antigen-type -- i.e. "
                    "whether dropping these rows disproportionately removes "
                    "one kind of entry rather than being incidental."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--struct_dir", required=True,
                         help="Same --struct_dir passed to preprocess_sabdab.py")
    parser.add_argument("--pdb_ids", default="2kh2,2ltq,7ssh,7st3,7stg,7ums",
                         help="Comma-separated PDB ids to investigate")
    args = parser.parse_args()
    cfg = load_config(args.config)
    work_dir = cfg["paths"]["work_dir"]
    tsv_path = cfg["paths"]["sabdab_summary_tsv"]
    pdb_ids = [p.strip().lower() for p in args.pdb_ids.split(",")]

    require_path(tsv_path, "SAbDab summary TSV")
    log(f"Loading full raw TSV from {tsv_path}")
    df = load_full_tsv(tsv_path)
    log(f"Loaded {len(df)} raw rows, {len(df.columns)} columns")

    log("=" * 70)
    log("Check 1: raw TSV rows for the 6 affected PDBs")
    check1 = check_1_raw_tsv_rows(df, pdb_ids)
    for pdb, info in check1.items():
        log(f"  {pdb}: {info['n_rows_total']} total rows, "
            f"{info['n_numeric_models']} numeric-model rows, "
            f"{info['n_letter_models']} letter-model rows, "
            f"model values: {info['model_value_counts']}")

    log("=" * 70)
    log(f"Check 2: raw MODEL records in structure files (struct_dir={args.struct_dir})")
    check2 = check_2_raw_model_records(args.struct_dir, pdb_ids)
    for pdb, info in check2.items():
        if info.get("status") == "OK":
            log(f"  {pdb}: {info['n_model_records']} MODEL records in file: "
                f"{info['model_record_lines'][:5]}")
        else:
            log(f"  {pdb}: {info.get('status')} -- {info}")

    log("=" * 70)
    log("Check 3: compositional comparison (affected 6 PDBs vs. corpus baseline)")
    check3 = check_3_compositional_comparison(df, pdb_ids)
    log(f"  Affected rows: {check3['affected_6_pdbs']['n_rows']}")
    log(f"  Affected species: {check3['affected_6_pdbs']['heavy_species_counts']}")
    log(f"  Affected method: {check3['affected_6_pdbs']['method_counts']}")
    log(f"  Corpus baseline species (top5): {check3['corpus_baseline']['heavy_species_top5']}")
    log(f"  Corpus baseline method (top5): {check3['corpus_baseline']['method_top5']}")

    report = {"check_1_raw_tsv_rows": check1,
              "check_2_raw_model_records": check2,
              "check_3_compositional_comparison": check3}
    out_path = os.path.join(work_dir, "tables", "nonnumeric_model_investigation.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
