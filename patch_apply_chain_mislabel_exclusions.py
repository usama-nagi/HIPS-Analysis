"""
patch_apply_chain_mislabel_exclusions.py

Self-verifying patch: adds sabdab_chain_mislabel as a post-preprocessing
exclusion stage, writes a clean master table, and appends to exclusions.csv.

ABORTS loudly on any count mismatch BEFORE writing anything.
Run with --dry-run first to confirm state, then without to write.
"""
import argparse, csv, json, os, sys, shutil
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--mislabel-json",  default="tables/chain_mislabel_scope.json")
ap.add_argument("--master-csv",     default="tables/master_antibodies.csv")
ap.add_argument("--exclusions-csv", default="tables/exclusions.csv")
ap.add_argument("--out-master-csv", default="tables/master_antibodies_clean.csv")
args = ap.parse_args()

def abort(msg):
    print(f"[ABORT] {msg}", file=sys.stderr); sys.exit(1)

# 1. Load mislabeled stems
with open(args.mislabel_json) as f:
    scope = json.load(f)
bad_stems = {
    r["filename_stem"]
    for r in scope["per_entry"]
    if r.get("anarci_chain_type") in ("K", "L")
}
EXPECTED_N = 189
if len(bad_stems) != EXPECTED_N:
    abort(f"Expected {EXPECTED_N} mislabeled stems, got {len(bad_stems)}")
print(f"[OK] Loaded {len(bad_stems)} mislabeled stems (expected {EXPECTED_N})")

# 2. Load master CSV -- assert all stems present
import pandas as pd
df = pd.read_csv(args.master_csv)
EXPECTED_TOTAL_BEFORE = 20037
if len(df) != EXPECTED_TOTAL_BEFORE:
    abort(f"Expected {EXPECTED_TOTAL_BEFORE} rows in master, got {len(df)}")

in_master = bad_stems & set(df["filename_stem"])
if len(in_master) != EXPECTED_N:
    abort(f"Expected all {EXPECTED_N} mislabeled stems in master; only {len(in_master)} found. "
          f"Missing: {bad_stems - in_master}")
print(f"[OK] All {EXPECTED_N} stems present in master ({EXPECTED_TOTAL_BEFORE} rows total)")

# 3. Assert these are all human/mouse (sanity check)
bad_species = df[df["filename_stem"].isin(bad_stems)]["heavy_species"].unique()
if not all(s in ("homo sapiens", "mus musculus") for s in bad_species):
    abort(f"Unexpected species in mislabeled set: {bad_species}")
print(f"[OK] Species check: {dict(df[df['filename_stem'].isin(bad_stems)]['heavy_species'].value_counts())}")

# 4. Check exclusions.csv doesn't already have sabdab_chain_mislabel rows
if os.path.exists(args.exclusions_csv):
    excl_df = pd.read_csv(args.exclusions_csv)
    already = excl_df[excl_df.get("stage", pd.Series(dtype=str)) == "sabdab_chain_mislabel"] \
              if "stage" in excl_df.columns else pd.DataFrame()
    if len(already) > 0:
        abort(f"exclusions.csv already contains {len(already)} sabdab_chain_mislabel rows -- "
              f"patch already applied? Aborting to avoid double-application.")
    print(f"[OK] exclusions.csv exists ({len(excl_df)} rows), no sabdab_chain_mislabel rows yet")
else:
    abort(f"exclusions.csv not found at {args.exclusions_csv} -- pass the correct "
          f"--exclusions-csv path.")

# 5. Build clean master
clean_df = df[~df["filename_stem"].isin(bad_stems)].copy()
EXPECTED_CLEAN = 19848
if len(clean_df) != EXPECTED_CLEAN:
    abort(f"Expected {EXPECTED_CLEAN} rows in clean master, got {len(clean_df)}")

# 6. Build new exclusion rows
bad_df = df[df["filename_stem"].isin(bad_stems)].copy()
scope_by_stem = {r["filename_stem"]: r for r in scope["per_entry"]
                 if r.get("anarci_chain_type") in ("K", "L")}
new_rows = []
for _, row in bad_df.iterrows():
    chain = scope_by_stem.get(row["filename_stem"], {}).get("anarci_chain_type", "?")
    new_rows.append({
        "filename_stem": row["filename_stem"],
        "pdb_id": row.get("pdb_id", ""),
        "stage": "sabdab_chain_mislabel",
        "detail": (
            f"ANARCI HMM classifier identified the sequence stored in SAbDab's "
            f"Hchain slot as a light-chain locus ({'kappa' if chain=='K' else 'lambda'}). "
            f"Direct sequence inspection confirms a canonical light-chain variable domain. "
            f"Root cause: SAbDab annotation error -- Hchain assigned to a light-chain "
            f"sequence in a light-chain-only deposition with no Lchain partner recorded."
        ),
        "heavy_species": row.get("heavy_species", ""),
        "anarci_chain_type": chain,
    })
new_excl_df = pd.DataFrame(new_rows)
if len(new_excl_df) != EXPECTED_N:
    abort(f"Built {len(new_excl_df)} exclusion rows, expected {EXPECTED_N}")

print(f"\n[DRY-RUN SUMMARY]" if args.dry_run else "\n[WRITE SUMMARY]")
print(f"  master_antibodies.csv: {EXPECTED_TOTAL_BEFORE} rows")
print(f"  master_antibodies_clean.csv: {EXPECTED_CLEAN} rows (to write)")
print(f"  exclusions.csv: {len(excl_df)} rows + {EXPECTED_N} new sabdab_chain_mislabel rows")
print(f"  Species removed: "
      f"{dict(bad_df['heavy_species'].value_counts())}")

if args.dry_run:
    print("\n[DRY-RUN] No files written. Re-run without --dry-run to apply.")
    sys.exit(0)

# 7. Write -- master_antibodies_clean.csv first, then exclusions
clean_df.to_csv(args.out_master_csv, index=False)
print(f"[WROTE] {args.out_master_csv} ({len(clean_df)} rows)")

# Append to exclusions.csv
updated_excl = pd.concat([excl_df, new_excl_df], ignore_index=True)
updated_excl.to_csv(args.exclusions_csv, index=False)
print(f"[WROTE] {args.exclusions_csv} ({len(updated_excl)} rows, was {len(excl_df)})")

# Integrity check
reloaded = pd.read_csv(args.out_master_csv)
if len(reloaded) != EXPECTED_CLEAN:
    abort(f"[INTEGRITY] Re-read master_antibodies_clean.csv: expected {EXPECTED_CLEAN}, "
          f"got {len(reloaded)}")
reloaded_excl = pd.read_csv(args.exclusions_csv)
new_stage_count = (reloaded_excl["stage"] == "sabdab_chain_mislabel").sum()
if new_stage_count != EXPECTED_N:
    abort(f"[INTEGRITY] sabdab_chain_mislabel rows in exclusions.csv: "
          f"expected {EXPECTED_N}, got {new_stage_count}")
print(f"[INTEGRITY OK] master_antibodies_clean.csv={len(reloaded)} rows, "
      f"exclusions sabdab_chain_mislabel={new_stage_count}")
print("\n[DONE] Patch applied successfully.")
