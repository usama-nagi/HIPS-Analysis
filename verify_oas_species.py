#!/usr/bin/env python3
"""
verify_oas_species.py
======================
Independently verifies the species restriction of the OAS derivative used
in RQ2, by reading the embedded per-file metadata header of every raw OAS
unit file directly (rather than trusting the runtime loader's species
filter). Supports the paper's OAS species-verification appendix.

Usage
-----
    python scripts/verify_oas_species.py --oas_raw_dir <raw_data>/oas_raw
"""
import argparse
import gzip, ast, json, glob, os
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--oas_raw_dir", required=True,
                     help="Directory containing the raw OAS unit files, "
                          "with human_heavy/, human_light/, and paired/ "
                          "subdirectories of *.csv.gz files.")
args = parser.parse_args()

base = args.oas_raw_dir
dirs = ["human_heavy", "human_light", "paired"]

def parse_header(line):
    line = line.strip()
    if line.startswith('"') and line.endswith('"'):
        line = line[1:-1].replace('""', '"')
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)

overall = Counter()
by_dir = {}
non_human_files = []
unreadable = []

for d in dirs:
    files = sorted(glob.glob(os.path.join(base, d, "*.csv.gz")))
    tally = Counter()
    for fp in files:
        try:
            with gzip.open(fp, "rt") as f:
                line = f.readline()
            meta = parse_header(line)
            if not isinstance(meta, dict):
                raise ValueError(f"parsed to {type(meta).__name__}, not dict")
            species = str(meta.get("Species", "MISSING")).lower()
        except Exception as e:
            species = "UNREADABLE"
            unreadable.append((fp, str(e)))
        tally[species] += 1
        overall[species] += 1
        if species not in ("human", "unreadable"):
            non_human_files.append((fp, species))
    by_dir[d] = tally
    print(f"{d} ({len(files)} files): {dict(tally)}")

print("\n=== OVERALL ===")
print(dict(overall))
print(f"\nTotal files scanned: {sum(overall.values())}")

if non_human_files:
    print(f"\n=== {len(non_human_files)} NON-HUMAN / UNEXPECTED FILES ===")
    for fp, sp in non_human_files[:30]:
        print(f"  {sp}: {os.path.basename(fp)}")
if unreadable:
    print(f"\n=== {len(unreadable)} UNREADABLE FILES (first 10) ===")
    for fp, err in unreadable[:10]:
        print(f"  {os.path.basename(fp)}: {err}")
