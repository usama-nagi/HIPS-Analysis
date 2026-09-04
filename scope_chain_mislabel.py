"""
scope_chain_mislabel.py

Scope the heavy/light chain mislabeling found via the ANARCI cross-
validation: run ANARCI's chain-type call (assign_germline=False) on
every human/mouse entry SAbDab recorded as unpaired (light_species NaN),
since camelid entries in this same bucket are expected to be genuinely
single-domain and are excluded from this check.

Read-only: does not modify master_antibodies.csv or any .pt file.
Writes: <work_dir>/tables/chain_mislabel_scope.json

Usage
-----
    python scripts/scope_chain_mislabel.py --config configs/config.yaml \
        [--hmmerpath /path/to/hmmer/bin]
"""
import os
import sys
from pathlib import Path
import pandas as pd
import torch
import json
import time
import argparse
from anarci import anarci

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--hmmerpath", default="")
ap.add_argument("--master_csv", default=None,
                 help="Defaults to <work_dir>/tables/master_antibodies.csv")
ap.add_argument("--out", default=None,
                 help="Defaults to <work_dir>/tables/chain_mislabel_scope.json")
args = ap.parse_args()

cfg = load_config(args.config)
work_dir = cfg["paths"]["work_dir"]
master_csv = args.master_csv or os.path.join(work_dir, "tables", "master_antibodies.csv")
out_path = args.out or os.path.join(work_dir, "tables", "chain_mislabel_scope.json")

df = pd.read_csv(master_csv)
candidates = df[
    df["light_species"].isna()
    & df["heavy_species"].isin(["homo sapiens", "mus musculus"])
]
log(f"Candidate pool (human/mouse, SAbDab-recorded unpaired): {len(candidates)}")

sequences = []
stem_meta = {}
n_load_errors = 0
for _, row in candidates.iterrows():
    stem = row["filename_stem"]
    try:
        sample = torch.load(row["pt_path"], map_location="cpu", weights_only=False)
        seq = sample.get("heavy", {}).get("sequence_aa", "")
    except Exception as e:
        seq = ""
        n_load_errors += 1
        log(f"[LOAD ERROR] {stem}: {e}")
    if seq and len(seq) >= 20:
        sequences.append((stem, seq))
        stem_meta[stem] = {
            "pdb_id": row.get("pdb_id"),
            "heavy_species": row.get("heavy_species"),
            "heavy_len": len(seq),
        }
log(f"Loaded {len(sequences)} sequences ({n_load_errors} .pt load errors).")

t0 = time.time()
log(f"Running ANARCI chain-type call on {len(sequences)} sequences...")
numbered, alignment_details, hit_tables = anarci(
    sequences, scheme="imgt", assign_germline=False,
    allowed_species=["human", "mouse"], hmmerpath=args.hmmerpath,
)
log(f"ANARCI finished in {time.time() - t0:.1f}s.")

results = []
for (stem, seq), domains in zip(sequences, alignment_details):
    meta = stem_meta[stem]
    if not domains:
        results.append({**meta, "filename_stem": stem, "anarci_called": False,
                         "anarci_chain_type": None, "anarci_species": None})
        continue
    dom = domains[0]
    results.append({
        **meta, "filename_stem": stem, "anarci_called": True,
        "anarci_chain_type": dom.get("chain_type"),
        "anarci_species": dom.get("species"),
        "n_domains_found": len(domains),
    })

n_called = sum(1 for r in results if r["anarci_called"])
n_heavy = sum(1 for r in results if r["anarci_chain_type"] == "H")
n_light = sum(1 for r in results if r["anarci_chain_type"] in ("K", "L"))
n_no_call = sum(1 for r in results if not r["anarci_called"])

summary = {
    "n_candidates": len(candidates),
    "n_sequences_checked": len(sequences),
    "n_load_errors": n_load_errors,
    "n_anarci_called": n_called,
    "n_chain_type_H_correct": n_heavy,
    "n_chain_type_light_MISLABELED": n_light,
    "n_chain_type_K": sum(1 for r in results if r["anarci_chain_type"] == "K"),
    "n_chain_type_L": sum(1 for r in results if r["anarci_chain_type"] == "L"),
    "n_anarci_no_call": n_no_call,
    "by_species": {
        sp: {
            "n": len([r for r in results if r["heavy_species"] == sp]),
            "n_mislabeled": len([r for r in results if r["heavy_species"] == sp
                                  and r["anarci_chain_type"] in ("K", "L")]),
        }
        for sp in ["homo sapiens", "mus musculus"]
    },
}
log(json.dumps(summary, indent=2))

with open(out_path, "w") as f:
    json.dump({"summary": summary, "per_entry": results}, f, indent=2)
log(f"Wrote {out_path}")
