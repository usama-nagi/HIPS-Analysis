"""
anarci_germline_validation.py

NOTE ON MOUSE: unlike OGRDB (whose mouse V-gene names are documented
placeholders with no real family information -- see rq_germline_01_assign
.py's docstring), ANARCI's bundled mouse germline set uses real,
family-informative names (e.g. IGHV1-69*01, IGHV9-1*01). This lets this
script do something the paper's existing method explicitly cannot: report
a genuine mouse family-level distribution, independent of OGRDB. Treat
this as a secondary, exploratory finding -- it was not the primary
purpose of this validation and has not been cross-checked further.

NOTE ON CAMELIDS: ANARCI's allowed_species also includes 'alpaca', which
OGRDB does not cover at all. Not used by this script (out of scope for
this validation), but worth knowing about for the species-scope
limitation already discussed in the paper.

Usage:
    python anarci_germline_validation.py --config configs/config.yaml \
        --germline_json tables/rq_germline_allele_assignment.json \
        --master_csv tables/master_antibodies.csv \
        --n_sample 300 --seed 0

Reads:
    tables/rq_germline_allele_assignment.json (from rq_germline_01_assign.py)
    tables/master_antibodies.csv (for pt_path per filename_stem)
    <pt_path> per sampled entry (for the real heavy_species/sequence_aa)

Writes:
    tables/anarci_germline_cross_validation.json
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter

import pandas as pd
import torch
import yaml

try:
    from anarci import anarci
except ImportError:
    print(
        "[FATAL] Could not import anarci. This script requires the anarci "
        "package to be importable, and the hmmscan binary to be on PATH "
        "(`apt-get install hmmer` on Debian/Ubuntu).",
        file=sys.stderr,
    )
    raise


def log(msg):
    print(f"[anarci_germline_cross_validation] {msg}", flush=True)


def family_from_gene_id(gene_id):
    """Extracts e.g. 'IGHV1' from 'IGHV1-18*01'. Returns None on no match
    (mirrors rq_germline_01_assign.py's family_from_human_gene_id, but not
    restricted to human here since ANARCI's mouse names are also
    real/family-informative)."""
    if not gene_id:
        return None
    m = re.match(r"^(IG[HKL][VJ]\d+)", gene_id)
    return m.group(1) if m else None

OUR_SPECIES_TO_ANARCI = {
    "homo sapiens": "human",
    "mus musculus": "mouse",
}


def normalize_our_species(species):
    return OUR_SPECIES_TO_ANARCI.get(species, species)


def load_our_calls(germline_json_path):
    """Loads rq_germline_01_assign.py's per-entry heavy-chain v_gene calls,
    keyed by filename_stem. Only entries with a confident call (v_gene is
    not None) are retained -- comparing against "no call" is not a
    meaningful agreement question."""
    if not os.path.exists(germline_json_path):
        raise FileNotFoundError(
            f"[MISSING] {germline_json_path} not found. Run "
            f"rq_germline_01_assign.py first."
        )
    with open(germline_json_path) as f:
        data = json.load(f)
    by_stem = {}
    for entry in data["per_entry"]:
        heavy = entry.get("heavy")
        if heavy and heavy.get("v_gene"):
            by_stem[entry["filename_stem"]] = {
                "species": heavy["species"],
                "v_gene": heavy["v_gene"],
                "v_identity": heavy["v_identity"],
                "v_family": heavy.get("v_family"),
            }
    log(f"Loaded {len(by_stem)} entries with a confident heavy-chain call "
        f"from {germline_json_path}")
    return by_stem


def load_sabdab_confidence(master_df):
    """Maps filename_stem -> whether SAbDab's own pipeline made a confident
    heavy_subclass call, for the paper's existing 'low-confidence entries
    are more atypical' finding to be cross-checked against ANARCI
    agreement rather than just against the nearest-reference method's own
    identity score."""
    out = {}
    for _, row in master_df.iterrows():
        stem = row.get("filename_stem")
        subclass = str(row.get("heavy_subclass", "") or "").strip().lower()
        out[stem] = (subclass != "unknown" and subclass != "")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--germline_json", required=True)
    ap.add_argument("--master_csv", default=None)
    ap.add_argument("--n_sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hmmerpath", default="",
                     help="Path to hmmscan binary if not on PATH.")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    work_dir = config["paths"]["work_dir"]
    master_path = args.master_csv or os.path.join(work_dir, "tables", "master_antibodies.csv")

    master_df = pd.read_csv(master_path)
    log(f"Loaded {len(master_df)} entries from {master_path}")

    our_calls = load_our_calls(args.germline_json)
    sabdab_confident = load_sabdab_confidence(master_df)

    # Restrict the sampling frame to entries our method actually called AND
    # that exist in master_df with a pt_path -- fail loudly if the join is
    # unexpectedly small rather than silently sampling from a tiny pool.
    stem_to_ptpath = dict(zip(master_df["filename_stem"], master_df["pt_path"]))
    eligible_stems = [s for s in our_calls if s in stem_to_ptpath]
    if len(eligible_stems) < args.n_sample:
        log(f"[WARNING] Only {len(eligible_stems)} eligible entries found, "
            f"fewer than requested --n_sample {args.n_sample}. Using all "
            f"of them.")
    log(f"{len(eligible_stems)} entries eligible for sampling "
        f"(have both a confident our-method call and a pt_path).")

    rng = random.Random(args.seed)
    sample_stems = rng.sample(eligible_stems, min(args.n_sample, len(eligible_stems)))

    # Load real sequences for the sample
    sequences = []  # (stem, seq) for anarci()
    n_load_errors = 0
    for stem in sample_stems:
        pt_path = stem_to_ptpath[stem]
        try:
            sample = torch.load(pt_path, map_location="cpu", weights_only=False)
            seq = sample.get("heavy", {}).get("sequence_aa", "")
        except Exception as e:
            seq = ""
            n_load_errors += 1
            log(f"[LOAD ERROR] {stem}: {e}")
        if seq and len(seq) >= 20:
            sequences.append((stem, seq))
    log(f"Loaded {len(sequences)} real heavy-chain sequences "
        f"({n_load_errors} .pt load errors).")
    if not sequences:
        raise RuntimeError("[FATAL] No sequences loaded -- cannot proceed.")

    # --- Run real ANARCI ---
    t0 = time.time()
    log(f"Running ANARCI (assign_germline=True) on {len(sequences)} sequences...")
    numbered, alignment_details, hit_tables = anarci(
        sequences, scheme="imgt", assign_germline=True,
        allowed_species=["human", "mouse"], hmmerpath=args.hmmerpath,
    )
    log(f"ANARCI finished in {time.time() - t0:.1f}s.")

    # Compare, entry by entry
    comparisons = []
    for (stem, seq), domains in zip(sequences, alignment_details):
        our = our_calls[stem]
        if not domains:
            comparisons.append({
                "filename_stem": stem, "anarci_called": False,
                "our_v_gene": our["v_gene"], "our_species": our["species"],
            })
            continue
        # A sequence can contain more than one recognised domain (e.g. a
        # scFv); take the first, consistent with this being a per-chain
        # heavy-sequence input.
        dom = domains[0]
        anarci_species = dom.get("species")
        anarci_chain_type = dom.get("chain_type")
        germlines = dom.get("germlines", {})
        anarci_v_gene, anarci_v_identity = (germlines.get("v_gene") or (None, None))
        if isinstance(anarci_v_gene, tuple):
            # germlines['v_gene'] is [(species, gene_name), identity] per
            # anarci.py's run_germline_assignment -- unpack defensively.
            anarci_v_gene = anarci_v_gene[1]

        our_family = our["v_family"] or family_from_gene_id(our["v_gene"])
        anarci_family = family_from_gene_id(anarci_v_gene)

        comparisons.append({
            "filename_stem": stem,
            "anarci_called": True,
            "our_species": our["species"],
            "our_v_gene": our["v_gene"],
            "our_v_identity": our["v_identity"],
            "our_family": our_family,
            "anarci_species": anarci_species,
            "anarci_chain_type": anarci_chain_type,
            "anarci_v_gene": anarci_v_gene,
            "anarci_v_identity": round(anarci_v_identity, 4) if anarci_v_identity is not None else None,
            "anarci_family": anarci_family,
            "species_agree": (anarci_species == normalize_our_species(our["species"])),
            "family_agree": (our_family is not None and anarci_family is not None
                              and our_family == anarci_family),
            "allele_agree": (anarci_v_gene == our["v_gene"]),
            "sabdab_confident_heavy_call": sabdab_confident.get(stem),
        })

    # Summary statistics
    called = [c for c in comparisons if c.get("anarci_called")]
    n_no_anarci_call = len(comparisons) - len(called)

    def agreement_rate(subset, key):
        if not subset:
            return None
        return sum(1 for c in subset if c[key]) / len(subset)

    human_subset = [c for c in called if c["our_species"] == "homo sapiens"]
    mouse_subset = [c for c in called if c["our_species"] == "mus musculus"]

    sabdab_confident_subset = [c for c in called if c["sabdab_confident_heavy_call"] is True]
    sabdab_unconfident_subset = [c for c in called if c["sabdab_confident_heavy_call"] is False]

    summary = {
        "n_sampled": len(sample_stems),
        "n_sequences_loaded": len(sequences),
        "n_anarci_called": len(called),
        "n_anarci_no_call": n_no_anarci_call,
        "overall": {
            "species_agreement": agreement_rate(called, "species_agree"),
            "family_agreement": agreement_rate(called, "family_agree"),
            "allele_agreement": agreement_rate(called, "allele_agree"),
        },
        "human_only": {
            "n": len(human_subset),
            "family_agreement": agreement_rate(human_subset, "family_agree"),
            "allele_agreement": agreement_rate(human_subset, "allele_agree"),
        },
        "mouse_only": {
            "n": len(mouse_subset),
            "note": "Family-level agreement is NOT computed for mouse "
                    "against our_family, since our method's mouse "
                    "v_family is always null by design (OGRDB placeholder "
                    "names -- see rq_germline_01_assign.py). ANARCI's own "
                    "mouse family calls are reported separately below as "
                    "an exploratory, independent finding.",
            "anarci_family_distribution": dict(
                Counter(c["anarci_family"] for c in mouse_subset if c["anarci_family"]).most_common()
            ),
        },
        "sabdab_confidence_crosscheck": {
            "description": (
                "Does ANARCI agreement corroborate the paper's existing "
                "finding that entries SAbDab's own pipeline could not "
                "confidently germline-call (heavy_subclass == 'unknown') "
                "are more atypical? Compares allele agreement between "
                "our method and ANARCI, split by SAbDab's own confidence."
            ),
            "sabdab_confident": {
                "n": len(sabdab_confident_subset),
                "allele_agreement": agreement_rate(sabdab_confident_subset, "allele_agree"),
                "family_agreement": agreement_rate(sabdab_confident_subset, "family_agree"),
            },
            "sabdab_unconfident": {
                "n": len(sabdab_unconfident_subset),
                "allele_agreement": agreement_rate(sabdab_unconfident_subset, "allele_agree"),
                "family_agreement": agreement_rate(sabdab_unconfident_subset, "family_agree"),
            },
        },
    }

    out_path = os.path.join(work_dir, "tables", "anarci_germline_cross_validation.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_entry": comparisons}, f, indent=2)

    log(f"Wrote {out_path}")
    log(f"Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
