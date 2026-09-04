"""
rq_germline_01_assign.py

Step 2 of 2 for the germline allele-resolution extension (Limitations and
Future Directions, Section 6). For each SAbDab antibody whose heavy_species
or light_species is Homo sapiens or Mus musculus, finds the best-matching
germline V gene and J gene (by local pairwise alignment identity, BLOSUM62)
against the OGRDB reference built in step 1, and reports the matched
allele name and identity percentage -- the same quantities ANARCI's
--assign_germline mode reports (v_gene, v_identity, j_gene, j_identity),
computed independently here rather than via ANARCI itself (see step 1's
docstring and this project's earlier investigation for why: HMMER's
build-from-source requirements and ANARCI's own currently-broken
IMGT-scraping install step made the original ANARCI route impractical for
this pip-only environment).

ASYMMETRIC REPORTING BY SPECIES -- READ BEFORE USING THIS SCRIPT'S OUTPUT:
Human germline reference names retain real, IMGT-derived family
information (e.g. IGHV1-18*01 -> family IGHV1), so human results are
reported at BOTH the per-allele match level AND rolled up into
family-level summary statistics, extending the paper's existing
family-level germline analysis to allele resolution.

Mouse germline reference names, by contrast, are OGRDB's own documented
placeholder convention for strain-inferred sequences that have not been
mapped to a confirmed gene/family (e.g. IGHV0-24BS*00 -- the "0" subgroup
and "*00" allele are explicitly non-meaningful placeholders, per OGRDB's
own published documentation, not real IMGT family designations). Every
mouse gene name in this reference carries the same uninformative "0"
family digit regardless of which gene it actually is. Reporting a
family-level rollup from these names would silently manufacture a false
"100% concentration in family 0" result for every mouse antibody -- not
a real finding, an artifact of the reference's naming convention. Mouse
results are therefore reported at the per-allele match and identity level
ONLY, with no family-level rollup attempted. This asymmetry is
intentional and should be preserved in any paper text drawing on this
output: human allele-level statistics are directly comparable to the
paper's existing family-level numbers; mouse allele-level statistics
exist but do not extend the family-level analysis the way human's does.

ALIGNMENT METHOD:
Local pairwise alignment (Bio.Align.PairwiseAligner, mode='local',
BLOSUM62 substitution matrix, gap open -10 / extend -0.5) rather than
global alignment, because a query's full heavy/light chain sequence
extends beyond the germline V gene's coverage (into CDR3/junction/FR4,
which is not part of the V gene segment) -- global alignment would
force-align this extra tail and deflate every identity score. Verified
against synthetic test cases (a query = germline + extra tail; a query
with introduced point mutations) before trusting this on real data: local
alignment correctly recovers exact and near-exact identity over the
germline-covered region without tail-length penalty.

Usage:
    python rq_germline_01_assign.py --config configs/config.yaml \
        --germline_dir tables/germline_reference \
        --master_csv tables/master_antibodies.csv

Reads:
    <germline_dir>/{Homo_sapiens,Mus_musculus_C57BL6}_{IGH,IGK,IGL}_{V,J}_aa.fasta
    (produced by rq_germline_00_fetch_reference.py)
    tables/master_antibodies.csv

Writes:
    tables/rq_germline_allele_assignment.json
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import pandas as pd
import torch
import yaml
from Bio import Align, SeqIO
from Bio.Align import substitution_matrices


def log(msg):
    print(f"[rq_germline_01_assign] {msg}", flush=True)


def build_aligner():
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    return aligner


def load_reference_set(germline_dir, prefix):
    """Loads a {prefix}_V_aa.fasta / {prefix}_J_aa.fasta pair from step 1's
    output into {gene_id: sequence} dicts. Raises clearly if either file
    is missing or empty, rather than silently matching against an
    incomplete reference."""
    v_path = os.path.join(germline_dir, f"{prefix}_V_aa.fasta")
    j_path = os.path.join(germline_dir, f"{prefix}_J_aa.fasta")
    for p, label in [(v_path, "V"), (j_path, "J")]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"[REFERENCE MISSING] {p} not found. Run "
                f"rq_germline_00_fetch_reference.py first and confirm it "
                f"reported all_ok=true before running this script."
            )
    v_genes = {rec.id: str(rec.seq) for rec in SeqIO.parse(v_path, "fasta")}
    j_genes = {rec.id: str(rec.seq) for rec in SeqIO.parse(j_path, "fasta")}
    if not v_genes or not j_genes:
        raise ValueError(
            f"[REFERENCE EMPTY] {prefix}: V genes={len(v_genes)}, "
            f"J genes={len(j_genes)} -- at least one is empty. Do not "
            f"proceed with an empty reference set."
        )
    log(f"Loaded reference {prefix}: {len(v_genes)} V genes, {len(j_genes)} J genes")
    return v_genes, j_genes


def best_match(aligner, query_seq, reference_dict, min_coverage=0.80):
    """Aligns query_seq against every sequence in reference_dict, returns
    (best_gene_id, identity_fraction, aligned_length, coverage_fraction)
    for the highest-identity match AMONG those whose alignment covers at
    least min_coverage of the reference gene's own length.

    WHY THE COVERAGE FLOOR IS NECESSARY:
    Identity computed as matches/aligned_columns alone can be gamed by a
    short, high-identity partial alignment that happens to avoid a
    query's point mutations, beating a longer, complete, biologically
    correct alignment that legitimately includes those mutations. For
    example: a synthetic query with 3 introduced mutations relative to
    its true germline gene can match a deliberately-corrupted decoy
    reference (real sequence for the first 50 residues, garbage for the
    rest) at 98% identity over a 50-residue aligned window, beating the
    true gene's 96.9% identity over the full 98-residue alignment --
    because the decoy's 50-residue window happens not to contain any of
    the 3 mutated positions. Requiring at least 80% of the reference
    gene's length to be covered by the alignment eliminates this failure
    mode: the decoy's 51% coverage is correctly rejected, while the true
    gene's 100% coverage is correctly accepted.

    Candidates with insufficient coverage are not considered at all, not
    just down-ranked -- if every candidate in a reference set fails the
    coverage floor for a given query (e.g. a malformed or truncated
    query sequence), this returns (None, 0.0, 0, 0.0) rather than
    falling back to a low-coverage guess."""
    best_id = None
    best_identity = -1.0
    best_aligned_len = 0
    best_coverage = 0.0
    for gene_id in sorted(reference_dict.keys()):
        ref_seq = reference_dict[gene_id]
        if len(ref_seq) == 0:
            continue
        try:
            aln = aligner.align(query_seq, ref_seq)[0]
        except Exception:
            continue
        aligned_q = str(aln[0])
        aligned_r = str(aln[1])
        matches = sum(1 for a, b in zip(aligned_q, aligned_r) if a == b and a != "-")
        aligned_len = sum(1 for a, b in zip(aligned_q, aligned_r) if a != "-" and b != "-")
        if aligned_len == 0:
            continue
        coverage = aligned_len / len(ref_seq)
        if coverage < min_coverage:
            continue
        identity = matches / aligned_len
        if identity > best_identity:
            best_identity = identity
            best_id = gene_id
            best_aligned_len = aligned_len
            best_coverage = coverage
    if best_id is None:
        return None, 0.0, 0, 0.0
    return best_id, best_identity, best_aligned_len, best_coverage


def family_from_human_gene_id(gene_id):
    """Extracts e.g. 'IGHV1' from 'IGHV1-18*01'. Returns None if the
    pattern doesn't match (should not happen for real human OGRDB names,
    but fails loudly rather than silently mis-grouping if it ever does)."""
    import re
    m = re.match(r"^(IG[HKL][VJ]\d+)", gene_id)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--germline_dir", required=True)
    ap.add_argument("--master_csv", default=None)
    ap.add_argument("--limit", type=int, default=None,
                     help="Process only the first N rows -- useful for a "
                          "quick correctness check before committing to "
                          "the full ~20,000-row run.")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    work_dir = config["paths"]["work_dir"]
    master_path = args.master_csv or os.path.join(work_dir, "tables", "master_antibodies.csv")

    master_df = pd.read_csv(master_path)
    if args.limit:
        master_df = master_df.head(args.limit)
        log(f"--limit {args.limit} applied: processing only the first "
            f"{len(master_df)} rows (NOT a full run -- do not treat "
            f"output from a --limit run as final).")
    log(f"Loaded {len(master_df)} antibody entries from {master_path}")

    aligner = build_aligner()

    # Load all six reference sets up front and fail fast if any are missing,
    # rather than discovering a missing reference partway through a long run.
    references = {}
    for prefix in ["Homo_sapiens_IGH", "Homo_sapiens_IGK", "Homo_sapiens_IGL",
                   "Mus_musculus_C57BL6_IGH", "Mus_musculus_C57BL6_IGK",
                   "Mus_musculus_C57BL6_IGL"]:
        references[prefix] = load_reference_set(args.germline_dir, prefix)

    species_to_chain_prefixes = {
        "homo sapiens": {"heavy": "Homo_sapiens_IGH",
                          "light_kappa": "Homo_sapiens_IGK",
                          "light_lambda": "Homo_sapiens_IGL"},
        "mus musculus": {"heavy": "Mus_musculus_C57BL6_IGH",
                          "light_kappa": "Mus_musculus_C57BL6_IGK",
                          "light_lambda": "Mus_musculus_C57BL6_IGL"},
    }

    results = []
    n_processed = 0
    n_skipped_species = 0
    n_skipped_no_sequence = 0
    n_errors = 0
    t_start = time.time()

    for idx, row in master_df.iterrows():
        heavy_species_raw = str(row.get("heavy_species", "") or "").strip().lower()
        light_species_raw = str(row.get("light_species", "") or "").strip().lower()

        entry_result = {
            "filename_stem": row.get("filename_stem"),
            "heavy_species": row.get("heavy_species"),
            "light_species": row.get("light_species"),
            "heavy": None,
            "light": None,
        }

        any_chain_processed = False

        # --- Heavy chain ---
        if heavy_species_raw in species_to_chain_prefixes:
            pt_path = row.get("pt_path")
            try:
                sample = torch.load(pt_path, map_location="cpu", weights_only=False)
                heavy_seq = sample.get("heavy", {}).get("sequence_aa", "")
            except Exception as e:
                heavy_seq = ""
                n_errors += 1

            if heavy_seq and len(heavy_seq) >= 20:
                prefix = species_to_chain_prefixes[heavy_species_raw]["heavy"]
                v_genes, j_genes = references[prefix]
                v_id, v_identity, v_len, v_cov = best_match(aligner, heavy_seq, v_genes)
                j_id, j_identity, j_len, j_cov = best_match(aligner, heavy_seq, j_genes)
                entry_result["heavy"] = {
                    "species": heavy_species_raw, "reference_prefix": prefix,
                    "v_gene": v_id, "v_identity": round(v_identity, 4),
                    "v_aligned_length": v_len, "v_coverage": round(v_cov, 4),
                    "j_gene": j_id, "j_identity": round(j_identity, 4),
                    "j_aligned_length": j_len, "j_coverage": round(j_cov, 4),
                    "v_family": (family_from_human_gene_id(v_id)
                                 if heavy_species_raw == "homo sapiens" and v_id else None),
                }
                any_chain_processed = True
            else:
                n_skipped_no_sequence += 1
        else:
            n_skipped_species += 1

        # --- Light chain (try kappa and lambda references, keep the
        # higher-identity V-gene match -- SAbDab's light_subclass field
        # does not reliably distinguish kappa/lambda for every entry, so
        # we let the alignment itself decide rather than trust a
        # potentially-missing or inconsistent upstream field) ---
        if light_species_raw in species_to_chain_prefixes:
            pt_path = row.get("pt_path")
            try:
                sample = torch.load(pt_path, map_location="cpu", weights_only=False)
                light_seq = sample.get("light", {}).get("sequence_aa", "")
            except Exception:
                light_seq = ""
                n_errors += 1

            if light_seq and len(light_seq) >= 20:
                kappa_prefix = species_to_chain_prefixes[light_species_raw]["light_kappa"]
                lambda_prefix = species_to_chain_prefixes[light_species_raw]["light_lambda"]

                kv_genes, kj_genes = references[kappa_prefix]
                lv_genes, lj_genes = references[lambda_prefix]

                kv_id, kv_identity, kv_len, kv_cov = best_match(aligner, light_seq, kv_genes)
                lv_id, lv_identity, lv_len, lv_cov = best_match(aligner, light_seq, lv_genes)

                # Both kappa and lambda failing the coverage floor entirely
                # (kv_id and lv_id both None) means this sequence couldn't
                # be confidently assigned to either locus -- record that
                # rather than defaulting to an arbitrary choice.
                if kv_id is None and lv_id is None:
                    entry_result["light"] = {
                        "species": light_species_raw, "reference_prefix": None,
                        "inferred_locus": None, "v_gene": None, "v_identity": 0.0,
                        "note": "Neither kappa nor lambda V reference met the "
                                "minimum coverage threshold for this sequence.",
                    }
                else:
                    if kv_identity >= lv_identity:
                        chosen_locus, chosen_prefix = "IGK", kappa_prefix
                        v_id, v_identity, v_len, v_cov = kv_id, kv_identity, kv_len, kv_cov
                        j_genes_for_chosen = kj_genes
                    else:
                        chosen_locus, chosen_prefix = "IGL", lambda_prefix
                        v_id, v_identity, v_len, v_cov = lv_id, lv_identity, lv_len, lv_cov
                        j_genes_for_chosen = lj_genes

                    j_id, j_identity, j_len, j_cov = best_match(aligner, light_seq, j_genes_for_chosen)

                    entry_result["light"] = {
                        "species": light_species_raw, "reference_prefix": chosen_prefix,
                        "inferred_locus": chosen_locus,
                        "kappa_v_identity": round(kv_identity, 4),
                        "lambda_v_identity": round(lv_identity, 4),
                        "v_gene": v_id, "v_identity": round(v_identity, 4),
                        "v_aligned_length": v_len, "v_coverage": round(v_cov, 4),
                        "j_gene": j_id, "j_identity": round(j_identity, 4),
                        "j_aligned_length": j_len, "j_coverage": round(j_cov, 4),
                        "v_family": (family_from_human_gene_id(v_id)
                                     if light_species_raw == "homo sapiens" and v_id else None),
                    }
                any_chain_processed = True

        if any_chain_processed:
            n_processed += 1
        results.append(entry_result)

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            eta_min = (len(master_df) - (idx + 1)) / rate / 60 if rate > 0 else float("nan")
            log(f"{idx + 1}/{len(master_df)} rows ({rate:.1f}/s, ETA {eta_min:.1f} min)")

    log(f"Done: {n_processed} entries had at least one chain assigned "
        f"({n_errors} .pt load errors during processing).")

    # --- Summary statistics ---
    # n_skipped_species and n_skipped_no_sequence are per-chain diagnostic
    # counters that only increment from the heavy-chain branch of the loop
    # above; there is no equivalent counter for the light chain. They do
    # NOT partition n_total_entries and must not be summed against it or
    # against each other: an entry whose heavy chain is out-of-scope but
    # whose light chain succeeds is counted both in n_processed (via
    # any_chain_processed) and in n_skipped_species (via the heavy-only
    # counter).
    #
    # The correct, disjoint entry-level partition is:
    #   n_entries_with_assignment (any_chain_processed) + n_entries_with_no_assignment
    #   == n_total_entries
    # Per-chain diagnostic counters are kept below for transparency about
    # *why* a chain didn't contribute, but are explicitly labeled as
    # per-chain counts that do not sum against n_total_entries or against
    # each other in any simple way.
    n_no_assignment = len(master_df) - n_processed
    summary = {
        "n_total_entries": len(master_df),
        "n_entries_with_assignment": n_processed,
        "n_entries_with_no_assignment": n_no_assignment,
        "entry_level_partition_note": (
            "n_entries_with_assignment + n_entries_with_no_assignment == "
            "n_total_entries by construction (any_chain_processed is a "
            "single boolean per entry) -- this is the correct top-line "
            "partition. The per_chain_diagnostics block below reports heavy "
            "and light chain outcomes SEPARATELY and does NOT sum against "
            "n_total_entries or against n_entries_with_assignment, because "
            "an entry can have one chain skipped (out-of-scope species or "
            "no usable sequence) while its other chain is successfully "
            "assigned -- treat per_chain_diagnostics as descriptive detail, "
            "not as a further partition of the entries above it."
        ),
        "n_pt_load_errors": n_errors,
        "species_scope": ["Homo sapiens", "Mus musculus"],
        "per_chain_diagnostics": {
            "heavy": {
                "n_species_out_of_scope": n_skipped_species,
                "n_species_in_scope_but_no_usable_sequence": n_skipped_no_sequence,
            },
            "note": "Light-chain-only diagnostic counters were not tracked "
                    "by this run (no equivalent to n_skipped_species/"
                    "n_skipped_no_sequence exists for the light-chain branch "
                    "above); derive them post hoc from per_entry if needed.",
        },
        "mouse_caveat": (
            "Mouse V/J gene names are OGRDB strain-inferred placeholder "
            "identifiers (e.g. IGHV0-24BS*00), not real IMGT family "
            "designations -- every mouse gene shares the same '0' family "
            "digit regardless of identity. No family-level rollup is "
            "computed for mouse; v_family is null for all mouse entries "
            "by design, not by omission."
        ),
    }

    # Human family-level rollup (heavy chain only, mirroring the paper's
    # existing family-level heavy_subclass statistic for direct comparability)
    human_heavy_families = Counter(
        r["heavy"]["v_family"] for r in results
        if r["heavy"] and r["heavy"]["species"] == "homo sapiens" and r["heavy"]["v_family"]
    )
    if human_heavy_families:
        total = sum(human_heavy_families.values())
        summary["human_heavy_v_family_allele_resolved"] = {
            "n_entries": total,
            "family_counts": dict(human_heavy_families.most_common()),
            "top2_fraction": sum(c for _, c in human_heavy_families.most_common(2)) / total,
        }

    human_heavy_identities = [
        r["heavy"]["v_identity"] for r in results
        if r["heavy"] and r["heavy"]["species"] == "homo sapiens" and r["heavy"]["v_gene"]
    ]
    mouse_heavy_identities = [
        r["heavy"]["v_identity"] for r in results
        if r["heavy"] and r["heavy"]["species"] == "mus musculus" and r["heavy"]["v_gene"]
    ]
    n_human_heavy_no_confident_match = sum(
        1 for r in results
        if r["heavy"] and r["heavy"]["species"] == "homo sapiens" and not r["heavy"]["v_gene"]
    )
    n_mouse_heavy_no_confident_match = sum(
        1 for r in results
        if r["heavy"] and r["heavy"]["species"] == "mus musculus" and not r["heavy"]["v_gene"]
    )
    if human_heavy_identities:
        summary["human_heavy_v_identity"] = {
            "n": len(human_heavy_identities),
            "mean": sum(human_heavy_identities) / len(human_heavy_identities),
            "min": min(human_heavy_identities),
            "max": max(human_heavy_identities),
            "n_no_confident_match": n_human_heavy_no_confident_match,
        }
    if mouse_heavy_identities:
        summary["mouse_heavy_v_identity"] = {
            "n": len(mouse_heavy_identities),
            "mean": sum(mouse_heavy_identities) / len(mouse_heavy_identities),
            "min": min(mouse_heavy_identities),
            "max": max(mouse_heavy_identities),
            "n_no_confident_match": n_mouse_heavy_no_confident_match,
        }

    out_path = os.path.join(work_dir, "tables", "rq_germline_allele_assignment.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_entry": results}, f, indent=2)

    log(f"Wrote {out_path}")
    log(f"Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()