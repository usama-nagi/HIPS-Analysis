#!/usr/bin/env python3
"""
rq1_sequence_structural_bias.py
=================================
RQ1: What sequence, structural, and metadata biases exist in SAbDab's
CDR-H3 repertoire?

Five sections, each independently runnable via --only:

  SECTION A — sequence & metadata bias
      length, family-level germline, species, therapeutic enrichment
      (via Thera-SAbDab join), temporal trend, method/resolution confound

  SECTION B — structural redundancy & paratope composition (H3 only)
      CDR-H3 sequence redundancy via MMseqs2 (near-duplicate collapse,
      NOT full backbone RMSD/TM-score clustering -- see Section D for the
      bounded backbone-RMSD extension that partially covers this), plus
      paratope (interface_mask) vs. full-loop amino-acid composition for
      CDR-H3 specifically.

  SECTION C — antigen landscape
      coarse antigen-class classification (keyword heuristic, stated as
      a limitation) and antigen-class-vs-therapeutic cross-tabulation.
      Antigen redundancy itself is not recomputed here -- it reuses
      antigen_cluster_id, already computed by assign_antigen_clusters.py
      (see Section B's redundancy summary, which reports on that column
      directly).

  SECTION D — backbone structural redundancy within CDR-H3 clusters
      Computes backbone C-alpha RMSD WITHIN each non-singleton CDR-H3
      sequence cluster from Section B, never across clusters -- this is a
      bounded, cheap way to check whether sequence-redundant entries are
      ALSO structurally redundant, without attempting full all-pairs
      backbone clustering across the whole 20,003-entry dataset (which
      remains out of scope; see module docstring inline in run_section_d).
      DEPENDS ON Section B having already been run at least once, since it
      reads tables/rq1_cdrh3_clusters.tsv.

  SECTION E — paratope composition across all six CDR loops
      Extends Section B's H3-only paratope composition to all six CDR
      loops (H1, H2, H3, L1, L2, L3), using the same cdr_mask /
      interface_mask fields. Reports full-loop composition under TWO
      scopes -- all entries, and antigen-bound-only -- because Section B's
      own full_h3_aa_composition is, despite its field name, already
      silently restricted to antigen-bound entries (its antigen-gating
      `continue` fires before its full_counts counter is populated). The
      antigen_bound_only scope here is what's directly comparable to
      Section B's H3 numbers; see run_section_e's docstring for the full
      explanation of why this distinction matters.
      INDEPENDENT of Section B/D -- rescans .pt files itself.

Reads ONLY outputs/tables/master_antibodies.csv (built by
00_build_dataset.py) plus the underlying .pt files (Sections B, D, E need
the raw heavy_seq/cdr_mask/interface_mask/coords, which aren't flattened
into the CSV). No dependency on preprocess_sabdab.py or
assign_antigen_clusters.py -- those are read only as data, never imported.

Output (all under outputs/tables/)
-----------------------------------
rq1_sequence_metadata_bias.json        (Section A)
rq1_length_distribution.csv
rq1_heavy_germline_family_distribution.csv
rq1_light_germline_family_distribution.csv
rq1_heavy_species_distribution.csv
rq1_antigen_species_distribution.csv
rq1_year_distribution.csv
rq1_method_vs_length.csv

rq1_cdrh3_clusters.tsv                 (Section B)
rq1_structural_redundancy_paratope.json

rq1_antigen_class_distribution.csv     (Section C)
rq1_antigen_class_vs_therapeutic.csv
rq1_antigen_landscape.json

rq1_backbone_redundancy.json           (Section D)

rq1_paratope_all_cdrs.json             (Section E)

Usage
-----
    python scripts/rq1_sequence_structural_bias.py --config configs/config.yaml
    python scripts/rq1_sequence_structural_bias.py --config configs/config.yaml --only A
    python scripts/rq1_sequence_structural_bias.py --config configs/config.yaml --only B --skip_mmseqs
    python scripts/rq1_sequence_structural_bias.py --config configs/config.yaml --only D
    python scripts/rq1_sequence_structural_bias.py --config configs/config.yaml --only E

NOTE ON RUN ORDER: Section D requires Section B to have been run at least
once already (it reads tables/rq1_cdrh3_clusters.tsv, which Section B
produces). Running --only D before B has ever run will fail with a clear
file-not-found error rather than silently doing nothing. Default (no
--only) runs A, B, C, D, E in that order, which satisfies this dependency
automatically.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from collections import Counter

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    load_config, require_path, shannon_entropy, normalized_entropy,
    gini_coefficient, sha8, log,
)

AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")

# Keyword lists for the antigen-class heuristic. Validated against a
# manually-annotated sample (see validate_antigen_classifier_sample.py /
# validate_antigen_classifier_score.py and the paper's classifier-validation
# appendix); precision/recall by class are reported there, not here. Two
# properties of the matching logic are deliberate design choices, not
# defaults:
#   1. Matching uses word boundaries (see classify_antigen below), not raw
#      substring containment, so e.g. "il-" does not match inside
#      "microfibril-associated", "cd3" does not match inside "cd320", and
#      "rna"/"dna" do not match inside unrelated compound words. This is
#      not applied to "toxin" or "tumor"/"tumour" -- see
#      KEYWORD_EXCLUSIONS below for why those two need substring matching.
#   2. KEYWORD_EXCLUSIONS (below) handles the two keywords where word-
#      boundary matching alone is insufficient.
# Several entries below are synonym or spelled-out forms (e.g. "erbb-2" for
# HER2, "epidermal growth factor receptor" for EGFR) included because the
# dataset contains entries under those forms rather than the primary
# abbreviation, so the abbreviation alone would miss them.
ANTIGEN_CLASS_KEYWORDS = [
    ("viral", [
        "virus", "viral", "spike", "glycoprotein f0", "hemagglutinin",
        "influenza", "hiv", "sars", "coronavirus", "ebola", "dengue",
        "morbillivirus", "rsv", "respiratory syncytial",
        # Viral structural-protein and assembly terminology: capsid,
        # envelope/fusion glycoproteins, virion components, and named viral
        # proteins (gp120/gp160/gp41/gp140, hcv, mers, etc.). This
        # vocabulary is essentially always viral in an antibody-antigen
        # structural database.
        "capsid protein", "gp120", "gp160", "gp41", "gp140",
        "glycoprotein b", "glycoprotein h", "glycoprotein g",
        "genome polyprotein", "envelope glycoprotein", "envelope protein",
        "envelope polyprotein", "virion", "hcv", "mers", "matrix protein 2",
        "nucleoprotein", "fusion glycoprotein", "premembrane",
        "vp1", "vp2", "vp3", "sosip", "togavirin",
    ]),
    ("bacterial", [
        "bacteri", "toxin", "lipopolysaccharide", "lps", "anthrax",
        "staphylococcus", "streptococcus", "clostridium",
    ]),
    ("cancer_associated", [
        "tumor", "tumour", "carcinoma", "oncogen", "cd20", "cd19", "her2",
        "egfr", "psma", "muc16", "ca125", "pd-1", "pd1", "pdl1", "pd-l1",
        "ctla", "cd38", "bcma",
        # Synonym / spelled-out forms for targets already listed above by
        # abbreviation, needed because entries use one form or the other,
        # not both.
        "erbb-2", "erbb2",  # = HER2
        "epidermal growth factor receptor",  # = EGFR
        "glutamate carboxypeptidase 2",  # = PSMA
        "programmed cell death protein 1",  # = PD-1
        "cytotoxic t-lymphocyte protein 4",  # = CTLA-4
        "adp-ribosyl cyclase",  # = CD38
    ]),
    ("immune_receptor", [
        "cd3", "cd28", "cd40", "tcr", "t-cell receptor", "fc receptor",
        "fc gamma", "complement", "tnfrsf", "interleukin", "il-", "il2",
        "il6", "il13", "cytokine",
        "toll-like receptor", "hla-", "mhc class", "histocompatibility antigen",
        # TNF and its receptor superfamily are cytokines/cytokine receptors,
        # almost always spelled out in full rather than as the "tnf"/
        # "tnfrsf" abbreviation above, so both forms are listed. This also
        # gives the KEYWORD_EXCLUSIONS skip on "tumor" (below) a correct
        # category to land in instead of falling through to other_protein.
        "tnf", "tumor necrosis factor", "tumour necrosis factor",
        "tumor necrosis factor receptor superfamily",
        "tumour necrosis factor receptor superfamily",
        # C5a/C3a anaphylatoxin receptors are genuine complement-system
        # immune receptors, kept as their own specific keyword rather than
        # bare "anaphylatoxin" (which is excluded from "toxin" only, not
        # added as a positive match anywhere, since not every
        # anaphylatoxin-containing name refers to this receptor).
        "anaphylatoxin receptor", "anaphylatoxin chemotactic receptor",
    ]),
    ("enzyme", [
        "kinase", "protease", "polymerase", "synthase", "phosphatase",
        "nuclease",
        # Additional enzyme-class keywords; without these, entries using
        # them fall through to nucleic_acid_or_carbohydrate via an
        # unrelated "dna"/"rna" substring collision (e.g. "ATP-dependent
        # DNA helicase", "RNase A").
        "amylase", "aldolase", "helicase", "isomerase", "oxidase",
        "dismutase", "decarboxylase", "peptidase", "rnase",
    ]),
    ("nucleic_acid_or_carbohydrate", [
        "dna", "rna", "polysaccharide", "glycan",
    ]),
    ("unknown_or_other", [
        "unknown",
    ]),
]

# Per-keyword exclusion substrings: if any exclusion string for a keyword is
# present in the text, that keyword is skipped for this entry (other
# keywords and categories are still checked normally). Word-boundary
# matching (see classify_antigen) does not help for either entry below:
#   - "toxin": "anaphylatoxin"/"lymphotoxin" (immune cytokines) and non-
#     bacterial venom toxin names (bungarotoxin/cobratoxin/theraphotoxin/
#     conotoxin/"mammal toxin") are fused compound words with no word
#     boundary before "toxin", the same shape as genuinely bacterial fused
#     compounds this keyword must keep matching (neurotoxin, enterotoxin)
#     -- a boundary rule cannot distinguish them, only an explicit list of
#     the non-bacterial cases can.
#   - "tumor"/"tumour": "tumor necrosis factor" is a genuine standalone-word
#     match, not a substring collision, so no boundary rule applies at all;
#     TNF and its receptor superfamily are cytokines/cytokine receptors,
#     not tumor antigens.
KEYWORD_EXCLUSIONS = {
    "toxin": [
        "anaphylatoxin", "lymphotoxin", "bungarotoxin", "cobratoxin",
        "theraphotoxin", "conotoxin", "mammal toxin", "toxin extrusion",
    ],
    "tumor": ["tumor necrosis factor"],
    "tumour": ["tumour necrosis factor"],
}


# ═════════════════════════════════════════════════════════════════════════
# SECTION A — sequence & metadata bias
# ═════════════════════════════════════════════════════════════════════════

def year_from_date(date_str: str):
    """SAbDab date format observed: MM/DD/YY (e.g. '04/15/26'). Two-digit
    years are ambiguous so we apply the standard convention: 00-69 ->
    2000-2069, 70-99 -> 1970-1999, stated explicitly rather than left
    implicit."""
    if not date_str or not isinstance(date_str, str):
        return None
    parts = date_str.strip().split("/")
    if len(parts) != 3:
        return None
    try:
        yy = int(parts[2])
    except ValueError:
        return None
    return 2000 + yy if yy <= 69 else 1900 + yy


def run_section_a(df: pd.DataFrame, work_dir: str, cdr3_min: int, cdr3_max: int) -> dict:
    log("=" * 70)
    log("SECTION A: sequence & metadata bias")

    n_before = len(df)
    df = df[(df["cdr3_len_actual"] >= cdr3_min) & (df["cdr3_len_actual"] <= cdr3_max)]
    log(f"Applied CDR3 length filter [{cdr3_min},{cdr3_max}]: {n_before} -> {len(df)} rows")

    out = {"n_antibodies_analyzed": len(df), "cdr3_length_window": [cdr3_min, cdr3_max]}

    lens = df["cdr3_len_actual"].dropna().astype(int)
    len_counts = lens.value_counts().sort_index()
    out["length"] = {
        "mean": float(lens.mean()), "median": float(lens.median()),
        "std": float(lens.std()), "min": int(lens.min()), "max": int(lens.max()),
        "entropy_bits": shannon_entropy(len_counts.values),
        "normalized_entropy": normalized_entropy(len_counts.values),
        "gini": gini_coefficient(len_counts.values),
    }
    len_counts.rename("count").to_csv(os.path.join(work_dir, "tables", "rq1_length_distribution.csv"))

    for col, label in [("heavy_subclass", "heavy_germline_family"),
                        ("light_subclass", "light_germline_family")]:
        if col not in df.columns:
            log(f"WARNING: column {col} absent — skipping {label}")
            continue
        vc = df[col].fillna("UNKNOWN").value_counts()
        out[label] = {
            "n_unique_families": int(vc.shape[0]),
            "entropy_bits": shannon_entropy(vc.values),
            "normalized_entropy": normalized_entropy(vc.values),
            "gini": gini_coefficient(vc.values),
            "top5": vc.head(5).to_dict(),
            "top5_fraction_of_total": float(vc.head(5).sum() / vc.sum()),
        }
        vc.rename("count").to_csv(os.path.join(work_dir, "tables", f"rq1_{label}_distribution.csv"))

    for col, label in [("heavy_species", "heavy_species"),
                        ("antigen_species", "antigen_species")]:
        if col not in df.columns:
            continue
        vc = df[col].fillna("unknown").value_counts()
        out[label] = {
            "n_unique": int(vc.shape[0]),
            "entropy_bits": shannon_entropy(vc.values),
            "top5": vc.head(5).to_dict(),
        }
        vc.rename("count").to_csv(os.path.join(work_dir, "tables", f"rq1_{label}_distribution.csv"))

    # Heavy-vs-light species match rate, restricted to paired==True rows.
    # Reported both as a raw string comparison and as a whitespace/case-
    # normalized comparison, since some of the gap between the two is pure
    # formatting rather than genuine cross-species pairing; the normalized
    # rate is used as the primary statistic. Heavy-vs-antigen species match
    # rate is also reported separately below, as a distinct and independently
    # interesting number -- most antibodies legitimately target a
    # foreign-species antigen, so a low heavy-vs-antigen match rate reflects
    # normal antibody-antigen biology, not a chain-pairing quality signal,
    # and the two should not be conflated.
    if "paired" in df.columns and "heavy_species" in df.columns and "light_species" in df.columns:
        paired_mask = df["paired"] == True if df["paired"].dtype == bool else \
            df["paired"].astype(str).str.lower() == "true"
        paired_df = df[paired_mask].dropna(subset=["heavy_species", "light_species"])
        paired_df = paired_df[(paired_df["heavy_species"] != "") & (paired_df["light_species"] != "")]
        if len(paired_df) > 0:
            hs_raw = paired_df["heavy_species"]
            ls_raw = paired_df["light_species"]
            hs_norm = hs_raw.astype(str).str.strip().str.lower()
            ls_norm = ls_raw.astype(str).str.strip().str.lower()
            raw_match = (hs_raw == ls_raw)
            norm_match = (hs_norm == ls_norm)
            mismatches = paired_df[~norm_match]
            top_mismatch_pairs = (
                mismatches.groupby(["heavy_species", "light_species"]).size()
                .sort_values(ascending=False).head(10)
            )
            out["cross_species_pairing"] = {
                "description": "Fraction of paired (heavy+light) entries where "
                                "heavy_species and light_species match. This is a "
                                "chain-pairing consistency check, NOT a comparison "
                                "against antigen_species -- see "
                                "heavy_antigen_species_match below for that.",
                "n_evaluated": int(len(paired_df)),
                "fraction_same_species_normalized": float(norm_match.mean()),
                "fraction_same_species_raw_string": float(raw_match.mean()),
                "top10_mismatched_species_pairs": {
                    f"{h} / {l}": int(c) for (h, l), c in top_mismatch_pairs.items()
                },
            }

    if "heavy_species" in df.columns and "antigen_species" in df.columns:
        valid = df.dropna(subset=["heavy_species", "antigen_species"])
        valid = valid[(valid["heavy_species"] != "") & (valid["antigen_species"] != "")]
        if len(valid) > 0:
            hs_norm = valid["heavy_species"].astype(str).str.strip().str.lower()
            ags_norm = valid["antigen_species"].astype(str).str.strip().str.lower()
            match = (hs_norm == ags_norm)
            out["heavy_antigen_species_match"] = {
                "description": "Fraction of antigen-bearing entries where the "
                                "antibody's heavy_species matches its "
                                "antigen_species (e.g. self-antigen / same-species "
                                "target). Renamed from the original, misleadingly-"
                                "named 'cross_species_pairing' field this was "
                                "previously stored under -- see cross_species_pairing "
                                "above for the actual heavy/light chain-pairing "
                                "consistency statistic.",
                "n_evaluated": int(len(valid)),
                "fraction_same_species": float(match.mean()),
            }

    if "is_therapeutic" in df.columns:
        out["therapeutic"] = {
            "fraction_therapeutic": float(df["is_therapeutic"].mean()),
            "n_therapeutic": int(df["is_therapeutic"].sum()),
            "n_non_therapeutic": int((~df["is_therapeutic"]).sum()),
        }
        if "genetics_class" in df.columns:
            gc = df.loc[df["is_therapeutic"], "genetics_class"].dropna()
            if len(gc) > 0:
                out["therapeutic"]["genetics_class_breakdown"] = gc.value_counts().to_dict()
        if len(df.loc[df["is_therapeutic"], "cdr3_len_actual"].dropna()) > 0:
            out["therapeutic"]["mean_cdr3_len_therapeutic"] = float(
                df.loc[df["is_therapeutic"], "cdr3_len_actual"].mean())
            out["therapeutic"]["mean_cdr3_len_non_therapeutic"] = float(
                df.loc[~df["is_therapeutic"], "cdr3_len_actual"].mean())

    if "date" in df.columns:
        df = df.copy()
        df["year"] = df["date"].apply(year_from_date)
        year_counts = df["year"].dropna().astype(int).value_counts().sort_index()
        out["temporal"] = {
            "year_range": [int(year_counts.index.min()), int(year_counts.index.max())]
            if len(year_counts) else None,
            "n_with_valid_year": int(year_counts.sum()),
            "n_missing_year": int(df["year"].isna().sum()),
        }
        if "is_therapeutic" in df.columns:
            ther_by_year = df.dropna(subset=["year"]).groupby("year")["is_therapeutic"].mean()
            out["temporal"]["therapeutic_fraction_by_year"] = {
                int(k): float(v) for k, v in ther_by_year.items()
            }
        year_counts.rename("count").to_csv(os.path.join(work_dir, "tables", "rq1_year_distribution.csv"))

    if "method" in df.columns:
        method_len = df.groupby("method")["cdr3_len_actual"].agg(["mean", "std", "count"])
        out["method_length_confound"] = method_len.to_dict(orient="index")
        method_len.to_csv(os.path.join(work_dir, "tables", "rq1_method_vs_length.csv"))

    out_path = os.path.join(work_dir, "tables", "rq1_sequence_metadata_bias.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log(f"Wrote {out_path}")
    return out


# ═════════════════════════════════════════════════════════════════════════
# SECTION B — structural redundancy & paratope composition (H3 only)
# ═════════════════════════════════════════════════════════════════════════

def extract_cdrh3_sequences(master_df: pd.DataFrame):
    """Returns (cdrh3_seqs, index_to_filename_stem) where cdrh3_seqs is
    {integer_index: sequence} and index_to_filename_stem is
    {integer_index: filename_stem}.

    An integer index, not filename_stem, is used as the canonical FASTA
    sequence ID. 2,623 of 20,003 filename_stem values contain a literal
    space character (the multi-antigen-chain naming convention, e.g.
    "10en_FE_agC _ D_m0"), and MMseqs2's easy-cluster treats everything
    after the first whitespace in a FASTA header as a discarded
    description field, silently truncating any ID that contains one.
    Carrying the real filename_stem separately in Python and using an
    integer as the FASTA ID avoids this class of problem regardless of
    what characters filename_stem contains.
    """
    cdrh3_seqs = {}
    index_to_filename_stem = {}
    n_errors = 0
    idx = 0
    for _, row in master_df.iterrows():
        pt_path = row["pt_path"]
        try:
            sample = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception:
            n_errors += 1
            continue

        heavy_seq = sample.get("heavy", {}).get("sequence_aa", "")
        cdr_mask = sample.get("cdr_mask")
        if not heavy_seq or cdr_mask is None:
            continue

        h_len = len(heavy_seq)
        heavy_cdr_mask = cdr_mask[:h_len].numpy().astype(bool)

        padded = np.concatenate(([False], heavy_cdr_mask, [False]))
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        if len(starts) == 0:
            continue
        h3_start, h3_end = starts[-1], ends[-1]
        h3_seq = heavy_seq[h3_start:h3_end]

        expected_len = int(row.get("cdr3_len_actual", -1))
        if expected_len >= 0 and len(h3_seq) != expected_len:
            log(f"  WARNING: {pt_path} H3 length mismatch: "
                f"recomputed={len(h3_seq)} vs stored cdr3_len_actual={expected_len}")

        if len(h3_seq) >= 3:
            cdrh3_seqs[idx] = h3_seq
            index_to_filename_stem[idx] = row["filename_stem"]
            idx += 1

    log(f"Extracted {len(cdrh3_seqs)} CDR-H3 sequences ({n_errors} .pt load errors)")
    n_with_space = sum(1 for v in index_to_filename_stem.values() if " " in v)
    if n_with_space > 0:
        log(f"  {n_with_space} filename_stem values contain a space character "
            f"(handled via integer FASTA indices, translated back to real "
            f"filename_stem values after clustering; see extract_cdrh3_sequences "
            f"docstring).")
    return cdrh3_seqs, index_to_filename_stem


def run_mmseqs2_cluster(cdrh3_seqs: dict, index_to_filename_stem: dict, tmp_dir: str,
                         min_seq_id: float, coverage: float, n_threads: int) -> dict:
    """Same MMseqs2 invocation as before, but FASTA headers are integer
    indices, translated back to real filename_stem values after MMseqs2
    returns (see extract_cdrh3_sequences docstring for why). Returns
    {filename_stem: cluster_rep_filename_stem}, same shape callers expect.

    Exact-duplicate CDR-H3 strings are collapsed to one canonical index
    before clustering, for the following reason:

    MMseqs2's prefilter is k-mer-seed-based (default k=6 for proteins). A
    query sequence shorter than the k-mer length -- and CDR-H3 sequences as
    short as 3 residues are explicitly retained by this pipeline's own
    [3,35] length filter -- can fail to generate any seed, including
    against a byte-identical sequence, which can split exact-duplicate
    CDR-H3 strings across multiple "clusters" even though 100% identity
    trivially exceeds any min-seq-id threshold. This failure is
    length-correlated: short sequences are disproportionately affected
    relative to longer ones, consistent with k-mer prefilter seed failure
    rather than random noise.

    Rather than tuning MMseqs2's -k parameter (which only moves the
    threshold at which this can still happen, and would need independent
    justification/validation of its own), exact-duplicate strings are
    collapsed to one canonical index before clustering. Only one
    representative per unique string is ever written to the FASTA file and
    clustered; every other index sharing that exact string is guaranteed,
    by construction, to receive the same cluster_rep as its canonical
    index -- this is a Python dict lookup, not dependent on MMseqs2's
    seed-finding succeeding. This makes the exact-duplicate-split failure
    mode structurally impossible rather than merely rare. Near-duplicate
    (non-identical) clustering behavior for sequences too short to seed is
    a separate, harder problem, out of scope here since it does not affect
    any correctness claim this paper makes.
    """
    # ---- collapse exact duplicates to one canonical index per string ----
    seq_to_canonical_idx = {}
    idx_to_canonical_idx = {}
    canonical_seqs = {}
    for idx, seq in cdrh3_seqs.items():
        canon = seq_to_canonical_idx.setdefault(seq, idx)
        idx_to_canonical_idx[idx] = canon
        if canon == idx:
            canonical_seqs[idx] = seq

    n_total = len(cdrh3_seqs)
    n_canonical = len(canonical_seqs)
    log(f"Exact-duplicate collapse before clustering: {n_total} sequences -> "
        f"{n_canonical} unique strings clustered (structurally guarantees "
        f"exact duplicates can never be split across clusters, regardless "
        f"of MMseqs2 k-mer seeding behavior on short sequences).")

    os.makedirs(tmp_dir, exist_ok=True)
    fasta_path = os.path.join(tmp_dir, "cdrh3.fasta")
    with open(fasta_path, "w") as f:
        for idx, seq in canonical_seqs.items():
            f.write(f">{idx}\n{seq}\n")

    prefix = os.path.join(tmp_dir, "cdrh3_cluster")
    mmseqs_tmp = os.path.join(tmp_dir, "mmseqs_tmp")
    cmd = ["mmseqs", "easy-cluster", fasta_path, prefix, mmseqs_tmp,
           "--min-seq-id", str(min_seq_id), "--cov-mode", "0",
           "-c", str(coverage), "--threads", str(n_threads), "-v", "2"]
    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"MMseqs2 failed with exit code {result.returncode}")

    tsv_path = prefix + "_cluster.tsv"
    require_path(tsv_path, "MMseqs2 cluster output TSV")

    canonical_member_to_rep_idx = {}
    n_unmapped = 0
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                try:
                    rep_idx, member_idx = int(parts[0]), int(parts[1])
                except ValueError:
                    n_unmapped += 1
                    continue
                if rep_idx not in canonical_seqs or member_idx not in canonical_seqs:
                    n_unmapped += 1
                    continue
                canonical_member_to_rep_idx[member_idx] = rep_idx

    if n_unmapped > 0:
        raise RuntimeError(
            f"[INTEGRITY ERROR] {n_unmapped} lines in {tsv_path} referenced an index "
            f"not present in this run's canonical_seqs map. This should be "
            f"impossible with integer FASTA IDs -- investigate before trusting any "
            f"cluster assignment from this run."
        )

    missing_canonical = set(canonical_seqs.keys()) - set(canonical_member_to_rep_idx.keys())
    if missing_canonical:
        raise RuntimeError(
            f"[INTEGRITY ERROR] {len(missing_canonical)} canonical indices sent to "
            f"clustering do not appear in the parsed cluster TSV. Sample: "
            f"{list(missing_canonical)[:5]}. Investigate before trusting cluster_rep "
            f"assignments."
        )

    # ---- expand back: every original idx inherits its canonical idx's cluster_rep ----
    member_to_rep = {
        index_to_filename_stem[idx]:
            index_to_filename_stem[canonical_member_to_rep_idx[idx_to_canonical_idx[idx]]]
        for idx in cdrh3_seqs
    }

    missing = set(index_to_filename_stem.values()) - set(member_to_rep.keys())
    if missing:
        raise RuntimeError(
            f"[INTEGRITY ERROR] {len(missing)} filename_stem values extracted for "
            f"clustering do not appear in the final member_to_rep mapping. Sample: "
            f"{list(missing)[:5]}. This indicates a FASTA-ID or lookup mismatch -- "
            f"investigate before trusting cluster_rep assignments."
        )

    # ---- self-check: exact duplicates must be structurally incapable of ----
    # landing in different clusters. Re-verifies that invariant against the
    # final output, not just the intermediate step.
    check_df_rep = {}
    n_violations = 0
    for idx, seq in cdrh3_seqs.items():
        stem = index_to_filename_stem[idx]
        rep = member_to_rep[stem]
        prior = check_df_rep.setdefault(seq, rep)
        if prior != rep:
            n_violations += 1
    if n_violations > 0:
        raise RuntimeError(
            f"[INTEGRITY ERROR] {n_violations} exact-duplicate CDR-H3 strings were "
            f"still assigned inconsistent cluster_rep values after the collapse-and-"
            f"expand fix. This should be structurally impossible -- investigate the "
            f"patch itself before trusting this run."
        )

    return member_to_rep


def paratope_composition_analysis(master_df: pd.DataFrame) -> dict:
    """NOTE (see Section E docstring): full_counts here is only ever
    populated for has_antigen==True rows -- the antigen-gating `continue`
    below fires before full_counts is touched. "full_h3_aa_composition" in
    this function's output is, despite the name, already restricted to
    antigen-bound entries. Section E's full_loop_aa_composition_antigen_bound_only
    field reproduces this exact scope for all six loops."""
    full_counts = Counter()
    paratope_counts = Counter()
    n_with_interface = 0
    n_no_interface_residues = 0

    for _, row in master_df.iterrows():
        if not row.get("has_antigen", False):
            continue
        pt_path = row["pt_path"]
        try:
            sample = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception:
            continue

        heavy_seq = sample.get("heavy", {}).get("sequence_aa", "")
        cdr_mask = sample.get("cdr_mask")
        interface_mask = sample.get("interface_mask")
        if not heavy_seq or cdr_mask is None or interface_mask is None:
            continue

        h_len = len(heavy_seq)
        heavy_cdr = cdr_mask[:h_len].numpy().astype(bool)
        heavy_iface = interface_mask[:h_len].numpy().astype(bool)

        padded = np.concatenate(([False], heavy_cdr, [False]))
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        if len(starts) == 0:
            continue
        h3_start, h3_end = starts[-1], ends[-1]

        h3_seq = heavy_seq[h3_start:h3_end]
        h3_iface = heavy_iface[h3_start:h3_end]

        for aa in h3_seq:
            if aa in AA_ALPHABET:
                full_counts[aa] += 1

        n_with_interface += 1
        if h3_iface.any():
            for aa, is_iface in zip(h3_seq, h3_iface):
                if is_iface and aa in AA_ALPHABET:
                    paratope_counts[aa] += 1
        else:
            n_no_interface_residues += 1

    return {
        "n_antigen_complexes_evaluated": n_with_interface,
        "n_with_zero_interface_residues_in_h3": n_no_interface_residues,
        "full_h3_aa_composition": dict(full_counts),
        "paratope_h3_aa_composition": dict(paratope_counts),
    }


def run_section_b(master_df: pd.DataFrame, work_dir: str, cfg: dict, skip_mmseqs: bool) -> dict:
    log("=" * 70)
    log("SECTION B: structural redundancy (CDR-H3 sequence clustering) & paratope composition")
    log("NOTE: this is CDR-H3 SEQUENCE redundancy via MMseqs2, not full backbone "
        "RMSD/TM-score clustering. See Section D for a bounded, within-cluster-only "
        "backbone RMSD check that partially addresses this without the cost of full "
        "all-pairs structural clustering.")

    summary = {}

    if not skip_mmseqs:
        cdrh3_seqs, index_to_filename_stem = extract_cdrh3_sequences(master_df)
        tmp_dir = os.path.join(work_dir, "tmp_mmseqs_cdrh3")
        min_seq_id = cfg["mmseqs2"]["cdrh3_min_seq_id"]
        coverage = cfg["mmseqs2"]["cdrh3_coverage"]
        n_threads = cfg["mmseqs2"]["n_threads"]

        member_to_rep = run_mmseqs2_cluster(
            cdrh3_seqs, index_to_filename_stem, tmp_dir, min_seq_id, coverage, n_threads
        )

        # member_to_rep is {filename_stem: cluster_rep_filename_stem} for
        # every filename_stem -- run_mmseqs2_cluster raises an integrity
        # error itself if any are missing, so no silent .get(k, k) fallback
        # is used here (a fallback of that kind would turn any mis-keyed
        # row into a spurious singleton cluster without surfacing an error).
        clusters_df = pd.DataFrame([
            {"filename_stem": idx_to_stem, "cdrh3_seq": cdrh3_seqs[idx],
             "cluster_rep": member_to_rep[idx_to_stem]}
            for idx, idx_to_stem in index_to_filename_stem.items()
        ])
        clusters_df.to_csv(os.path.join(work_dir, "tables", "rq1_cdrh3_clusters.tsv"),
                            sep="\t", index=False)

        cluster_sizes = clusters_df["cluster_rep"].value_counts()
        n_singletons = int((cluster_sizes == 1).sum())
        summary["cdrh3_redundancy"] = {
            "min_seq_id_threshold": min_seq_id,
            "n_sequences": len(clusters_df),
            "n_unique_clusters": int(cluster_sizes.shape[0]),
            "n_singleton_clusters": n_singletons,
            "singleton_fraction": float(n_singletons / max(cluster_sizes.shape[0], 1)),
            "compression_ratio": float(len(clusters_df) / max(cluster_sizes.shape[0], 1)),
            "top10_largest_clusters": cluster_sizes.head(10).to_dict(),
            "exact_duplicate_fraction": float(
                clusters_df["cdrh3_seq"].duplicated(keep=False).mean()
            ),
        }
        log(f"CDR-H3 clustering: {cluster_sizes.shape[0]} unique clusters from "
            f"{len(clusters_df)} sequences (compression "
            f"{summary['cdrh3_redundancy']['compression_ratio']:.2f}x)")
    else:
        log("Skipping MMseqs2 clustering (--skip_mmseqs)")

    if "antigen_cluster_id" in master_df.columns:
        ag_df = master_df[master_df["has_antigen"] == True]
        ag_clusters = ag_df["antigen_cluster_id"].dropna()
        if len(ag_clusters) > 0:
            cluster_sizes = ag_clusters.value_counts()
            top10_frac = float(cluster_sizes.head(10).sum() / cluster_sizes.sum())
            summary["antigen_redundancy_reused_from_existing_pipeline"] = {
                "n_antigen_complexes": int(len(ag_clusters)),
                "n_unique_antigen_clusters": int(cluster_sizes.shape[0]),
                "gini": gini_coefficient(cluster_sizes.values),
                "top10_cluster_dominance_fraction": top10_frac,
                "note": "Reuses antigen_cluster_id already computed by "
                        "assign_antigen_clusters.py (MMseqs2, 70% identity) — "
                        "not recomputed here.",
            }

    log("Running paratope composition analysis (interface_mask)...")
    summary["paratope_composition"] = paratope_composition_analysis(master_df)

    # Merge into any existing output file rather than overwriting it, so
    # that a --skip_mmseqs run (which never populates cdrh3_redundancy)
    # does not discard a cdrh3_redundancy block written by an earlier full
    # run. The same merge-not-overwrite pattern is used in
    # rq2_oas_comparison.py for the same reason.
    out_path = os.path.join(work_dir, "tables", "rq1_structural_redundancy_paratope.json")
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        log(f"Loaded existing {out_path} to merge into (preserves any "
            f"block not recomputed this run, e.g. cdrh3_redundancy when "
            f"run with --skip_mmseqs).")
    existing.update(summary)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    log(f"Wrote {out_path}")
    return existing


# ═════════════════════════════════════════════════════════════════════════
# SECTION C — antigen landscape
# ═════════════════════════════════════════════════════════════════════════

import re

# Word-boundary matching, not raw substring containment.
# \b requires a transition between a word character (\w: letters, digits,
# underscore) and a non-word character (or string start/end) on each side
# of the keyword. This is what makes "cd3" correctly reject "cd320" (the
# "3"->"2" transition inside "cd320" is between two word characters, so
# there is no boundary there) while still matching a standalone "CD3".
# Keywords containing regex-special characters (only "-" appears in this
# list, which is not special outside a character class) are used as-is.
_KEYWORD_PATTERN_CACHE = {}


# Keywords that must use plain substring matching rather than word-boundary
# matching. "toxin" is the only current member: it has legitimate fused-
# compound uses that must keep matching (neurotoxin, enterotoxin -- both
# genuine bacterial toxins with no word boundary before "toxin"), which are
# structurally identical to the fused compounds that must not match
# (anaphylatoxin, bungarotoxin, etc.) -- a boundary rule cannot distinguish
# these, only KEYWORD_EXCLUSIONS (above) can, so this keyword needs
# substring matching preserved for the exclusion list to have anything to
# work against.
PLAIN_SUBSTRING_KEYWORDS = {"toxin"}


def _keyword_matches(keyword: str, text: str) -> bool:
    if keyword in PLAIN_SUBSTRING_KEYWORDS:
        return keyword in text
    pattern = _KEYWORD_PATTERN_CACHE.get(keyword)
    if pattern is None:
        pattern = re.compile(r"\b" + re.escape(keyword) + r"\b")
        _KEYWORD_PATTERN_CACHE[keyword] = pattern
    return bool(pattern.search(text))


def classify_antigen(antigen_name: str, antigen_type: str) -> str:
    if not isinstance(antigen_name, str):
        antigen_name = ""
    if not isinstance(antigen_type, str):
        antigen_type = ""
    text = (antigen_name + " " + antigen_type).lower()
    for label, keywords in ANTIGEN_CLASS_KEYWORDS:
        for kw in keywords:
            if kw in KEYWORD_EXCLUSIONS:
                if any(excl in text for excl in KEYWORD_EXCLUSIONS[kw]):
                    continue
            if _keyword_matches(kw, text):
                return label
    return "other_protein"


def run_section_c(df: pd.DataFrame, work_dir: str) -> dict:
    log("=" * 70)
    log("SECTION C: antigen landscape")

    ag_df = df[df["has_antigen"] == True].copy()
    log(f"{len(ag_df)}/{len(df)} entries have an antigen")

    ag_df["antigen_class"] = ag_df.apply(
        lambda r: classify_antigen(r.get("antigen_name", ""), r.get("antigen_type", "")),
        axis=1,
    )

    class_counts = ag_df["antigen_class"].value_counts()
    class_counts.rename("count").to_csv(
        os.path.join(work_dir, "tables", "rq1_antigen_class_distribution.csv"))

    summary = {
        "n_antigen_complexes": int(len(ag_df)),
        "antigen_class_counts": class_counts.to_dict(),
        "antigen_class_entropy_bits": shannon_entropy(class_counts.values),
        "antigen_class_gini": gini_coefficient(class_counts.values),
        "classification_method": "coarse keyword heuristic over antigen_name + "
                                  "antigen_type — see ANTIGEN_CLASS_KEYWORDS in this "
                                  "script; stated as a limitation, not authoritative.",
    }

    if "is_therapeutic" in ag_df.columns:
        cross = pd.crosstab(ag_df["antigen_class"], ag_df["is_therapeutic"])
        cross.to_csv(os.path.join(work_dir, "tables", "rq1_antigen_class_vs_therapeutic.csv"))
        summary["antigen_class_vs_therapeutic_crosstab"] = cross.to_dict()

    out_path = os.path.join(work_dir, "tables", "rq1_antigen_landscape.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"Wrote {out_path}")
    return summary


# ═════════════════════════════════════════════════════════════════════════
# SECTION D — backbone structural redundancy within CDR-H3 clusters
#             [FOLDED IN FROM rq1_structural_redundancy_backbone.py]
#
# Confirmed (by direct inspection of a real .pt file) that `coords` in
# this pipeline has shape (n_residues, 3) -- already CA-only, no separate
# atom axis. The functions below assume this confirmed layout.
# ═════════════════════════════════════════════════════════════════════════

def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Minimal Kabsch superposition RMSD between two (N,3) coordinate sets.
    Assumes P and Q are already correspondence-matched (same length, same
    residue order) -- caller's responsibility, enforced by an equal-length
    check before this is ever invoked."""
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch in kabsch_rmsd: {P.shape} vs {Q.shape}")
    if P.shape[0] < 3:
        raise ValueError(f"Too few atoms ({P.shape[0]}) for Kabsch alignment")

    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)

    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T

    P_aligned = (R @ Pc.T).T
    diff = P_aligned - Qc
    rmsd = np.sqrt((diff ** 2).sum(axis=1).mean())
    return float(rmsd)


def isolate_h3_ca_coords(pt_dict):
    """Same H3-isolation logic as Section B: last contiguous True-block of
    cdr_mask within the heavy-chain-length prefix. coords is indexed
    directly by residue (confirmed shape (n_residues, 3), no atom axis)."""
    if "cdr_mask" not in pt_dict or "coords" not in pt_dict:
        return None

    cdr_mask = pt_dict["cdr_mask"]
    coords = pt_dict["coords"]
    heavy_seq = pt_dict.get("heavy", {}).get("sequence_aa", None)
    if heavy_seq is None:
        return None
    heavy_len = len(heavy_seq)

    if isinstance(cdr_mask, torch.Tensor):
        cdr_mask = cdr_mask.numpy()
    if isinstance(coords, torch.Tensor):
        coords = coords.numpy()

    heavy_mask = cdr_mask[:heavy_len]

    true_idx = np.where(heavy_mask)[0]
    if len(true_idx) == 0:
        return None

    blocks = []
    start = true_idx[0]
    prev = true_idx[0]
    for idx in true_idx[1:]:
        if idx != prev + 1:
            blocks.append((start, prev))
            start = idx
        prev = idx
    blocks.append((start, prev))
    h3_start, h3_end = blocks[-1]

    h3_coords = coords[h3_start:h3_end + 1]

    if np.isnan(h3_coords).any():
        return None

    return h3_coords


def run_section_d(work_dir: str, pt_dir: str, max_cluster_size: int = 200) -> dict:
    log("=" * 70)
    log("SECTION D: backbone structural redundancy within CDR-H3 clusters")
    log("Computes backbone CA-RMSD WITHIN each non-singleton CDR-H3 sequence cluster "
        "from Section B only, never across clusters. This is NOT a recomputation of "
        "full all-pairs backbone clustering, which remains out of scope for this "
        "paper's time/compute budget.")
    log("Cluster size is read directly from Section B's actual output for this "
        "dataset (clusters range up to 649 members); every skipped cluster is "
        "identified by name in n_clusters_skipped_too_large below, not just counted.")

    clusters_path = os.path.join(work_dir, "tables", "rq1_cdrh3_clusters.tsv")
    require_path(clusters_path, "rq1_cdrh3_clusters.tsv (run Section B first)")
    clusters_df = pd.read_csv(clusters_path, sep="\t")

    cluster_groups = clusters_df.groupby("cluster_rep")
    non_singleton = {k: v for k, v in cluster_groups if len(v) > 1}
    actual_largest = max((len(v) for v in non_singleton.values()), default=0)
    log(f"{len(non_singleton)} non-singleton clusters to process "
        f"(out of {clusters_df['cluster_rep'].nunique()} total clusters); "
        f"actual largest non-singleton cluster has {actual_largest} members "
        f"(max_cluster_size threshold for this run: {max_cluster_size}).")

    within_cluster_rmsds = []
    per_cluster_summaries = []
    n_length_mismatch_pairs = 0
    n_equal_length_pairs = 0
    n_load_failures = 0
    n_clusters_skipped_too_large = 0
    n_clusters_no_valid_pairs = 0
    skipped_too_large = []  # [{"cluster_rep": ..., "n_members": ...}, ...] -- disclosed,
                             # not just counted, per this project's exclusion-disclosure rule

    for cluster_rep, group in non_singleton.items():
        if len(group) > max_cluster_size:
            n_clusters_skipped_too_large += 1
            skipped_too_large.append({"cluster_rep": cluster_rep, "n_members": len(group)})
            continue

        members = []
        for _, row in group.iterrows():
            pt_path = os.path.join(pt_dir, f"{row['filename_stem']}.pt")
            try:
                pt_dict = torch.load(pt_path, map_location="cpu", weights_only=False)
            except Exception:
                n_load_failures += 1
                continue
            h3_ca = isolate_h3_ca_coords(pt_dict)
            if h3_ca is None:
                n_load_failures += 1
                continue
            members.append({
                "filename_stem": row["filename_stem"],
                "h3_len": h3_ca.shape[0],
                "h3_ca": h3_ca,
            })

        if len(members) < 2:
            n_clusters_no_valid_pairs += 1
            continue

        cluster_rmsds = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if a["h3_len"] != b["h3_len"]:
                    n_length_mismatch_pairs += 1
                    continue
                n_equal_length_pairs += 1
                try:
                    r = kabsch_rmsd(a["h3_ca"], b["h3_ca"])
                except ValueError:
                    n_load_failures += 1
                    continue
                cluster_rmsds.append(r)
                within_cluster_rmsds.append(r)

        if cluster_rmsds:
            per_cluster_summaries.append({
                "cluster_rep": cluster_rep,
                "n_members": len(group),
                "n_valid_members": len(members),
                "n_equal_length_pairs_compared": len(cluster_rmsds),
                "mean_rmsd_angstrom": float(np.mean(cluster_rmsds)),
                "max_rmsd_angstrom": float(np.max(cluster_rmsds)),
            })
        else:
            n_clusters_no_valid_pairs += 1

    if within_cluster_rmsds:
        arr = np.array(within_cluster_rmsds)
        rmsd_summary = {
            "n_pairs": len(arr),
            "mean_angstrom": float(arr.mean()),
            "median_angstrom": float(np.median(arr)),
            "std_angstrom": float(arr.std()),
            "min_angstrom": float(arr.min()),
            "max_angstrom": float(arr.max()),
            "p95_angstrom": float(np.percentile(arr, 95)),
            "fraction_under_2A": float((arr < 2.0).mean()),
            "fraction_under_1A": float((arr < 1.0).mean()),
        }
    else:
        rmsd_summary = None
        log("WARNING: No equal-length pairs were successfully compared.")

    total_pairs_attempted = n_equal_length_pairs + n_length_mismatch_pairs
    result = {
        "method": (
            "Backbone CA-RMSD computed WITHIN each non-singleton CDR-H3 sequence "
            "cluster only (90% identity clusters from Section B), never across "
            "clusters. Pairs with unequal H3 length are reported as a separate "
            "length-mismatch bucket and are NOT forced into a misleading truncated/"
            "padded comparison."
        ),
        "n_non_singleton_clusters_total": len(non_singleton),
        "n_clusters_skipped_too_large": n_clusters_skipped_too_large,
        "clusters_skipped_too_large": sorted(
            skipped_too_large, key=lambda d: -d["n_members"]
        ),
        "max_cluster_size_threshold": max_cluster_size,
        "n_clusters_with_no_valid_comparable_pairs": n_clusters_no_valid_pairs,
        "n_load_or_isolation_failures": n_load_failures,
        "n_equal_length_pairs_compared": n_equal_length_pairs,
        "n_length_mismatch_pairs_excluded": n_length_mismatch_pairs,
        "length_mismatch_fraction_of_attempted_pairs": (
            n_length_mismatch_pairs / total_pairs_attempted if total_pairs_attempted else None
        ),
        "within_cluster_ca_rmsd": rmsd_summary,
        "per_cluster_summaries_top20_by_mean_rmsd": sorted(
            per_cluster_summaries, key=lambda d: -d["mean_rmsd_angstrom"]
        )[:20],
    }

    if skipped_too_large:
        log(f"WARNING: {len(skipped_too_large)} cluster(s) exceeded "
            f"max_cluster_size={max_cluster_size} and were EXCLUDED from the "
            f"within-cluster RMSD statistics: "
            f"{[(d['cluster_rep'], d['n_members']) for d in skipped_too_large]}. "
            f"This exclusion is disclosed in the output JSON's "
            f"'clusters_skipped_too_large' field -- do not report the RMSD summary "
            f"as covering all non-singleton clusters without noting this.")

    out_path = os.path.join(work_dir, "tables", "rq1_backbone_redundancy.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"Wrote {out_path}")
    if rmsd_summary:
        log(f"Within-cluster CA-RMSD: mean={rmsd_summary['mean_angstrom']:.3f}A, "
            f"median={rmsd_summary['median_angstrom']:.3f}A, "
            f"{rmsd_summary['fraction_under_1A']:.1%} under 1A, "
            f"{rmsd_summary['fraction_under_2A']:.1%} under 2A")
    return result


# ═════════════════════════════════════════════════════════════════════════
# SECTION E — paratope composition across all six CDR loops
#             [FOLDED IN FROM rq1_paratope_all_cdrs.py, v2]
# ═════════════════════════════════════════════════════════════════════════

def find_contiguous_blocks(mask_1d: np.ndarray):
    """Return list of (start, end) inclusive index tuples for each
    contiguous True block in a 1D boolean array, in order."""
    true_idx = np.where(mask_1d)[0]
    if len(true_idx) == 0:
        return []
    blocks = []
    start = true_idx[0]
    prev = true_idx[0]
    for idx in true_idx[1:]:
        if idx != prev + 1:
            blocks.append((start, prev))
            start = idx
        prev = idx
    blocks.append((start, prev))
    return blocks


def isolate_cdr_loops(pt_dict):
    """Returns {'H1': (start,end), ..., 'L3': (start,end)} with absolute
    indices into the FULL concatenated heavy+light sequence (matching
    cdr_mask's own indexing), or None if isolation fails. Generalizes
    Section B's H3-only isolation to all three loops per chain."""
    if "cdr_mask" not in pt_dict:
        return None
    heavy_seq = pt_dict.get("heavy", {}).get("sequence_aa", None)
    if heavy_seq is None:
        return None
    heavy_len = len(heavy_seq)

    cdr_mask = pt_dict["cdr_mask"]
    if isinstance(cdr_mask, torch.Tensor):
        cdr_mask = cdr_mask.numpy()

    result = {}

    heavy_mask = cdr_mask[:heavy_len]
    heavy_blocks = find_contiguous_blocks(heavy_mask)
    labels_h = ["H1", "H2", "H3"]
    if len(heavy_blocks) > 3:
        return None
    for lbl, block in zip(labels_h[-len(heavy_blocks):], heavy_blocks):
        result[lbl] = block

    light_seq = pt_dict.get("light", {}).get("sequence_aa", None)
    is_paired = pt_dict.get("paired", False)
    if is_paired and light_seq is not None and len(light_seq) > 0:
        light_start = heavy_len
        light_len = len(light_seq)
        light_mask = cdr_mask[light_start:light_start + light_len]
        light_blocks_rel = find_contiguous_blocks(light_mask)
        if len(light_blocks_rel) <= 3:
            labels_l = ["L1", "L2", "L3"]
            for lbl, (s, e) in zip(labels_l[-len(light_blocks_rel):], light_blocks_rel):
                result[lbl] = (s + light_start, e + light_start)

    return result if result else None


def get_full_sequence(pt_dict):
    heavy_seq = pt_dict.get("heavy", {}).get("sequence_aa", "") or ""
    light_seq = pt_dict.get("light", {}).get("sequence_aa", "") or ""
    return heavy_seq + light_seq


# def run_section_e(work_dir: str, pt_dir: str) -> dict:
def run_section_e(work_dir: str, pt_dir: str, master_df: pd.DataFrame) -> dict:
    log("=" * 70)
    log("SECTION E: paratope composition across all six CDR loops")
    log("Extends Section B's H3-only paratope analysis to H1, H2, H3, L1, L2, L3. "
        "Reports full-loop composition under TWO scopes per loop: all_entries "
        "(every entry regardless of antigen status) and antigen_bound_only "
        "(matches Section B's H3 scope exactly, for direct comparison).")

    # pt_files = [f for f in os.listdir(pt_dir) if f.endswith(".pt")]
    # log(f"Found {len(pt_files)} .pt files in {pt_dir}")
    valid_stems = set(master_df["filename_stem"].astype(str))
    all_pt_files = [f for f in os.listdir(pt_dir) if f.endswith(".pt")]

    pt_files = [
        f for f in all_pt_files
        if f[:-3] in valid_stems
    ]

    log(
        f"Found {len(all_pt_files)} .pt files on disk, "
        f"{len(pt_files)} match the current {len(master_df)}-row master population "
        f"({len(all_pt_files) - len(pt_files)} excluded as not in current corpus)"
    )

    loop_names = ["H1", "H2", "H3", "L1", "L2", "L3"]
    full_composition_all = {loop: Counter() for loop in loop_names}
    full_composition_antigen_bound_only = {loop: Counter() for loop in loop_names}
    paratope_composition = {loop: Counter() for loop in loop_names}
    n_complexes_with_antigen = 0
    n_entries_processed = 0
    n_load_failures = 0
    n_loop_isolation_failures = 0
    n_with_zero_interface_per_loop = {loop: 0 for loop in loop_names}

    for fname in pt_files:
        try:
            pt_dict = torch.load(os.path.join(pt_dir, fname), map_location="cpu", weights_only=False)
        except Exception:
            n_load_failures += 1
            continue
        n_entries_processed += 1

        loops = isolate_cdr_loops(pt_dict)
        if loops is None:
            n_loop_isolation_failures += 1
            continue

        full_seq = get_full_sequence(pt_dict)
        has_antigen = bool(pt_dict.get("has_antigen", False))
        interface_mask = pt_dict.get("interface_mask", None)
        if isinstance(interface_mask, torch.Tensor):
            interface_mask = interface_mask.numpy()

        if has_antigen:
            n_complexes_with_antigen += 1

        for loop_name, (start, end) in loops.items():
            loop_seq = full_seq[start:end + 1]
            for aa in loop_seq:
                if aa in AA_ALPHABET:
                    full_composition_all[loop_name][aa] += 1
                    if has_antigen:
                        full_composition_antigen_bound_only[loop_name][aa] += 1

            if has_antigen and interface_mask is not None:
                loop_interface = interface_mask[start:end + 1]
                if loop_interface.sum() == 0:
                    n_with_zero_interface_per_loop[loop_name] += 1
                for aa, in_interface in zip(loop_seq, loop_interface):
                    if in_interface and aa in AA_ALPHABET:
                        paratope_composition[loop_name][aa] += 1

    result = {
        "method": (
            "Extension of Section B's H3-only paratope composition analysis to all "
            "six CDR loops (H1, H2, H3, L1, L2, L3), using the same cdr_mask and "
            "interface_mask fields. CDR loops are isolated as up to three contiguous "
            "True-blocks per chain, in N-to-C order; entries whose mask doesn't "
            "cleanly decompose into the expected block structure are excluded and "
            "counted in n_loop_isolation_failures rather than guessed at."
        ),
        # "n_pt_files_found": len(pt_files),
        "n_pt_files_found_in_current_master": len(pt_files),
        "n_entries_processed": n_entries_processed,
        "n_load_failures": n_load_failures,
        "n_loop_isolation_failures": n_loop_isolation_failures,
        "n_complexes_with_antigen": n_complexes_with_antigen,
        "n_with_zero_interface_residues_per_loop": n_with_zero_interface_per_loop,
        "full_loop_aa_composition_all_entries": {
            loop: dict(full_composition_all[loop]) for loop in loop_names
        },
        "full_loop_aa_composition_antigen_bound_only": {
            loop: dict(full_composition_antigen_bound_only[loop]) for loop in loop_names
        },
        "scope_note": (
            "full_loop_aa_composition_all_entries counts every entry, regardless of "
            "antigen status. full_loop_aa_composition_antigen_bound_only restricts to "
            "has_antigen==True entries only -- this matches Section B's "
            "full_h3_aa_composition scope exactly (Section B's antigen-gating "
            "`continue` fires before its full_counts is populated, so its 'full' "
            "composition is already antigen-bound-only despite the field name). Use "
            "*_antigen_bound_only for any comparison against Section B's H3 numbers; "
            "use *_all_entries for a more highly-powered, antigen-status-independent "
            "baseline."
        ),
        "paratope_aa_composition": {loop: dict(paratope_composition[loop]) for loop in loop_names},
    }

    out_path = os.path.join(work_dir, "tables", "rq1_paratope_all_cdrs.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"Wrote {out_path}")
    log(f"Processed {n_entries_processed} entries, {n_loop_isolation_failures} "
        f"loop-isolation failures, {n_complexes_with_antigen} with antigen.")
    for loop in loop_names:
        ft = sum(full_composition_all[loop].values())
        fta = sum(full_composition_antigen_bound_only[loop].values())
        pt = sum(paratope_composition[loop].values())
        log(f"  {loop}: full_loop_residues_all={ft}, "
            f"full_loop_residues_antigen_bound_only={fta}, paratope_residues={pt}")
    return result


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--only", choices=["A", "B", "C", "D", "E"], default=None,
                         help="Run only one section (A=sequence/metadata, "
                              "B=structural redundancy/paratope H3, C=antigen "
                              "landscape, D=backbone redundancy within clusters, "
                              "E=paratope composition all 6 CDR loops). "
                              "Default: run all five in order (A,B,C,D,E).")
    parser.add_argument("--skip_mmseqs", action="store_true",
                         help="Section B only: skip MMseqs2 clustering, run paratope analysis only")
    parser.add_argument("--max_cluster_size", type=int, default=200,
                         help="Section D only: skip clusters larger than this. The largest "
                              "known cluster is 127 members, so the default of 200 should "
                              "never actually trigger a skip -- this is a safety valve.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    work_dir = cfg["paths"]["work_dir"]
    pt_dir = cfg["paths"]["sabdab_pt_dir"]
    os.makedirs(os.path.join(work_dir, "tables"), exist_ok=True)

    master_csv = os.path.join(work_dir, "tables", "master_antibodies.csv")
    require_path(master_csv, "master_antibodies.csv (run 00_build_dataset.py first)")
    df = pd.read_csv(master_csv)
    log(f"Loaded {len(df)} antibody entries from {master_csv}")

    sections_to_run = [args.only] if args.only else ["A", "B", "C", "D", "E"]

    if "A" in sections_to_run:
        run_section_a(df, work_dir, cfg["filters"]["cdr3_min_len"], cfg["filters"]["cdr3_max_len"])
    if "B" in sections_to_run:
        run_section_b(df, work_dir, cfg, args.skip_mmseqs)
    if "C" in sections_to_run:
        run_section_c(df, work_dir)
    if "D" in sections_to_run:
        run_section_d(work_dir, pt_dir, args.max_cluster_size)
    # if "E" in sections_to_run:
    #     run_section_e(work_dir, pt_dir)
    if "E" in sections_to_run:
            run_section_e(work_dir, pt_dir, df)

    log("RQ1 analysis complete.")


if __name__ == "__main__":
    main()