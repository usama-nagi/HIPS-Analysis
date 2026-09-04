"""
common.py
=========
Shared utilities for the SAbDab bias/benchmark analysis pipeline.
No dependency on the existing training-pipeline code (preprocess_sabdab.py,
assign_antigen_clusters.py, oas_loader.py) - this module only reads the
*outputs* of those pipelines (the .pt files, TSVs, and mmap directory) via
its own independent code path, so the paper's repo is self-contained and
auditable on its own.

Every analysis script imports from here for:
  - config loading
  - schema validation with loud, specific failures (no silent fallback to
    wrong numbers if a column or .bin file is missing/renamed)
  - shared diversity metrics (Shannon entropy, Gini coefficient, JS divergence)
"""

import os
import sys
import csv
import json
import yaml
import hashlib
import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# Delimiter detection
def detect_delimiter(path: str, candidates: Sequence[str] = ("\t", ",", ";")) -> str:
    """
    Sniff the field delimiter of a tabular text file. SAbDab's own export is
    tab-separated; Thera-SAbDab's public export is comma-separated - rather
    than hardcoding either, every script in this pipeline detects the
    delimiter from the actual file so a CSV-vs-TSV mismatch never silently
    produces a one-giant-column parse.

    Uses csv.Sniffer on the header line first; falls back to whichever
    candidate delimiter appears most often in the header if Sniffer can't
    decide (e.g. a header with no quoted fields and ambiguous punctuation).
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        header_line = f.readline()

    try:
        dialect = csv.Sniffer().sniff(header_line, delimiters="".join(candidates))
        detected = dialect.delimiter
        if detected in candidates:
            return detected
    except csv.Error:
        pass

    # Fallback: whichever candidate occurs most in the header line
    counts = {d: header_line.count(d) for d in candidates}
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        raise ValueError(
            f"[DELIMITER DETECTION FAILED] Could not detect a delimiter for "
            f"{path!r} among candidates {candidates!r} - header line was: "
            f"{header_line[:200]!r}. Pass the delimiter explicitly if this "
            f"file uses something unusual."
        )
    return best

# Config
def load_config(config_path: str = None) -> dict:
    """Load config.yaml. Defaults to configs/config.yaml relative to this file."""
    if config_path is None:
        config_path = str(Path(__file__).resolve().parent.parent / "configs" / "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config not found at {config_path}. "
            f"Copy configs/config.yaml and edit the `paths:` section to point "
            f"at your real SAbDab/Thera-SAbDab/OAS locations before running anything."
        )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg

def require_path(path: str, what: str):
    """Fail loudly and specifically if an expected input path doesn't exist."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"[MISSING INPUT] {what} not found at: {path!r}\n"
            f"  -> Check configs/config.yaml `paths:` section."
        )



# Schema validation
# Columns we depend on existing in the raw SAbDab summary TSV. If the live
# TSV doesn't have one of these (e.g. SAbDab changes its export schema),
# we want a clear error naming the missing column, not a KeyError three
# scripts downstream with no context.
REQUIRED_SABDAB_TSV_COLUMNS = [
    "pdb", "Hchain", "Lchain", "model", "antigen_chain", "antigen_type",
    "antigen_name", "date", "organism", "heavy_species", "light_species",
    "antigen_species", "resolution", "method", "heavy_subclass",
    "light_subclass", "light_ctype",
]

REQUIRED_THERA_SABDAB_COLUMNS = [
    "Therapeutic", "Format", "Highest_Clin_Trial (Feb '25)", "Est. Status",
    "100% SI Structure", "99% SI Structure", "95-98% SI Structure",
    "Target", "Genetics (Bispecifics delimited with semicolon)",
]

# Keys we depend on existing inside each preprocessed SAbDab .pt sample dict.
# Mirrors the `sample = {...}` dict built in preprocess_sabdab.py's
# process_entry(), plus the antigen_cluster_id added by
# assign_antigen_clusters.py. We read these as plain dict keys - we do not
# import either of those scripts.
REQUIRED_PT_KEYS = [
    "source", "pdb_id", "paired", "heavy", "light", "coords", "coord_mask",
    "cdr_mask", "cdr3_len_actual", "has_antigen", "antigen_cluster_id",
]

# Keys we depend on existing in the OAS mmap meta.json, mirroring the schema
# that data/loaders/oas_loader.py reads.
REQUIRED_OAS_META_KEYS = [
    "total_samples", "max_len", "source_map", "source_counts",
    "has_clono_hash", "has_cdr3_len", "hard_filters",
]

def validate_tsv_columns(fieldnames: Sequence[str], required: Sequence[str], source_name: str):
    missing = [c for c in required if c not in fieldnames]
    if missing:
        raise ValueError(
            f"[SCHEMA MISMATCH] {source_name} is missing expected column(s): {missing}\n"
            f"  Found columns: {list(fieldnames)}\n"
            f"  -> The source file's schema has likely changed. Update "
            f"REQUIRED_*_COLUMNS in common.py after confirming the new "
            f"column names, do not silently proceed."
        )


def validate_pt_sample(sample: dict, pt_path: str):
    missing = [k for k in REQUIRED_PT_KEYS if k not in sample]
    if missing:
        raise ValueError(
            f"[SCHEMA MISMATCH] {pt_path} is missing expected key(s): {missing}\n"
            f"  Found keys: {list(sample.keys())}\n"
            f"  -> This .pt file's schema doesn't match what preprocess_sabdab.py "
            f"is documented to produce. Re-check the preprocessing pipeline version."
        )

def validate_oas_meta(meta: dict, meta_path: str):
    missing = [k for k in REQUIRED_OAS_META_KEYS if k not in meta]
    if missing:
        raise ValueError(
            f"[SCHEMA MISMATCH] {meta_path} is missing expected key(s): {missing}\n"
            f"  Found keys: {list(meta.keys())}\n"
        )
    if not meta.get("has_cdr3_len", False):
        raise ValueError(
            f"[SCOPE BLOCKER] {meta_path} reports has_cdr3_len=False. "
            f"The OAS-comparison scripts in this pipeline require cdr3_len.bin. "
            f"Re-run your OAS mmap conversion with CDR3 length export enabled, "
            f"or remove the OAS-comparison step from scope."
        )

# Diversity metrics
def shannon_entropy(counts: np.ndarray, base: float = 2.0) -> float:
    """Shannon entropy in bits (base=2) over a frequency/count vector."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * (np.log(p) / np.log(base))).sum())

def normalized_entropy(counts: np.ndarray, base: float = 2.0) -> float:
    """Shannon entropy divided by log2(n_categories) - in [0, 1]. Lets you
    compare entropy across variables with different numbers of categories
    (e.g. 7 species vs. 40 germline families) on a common scale."""
    counts = np.asarray(counts, dtype=np.float64)
    n_categories = int((counts > 0).sum())
    if n_categories <= 1:
        return 0.0
    h = shannon_entropy(counts, base=base)
    h_max = np.log(n_categories) / np.log(base)
    return float(h / h_max) if h_max > 0 else 0.0

def gini_coefficient(counts: np.ndarray) -> float:
    """Gini coefficient over a frequency/count vector. 0 = perfectly even,
    1 = maximally concentrated in one category."""
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts >= 0]
    if counts.sum() == 0 or len(counts) == 0:
        return 0.0
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    cum = np.cumsum(sorted_counts)
    # Standard Gini formula via Lorenz curve summation
    gini = (2.0 * np.sum((np.arange(1, n + 1)) * sorted_counts) - (n + 1) * cum[-1]) / (n * cum[-1])
    return float(gini)

def simpson_diversity(counts: np.ndarray) -> float:
    """Simpson's diversity index (1 - sum(p_i^2)). Higher = more diverse."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return float(1.0 - np.sum(p ** 2))

def jensen_shannon_divergence(p_counts: np.ndarray, q_counts: np.ndarray,
                               support: Optional[Sequence] = None) -> float:
    """
    JS divergence (base-2, bounded [0,1]) between two empirical distributions
    given as raw counts over the SAME support. If support differs between
    the two count vectors (e.g. different length ranges observed), pass
    `support` as the union of categories so both are padded/aligned first
    by the caller - this function assumes p_counts and q_counts are already
    aligned to the same category order and length.
    """
    p_counts = np.asarray(p_counts, dtype=np.float64)
    q_counts = np.asarray(q_counts, dtype=np.float64)
    if p_counts.shape != q_counts.shape:
        raise ValueError(
            f"jensen_shannon_divergence: p_counts shape {p_counts.shape} != "
            f"q_counts shape {q_counts.shape}. Align both distributions to "
            f"the same category support before calling this function."
        )
    p = p_counts / max(p_counts.sum(), 1e-12)
    q = q_counts / max(q_counts.sum(), 1e-12)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-12))))

    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))

def align_distributions_on_support(counts_a: dict, counts_b: dict):
    """
    Given two {category: count} dicts with possibly different keys, return
    (support_list, aligned_a, aligned_b) as numpy arrays over the union of
    categories, zero-filled where a category is absent in one distribution.
    """
    support = sorted(set(counts_a.keys()) | set(counts_b.keys()))
    a = np.array([counts_a.get(k, 0) for k in support], dtype=np.float64)
    b = np.array([counts_b.get(k, 0) for k in support], dtype=np.float64)
    return support, a, b

def sha8(seq: str) -> str:
    """8-char SHA-256 prefix. Matches the hashing convention already used in
    assign_antigen_clusters.py, reimplemented independently here so this
    module has no import dependency on that script."""
    return hashlib.sha256(seq.encode()).hexdigest()[:8]


# Misc
def setup_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def log(msg: str):
    print(f"[{Path(sys.argv[0]).stem}] {msg}", flush=True)