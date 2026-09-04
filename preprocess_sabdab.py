#!/usr/bin/env python3
"""
preprocess_sabdab.py
====================================
Structural preprocessing for "Hidden in Plain Sight". Reads the raw SAbDab
summary table and structure files, applies the exclusion criteria described
in the paper's Methods section, and writes one retained-entry .pt file per
antibody chain-pair. Every excluded row is logged with an individually
named exclusion reason (see exclusions.csv below) rather than a single
undifferentiated "error" bucket, so every entry in the paper's reported
counts can be traced back to a specific, named cause.

Usage
-----
python scripts/preprocess_sabdab.py \
    --summary    <raw_data>/sabdab_summary_all.tsv \
    --struct_dir <raw_data>/all_structures \
    --out_dir    <processed_dir> \
    --n_workers  8

Output
------
<out_dir>/*.pt              one file per retained antibody chain-pair
<out_dir>/funnel_report.json  stage-by-stage counts with justifications
<out_dir>/exclusions.csv      one row per excluded entry: pdb_id, stage,
                               status, species, method, requested_antigen
"""

import os
import csv
import json
import argparse
import warnings
from pathlib import Path
from typing import Optional
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import numpy as np

try:
    from Bio import PDB
    from Bio.PDB import PDBParser, NeighborSearch
    from Bio.PDB.Polypeptide import is_aa
except ImportError:
    raise ImportError("pip install biopython")

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMGT_CDR_H = {
    "H1": (27, 38.9),
    "H2": (56, 65.9),
    "H3": (105, 117.9),
}
IMGT_CDR_L = {
    "L1": (27, 38.9),
    "L2": (56, 65.9),
    "L3": (105, 117.9),
}

AA_MAP = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

ANTIGEN_EXCLUDED_RESNAMES = {
    "HOH", "WAT", "DOD",
    "NA", "CL", "MG", "ZN", "K", "MN", "FE", "CU", "CO", "NI", "CD",
    "HG", "BA", "CS", "LI", "SR", "CA",
    "SO4", "PO4", "GOL", "EDO", "PEG", "PG4", "ACT", "TRS", "MPD",
    "BME", "DTT", "IMD", "FMT", "EPE", "NH4", "ACY", "UNK", "UNL",
    "1PE", "MES", "BOG", "CIT", "TAR", "OXL",
}

# DNA residues use a D-prefix (DA/DC/DG/DT); RNA residues have no prefix
# (A/C/G/U). Used only for antigen-sequence translation -- antibody chain
# translation (AA_MAP alone) is unaffected, since antibody chains are
# always protein.
NUCLEOTIDE_MAP = {
    "DA": "A", "DC": "C", "DG": "G", "DT": "T", "DU": "U",
    "A": "A", "C": "C", "G": "G", "U": "U", "I": "I",
}


def antigen_residue_letter(resname: str) -> str:
    """Translates an antigen residue to a one-letter code where a standard
    one exists (amino acid or nucleotide); any other non-standard residue
    (e.g. a monosaccharide) is retained as its bracketed three-letter PDB
    name rather than forced into a misleading amino-acid code (the old
    AA_MAP.get(..., "X") fallback silently mapped every non-standard
    residue, nucleotide or otherwise, to "X", losing all chemical
    identity)."""
    resname = resname.strip()
    if resname in AA_MAP:
        return AA_MAP[resname]
    if resname in NUCLEOTIDE_MAP:
        return NUCLEOTIDE_MAP[resname]
    return f"[{resname}]"


INTERFACE_CUTOFF = 10.0
FV_MAX_IMGT = 128.99
H3_MIN, H3_MAX = 3, 35   # matches the OAS-side filter used in RQ2 (Section 3.1)
SAFETY_LENGTH_CAP = 280  # should never trigger post-Fv-filtering; logged if it does

# Legacy PDB format's atom serial number field is 5 fixed columns (7-11),
# capping at 99,999. Files with more atoms than this overflow the field and
# corrupt fixed-column parsing for every line after the overflow point.
# Confirmed as the real cause of a "invalid literal for int()" ValueError
# for all 6 affected PDB entries in this dataset (see process_entry below).
ATOM_SERIAL_OVERFLOW_THRESHOLD = 99_999


def _count_atom_lines(pdb_file: str) -> int:
    """Count ATOM/HETATM records directly, independent of whether the file
    parses -- used to corroborate the atom_serial_overflow diagnosis rather
    than relying on the exception message text alone."""
    n = 0
    with open(pdb_file) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                n += 1
    return n

# It exists
# purely so an unusually large antigen gets logged and counted for a manual
# look (e.g. a mis-parsed symmetry-expanded assembly would be worth
# inspecting), never so it gets silently dropped or silently kept without
# anyone knowing it was unusual.
ANTIGEN_LENGTH_MONITOR_THRESHOLD = 5000

_NULL_TOKENS = {"", "none", "n/a", "na", "nan", "unknown", "null"}


def _norm_target(s):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    sl = s.lower()
    if sl in _NULL_TOKENS:
        return None
    parts = [p.strip() for p in sl.split("|") if p.strip()]
    return parts[0] if parts else None


# ---------------------------------------------------------------------------
# PDB parsing (unchanged logic from the original script -- IMGT extraction
# itself was not the source of the earlier discipline gap, so it is not
# being rewritten; only the accounting around it changes)
# ---------------------------------------------------------------------------
def _residue_aa(res) -> Optional[str]:
    if not is_aa(res, standard=True):
        return None
    return AA_MAP.get(res.get_resname().strip(), None)


def _ca_coords(res) -> Optional[np.ndarray]:
    try:
        return np.array(res["CA"].get_vector().get_array(), dtype=np.float32)
    except KeyError:
        return None


def get_fv_residues(chain):
    residues = [r for r in chain.get_residues() if is_aa(r, standard=True)]
    residues.sort(key=lambda r: (r.get_id()[1], r.get_id()[2]))
    return [r for r in residues if r.get_id()[1] <= 128]


def parse_chain(chain, is_heavy: bool):
    residues = get_fv_residues(chain)
    if not residues:
        return None, None, None, None, None

    seq, coords, coord_mask, imgt_nums = [], [], [], []
    for res in residues:
        aa = _residue_aa(res)
        if aa is None:
            continue
        seq.append(aa)
        ca = _ca_coords(res)
        if ca is not None:
            coords.append(ca)
            coord_mask.append(True)
        else:
            coords.append(np.zeros(3, dtype=np.float32))
            coord_mask.append(False)
        res_id = res.get_id()
        pos = res_id[1]
        icode = res_id[2].strip()
        if icode:
            pos = pos + (ord(icode.upper()) - ord('A') + 1) * 0.1
        imgt_nums.append(pos)

    seq = "".join(seq)
    coords = np.stack(coords)
    coord_mask = np.array(coord_mask)
    imgt_nums = np.array(imgt_nums)

    cdr_ranges = IMGT_CDR_H if is_heavy else IMGT_CDR_L
    cdr_mask = np.zeros(len(seq), dtype=bool)
    for (start, end) in cdr_ranges.values():
        cdr_mask |= (imgt_nums >= start) & (imgt_nums <= end)

    assert len(seq) == len(imgt_nums)

    fv_mask = imgt_nums <= FV_MAX_IMGT
    seq = "".join([seq[i] for i in range(len(seq)) if fv_mask[i]])
    coords = coords[fv_mask]
    coord_mask = coord_mask[fv_mask]
    cdr_mask = cdr_mask[fv_mask]
    imgt_nums = imgt_nums[fv_mask]

    if len(seq) < 10:
        return None, None, None, None, None

    return seq, coords, coord_mask, imgt_nums, cdr_mask


def compute_interface_mask(ab_residues, ag_residues, cutoff=INTERFACE_CUTOFF):
    if not ag_residues:
        return np.zeros(len(ab_residues), dtype=bool)
    ag_atoms = [a for res in ag_residues for a in res.get_atoms()]
    if not ag_atoms:
        return np.zeros(len(ab_residues), dtype=bool)
    ns = NeighborSearch(ag_atoms)
    mask = np.zeros(len(ab_residues), dtype=bool)
    for i, res in enumerate(ab_residues):
        for atom in res.get_atoms():
            if ns.search(atom.get_vector().get_array(), cutoff, "A"):
                mask[i] = True
                break
    return mask


# ---------------------------------------------------------------------------
# Per-entry processing -- every return path carries a `stage` tag used to
# build the funnel report. `meta` carries the lightweight fields needed to
# later cross-tab exclusions against species/method/antigen.
# ---------------------------------------------------------------------------
def process_entry(pdb_id, h_chain, l_chain, ag_chain, model_id, antigen_name,
                   heavy_species, light_species, method, struct_dir, out_dir):
    meta = {
        "pdb_id": pdb_id, "heavy_species": heavy_species,
        "light_species": light_species, "method": method,
        "requested_antigen": bool(ag_chain and ag_chain.upper() != "NA"),
    }

    pdb_id_l = pdb_id.lower()
    pdb_file = os.path.join(struct_dir, "imgt", f"{pdb_id_l}.pdb")
    if not os.path.isfile(pdb_file):
        return {"stage": "missing_pdb", **meta}

    safe_pdb, safe_h, safe_l = pdb_id_l.replace("|", "_"), h_chain.replace("|", "_"), l_chain.replace("|", "_")
    safe_ag, safe_model = ag_chain.replace("|", "_"), str(model_id).replace("|", "_")
    out_name = f"{safe_pdb}_{safe_h}{safe_l}_ag{safe_ag}_m{safe_model}.pt"
    out_path = os.path.join(out_dir, out_name)
    if os.path.exists(out_path):
        return {"stage": "cached", **meta}

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id_l, pdb_file)
        models = list(structure.get_models())
    except ValueError as e:
        # CORRECTED (real bug, found by tracing an actual traceback -- not
        # assumed): a prior version of this code, and its own comment
        # below, assumed any "invalid literal for int()" failure at this
        # point meant a non-numeric `model` field in the summary TSV, and
        # relied entirely on the SEPARATE int(model_id) check further down
        # to catch and label that case. That was never verified against a
        # real traceback. In every one of the 6 PDB entries this project
        # has actually observed with this exact error text, the exception
        # fires HERE -- inside BioPython's own fixed-column coordinate
        # parser (PDBParser._parse_coordinates, resSeq field at
        # line[22:26]) -- and never reaches the int(model_id) line below
        # at all. Root cause: these files' total atom count exceeds the
        # legacy PDB format's 99,999-atom serial-number field (columns
        # 7-11); once the counter overflows, every fixed-width field is
        # shifted for the rest of the file, and a chain-ID letter lands in
        # the resSeq column purely by coincidence of position -- producing
        # the same ValueError text a non-numeric model field would, for an
        # entirely unrelated reason. Detected here by checking the file's
        # own atom-record count rather than trusting the exception message
        # alone (message text is corroborating, not sufficient by itself).
        n_atom_lines = _count_atom_lines(pdb_file)
        if "invalid literal for int()" in str(e) and n_atom_lines > ATOM_SERIAL_OVERFLOW_THRESHOLD:
            return {"stage": "atom_serial_overflow", "detail": str(e),
                    "n_atom_lines": n_atom_lines, **meta}
        return {"stage": "parse_error", "detail": str(e), **meta}
    except Exception as e:
        return {"stage": "parse_error", "detail": str(e), **meta}

    # The summary table's "model" column is occasionally a letter rather
    # than an integer index. If this branch is ever actually reached, it is
    # a GENUINELY non-numeric model selector on an otherwise-parseable
    # structure -- distinct from the atom_serial_overflow case above, which
    # is caught earlier and never reaches here (confirmed: every real
    # instance seen in this dataset was atom_serial_overflow, not this).
    # Retained as its own labeled path rather than removed, in case a
    # genuinely non-numeric model value on a parseable structure turns up
    # in a future SAbDab release. We do not guess a fallback model index
    # (e.g. defaulting to 0), consistent with this project's general rule
    # against silently resolving an ambiguous indexing convention --
    # excluded and reported distinctly instead.
    try:
        model_idx = int(model_id)
    except ValueError:
        return {"stage": "non_numeric_model_id", "detail": f"model_id={model_id!r}", **meta}
    model = models[min(model_idx, len(models) - 1)]

    if h_chain not in model:
        return {"stage": "missing_H_chain", **meta}
    h_seq, h_coords, h_coord_mask, h_imgt, h_cdr = parse_chain(model[h_chain], is_heavy=True)
    if h_seq is None or len(h_seq) < 10:
        return {"stage": "short_H", **meta}

    l_seq, l_coords, l_coord_mask, l_cdr = "", None, None, None
    has_light = l_chain != "NA" and l_chain in model
    if has_light:
        l_seq, l_coords, l_coord_mask, l_imgt, l_cdr = parse_chain(model[l_chain], is_heavy=False)
        if l_seq is None or len(l_seq) < 10:
            has_light = False

    if has_light:
        seq = h_seq + l_seq
        coords = np.concatenate([h_coords, l_coords], axis=0)
        coord_mask = np.concatenate([h_coord_mask, l_coord_mask], axis=0)
        cdr_mask = np.concatenate([h_cdr, l_cdr], axis=0)
    else:
        seq, coords, coord_mask, cdr_mask = h_seq, h_coords, h_coord_mask, h_cdr

    safety_triggered = len(seq) > SAFETY_LENGTH_CAP
    if safety_triggered:
        seq, coords, coord_mask, cdr_mask = seq[:SAFETY_LENGTH_CAP], coords[:SAFETY_LENGTH_CAP], \
            coord_mask[:SAFETY_LENGTH_CAP], cdr_mask[:SAFETY_LENGTH_CAP]

    ab_chain_ids = {str(h_chain).strip().upper()}
    if has_light:
        ab_chain_ids.add(str(l_chain).strip().upper())

    ag_chains = []
    if ag_chain and ag_chain.upper() != "NA":
        requested_ag = {x.strip().upper() for x in ag_chain.split("|") if x.strip()} - ab_chain_ids
        if requested_ag:
            ag_chains = [c for c in model.get_chains() if str(c.id).strip().upper() in requested_ag]

    ag_residues = [r for c in ag_chains for r in c.get_residues()
                   if r.get_resname().strip() not in ANTIGEN_EXCLUDED_RESNAMES]
    has_antigen = len(ag_residues) > 0

    antigen_coords = antigen_mask = antigen_seq = None
    # antigen_length_flagged is monitor-only:
    # it never changes what gets stored, only whether this entry is logged
    # for a manual look at unusually large antigens.
    antigen_length_flagged = len(ag_residues) > ANTIGEN_LENGTH_MONITOR_THRESHOLD
    if has_antigen:
        ag_coords_list, ag_mask_list = [], []
        for res in ag_residues:
            ca = _ca_coords(res)
            if ca is not None:
                ag_coords_list.append(ca); ag_mask_list.append(True)
            else:
                ag_coords_list.append(np.zeros(3, dtype=np.float32)); ag_mask_list.append(False)
        antigen_coords = torch.tensor(np.stack(ag_coords_list), dtype=torch.float32)
        antigen_mask = torch.tensor(ag_mask_list, dtype=torch.bool)
        antigen_seq = "".join(antigen_residue_letter(r.get_resname()) for r in ag_residues)

    interface_mask = None
    if has_antigen:
        ab_residues_list = []
        for chain_id in ([h_chain] + ([l_chain] if has_light else [])):
            if chain_id in model:
                ab_residues_list.extend(get_fv_residues(model[chain_id]))
        # this 400 is a *different*, unrelated cap on antibody Fv
        # residues (max possible length ~260 given both chains are already
        # IMGT-filtered to <=128 each) -- coincidentally the same literal as
        # the old antigen bug, cannot trigger in practice, left untouched.
        ab_residues_list = ab_residues_list[:400]
        # Interface computation already used the FULL ag_residues
        iface_np = compute_interface_mask(ab_residues_list, ag_residues)[:len(seq)]
        interface_mask = torch.tensor(iface_np, dtype=torch.bool)

    if coord_mask.sum() < 3:
        return {"stage": "too_few_coords", **meta}
    if cdr_mask.sum() == 0:
        return {"stage": "no_cdr_positions", **meta}

    # Locate CDR-H3 as the 3rd contiguous IMGT-CDR block within the heavy
    # prefix (H1, H2, H3 in order) to get its length for the [3, 35] filter.
    if h_cdr.sum() > 0:
        padded = np.concatenate(([0], h_cdr.astype(np.uint8), [0]))
        diff = np.diff(padded.astype(np.int8))
        starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
        if len(starts) >= 3:
            h3_len = int(ends[2] - starts[2])
        elif len(starts) > 0:
            h3_len = int(ends[-1] - starts[-1])
        else:
            h3_len = 0
    else:
        h3_len = 0

    if h3_len < H3_MIN:
        return {"stage": f"h3_too_short_{h3_len}", "h3_len": h3_len, **meta}
    if h3_len > H3_MAX:
        return {"stage": f"h3_too_long_{h3_len}", "h3_len": h3_len, **meta}

    sample = {
        "source": "SAbDab", "pdb_id": pdb_id_l, "paired": has_light,
        "target": antigen_name,
        "heavy": {"sequence_aa": h_seq}, "light": {"sequence_aa": l_seq if has_light else ""},
        "coords": torch.tensor(coords, dtype=torch.float32),
        "coord_mask": torch.tensor(coord_mask, dtype=torch.bool),
        "antigen_coords": antigen_coords, "antigen_mask": antigen_mask,
        "antigen_seq": antigen_seq if has_antigen else "",
        "cdr_mask": torch.tensor(cdr_mask, dtype=torch.bool),
        "cdr3_len_actual": h3_len,
        "interface_mask": interface_mask,
        "has_antigen": has_antigen,
        "antigen_cluster_id": None,  # filled downstream by the antigen-clustering script
    }
    torch.save(sample, out_path)
    return {"stage": "ok", "safety_cap_triggered": safety_triggered,
            "antigen_length_flagged": antigen_length_flagged, **meta}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
STAGE_JUSTIFICATIONS = {
    "ok": "Retained: parsed successfully, passed all structural and CDR-H3-length checks.",
    "missing_pdb": "No local structure file for this PDB ID -- a download-availability gap, not a modeling choice.",
    "cached": "Output file already exists from a prior run (not an exclusion).",
    "parse_error": "BioPython could not parse the structure file.",
    "non_numeric_model_id": "The summary table's model field is a letter rather than an integer index, on an otherwise-parseable structure file (distinct from atom_serial_overflow below, where the structure file itself fails to parse). No fallback model index is guessed, consistent with this project's rule against silently resolving an ambiguous indexing convention. As of this pipeline version, every real occurrence of this literal error text observed in this dataset was actually atom_serial_overflow, not this -- this stage is retained for a genuinely non-numeric model value on a parseable structure, which has not yet been observed but could occur in a future SAbDab release.",
    "atom_serial_overflow": "The deposited structure file's total atom count exceeds the legacy PDB format's 99,999-atom serial-number field (columns 7-11), which corrupts fixed-column parsing for the remainder of the file once the counter overflows (traceable to BioPython's PDBParser._parse_coordinates, resSeq field). Not a property of the summary table's model field, which only coincidentally shares matching literal characters in the resulting error text. Two mechanisms observed: large NMR ensembles (many stacked conformer models) and large single-model assemblies with many antibody-antigen chain-pair copies packed into one asymmetric unit. Not recoverable without a separate mmCIF-based ingestion path (mmCIF has no atom-count ceiling); currently excluded rather than silently mis-parsed.",
    "missing_H_chain": "The summary table names a heavy chain not present in this deposited model; heavy_species is also blank at the source for these rows, consistent with partial/light-chain-only depositions rather than a parsing failure.",
    "short_H": "Heavy chain shorter than 10 residues after Fv (IMGT<=128) extraction -- not a usable variable domain.",
    "too_few_coords": "Fewer than 3 resolved C-alpha atoms -- insufficient for downstream structural analysis (e.g. backbone RMSD, Section 4.2.2).",
    "no_cdr_positions": "Zero IMGT-CDR-range positions found -- typically indicates non-standard/failed IMGT annotation for this chain.",
    "h3_too_long": "H3 length outside the [3, 35] filter used to match the RQ2 comparator (Methods 3.1) -- empirically dominated by Bos taurus (cattle) entries, whose ultralong CDR-H3 loops (commonly 40-60+ residues, a distinct genetic mechanism specific to that species) fall outside this range by construction, not by data-quality failure.",
    "h3_too_short": "H3 length outside the [3, 35] filter used to match the RQ2 comparator (Methods 3.1).",
    "unhandled_exception": "An exception not caught by any of the specific handlers above -- inspect exclusions.csv 'detail' column for this stage before trusting the funnel totals.",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--struct_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--max_entries", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    struct_dir = args.struct_dir
    if os.path.isdir(os.path.join(struct_dir, "sabdab_dataset")):
        struct_dir = os.path.join(struct_dir, "sabdab_dataset")

    entries = []
    with open(args.summary) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pdb = row.get("pdb", "").strip().lower()
            h = row.get("Hchain", "").strip()
            if not pdb or not h:
                continue
            entries.append((
                pdb, h, row.get("Lchain", "NA").strip(),
                row.get("antigen_chain", "").strip(), row.get("model", "0").strip(),
                _norm_target(row.get("antigen_name", "")),
                _norm_target(row.get("heavy_species", "")),
                _norm_target(row.get("light_species", "")),
                _norm_target(row.get("method", "")),
            ))

    n_summary_rows = len(entries)
    if args.max_entries:
        entries = entries[:args.max_entries]

    print(f"[preprocess] {n_summary_rows} rows in summary table"
          f"{' (capped to ' + str(len(entries)) + ' for this run)' if args.max_entries else ''}")
    print(f"[preprocess] struct_dir={struct_dir}  out_dir={args.out_dir}  n_workers={args.n_workers}")

    stage_counts = Counter()
    fv_safety_cap_triggers = 0
    antigen_length_flags = 0
    exclusions_path = os.path.join(args.out_dir, "exclusions.csv")
    exclusion_fields = ["pdb_id", "stage", "heavy_species", "light_species", "method", "requested_antigen", "detail"]

    with open(exclusions_path, "w", newline="") as exc_f:
        exc_writer = csv.DictWriter(exc_f, fieldnames=exclusion_fields, extrasaction="ignore")
        exc_writer.writeheader()

        with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
            futures = {
                ex.submit(process_entry, pdb, h, l, ag, model, antigen_name,
                          hs, ls, method, struct_dir, args.out_dir): pdb
                for pdb, h, l, ag, model, antigen_name, hs, ls, method in entries
            }
            for i, fut in enumerate(as_completed(futures)):
                try:
                    result = fut.result()
                except Exception as e:
                    result = {"stage": "unhandled_exception", "detail": str(e), "pdb_id": futures[fut]}

                stage = result["stage"]
                # collapse h3_too_short_N / h3_too_long_N into their family for the
                # top-level funnel, but keep the exact length in the exclusions CSV
                funnel_stage = "h3_too_short" if stage.startswith("h3_too_short_") else \
                    "h3_too_long" if stage.startswith("h3_too_long_") else stage
                stage_counts[funnel_stage] += 1
                if result.get("safety_cap_triggered"):
                    fv_safety_cap_triggers += 1
                if result.get("antigen_length_flagged"):
                    antigen_length_flags += 1

                if funnel_stage not in ("ok", "cached"):
                    exc_writer.writerow({**result, "stage": stage})

                if (i + 1) % 1000 == 0:
                    print(f"  [{i+1}/{len(entries)}] ok={stage_counts['ok']} "
                          f"cached={stage_counts['cached']} excluded={sum(v for k, v in stage_counts.items() if k not in ('ok', 'cached'))}")

    n_retained = stage_counts["ok"] + stage_counts["cached"]
    funnel_report = {
        "n_summary_rows": n_summary_rows,
        "n_retained": n_retained,
        "n_excluded_total": n_summary_rows - n_retained,
        "stages": {
            stage: {"count": count, "justification": STAGE_JUSTIFICATIONS.get(
                stage, f"UNRECOGNIZED STAGE '{stage}' -- not in STAGE_JUSTIFICATIONS; investigate before "
                       f"trusting this count or writing anything about it.")}
            for stage, count in sorted(stage_counts.items(), key=lambda kv: -kv[1])
        },
        "fv_safety_length_cap_triggers": fv_safety_cap_triggers,
        "antigen_length_monitor_flags": antigen_length_flags,
        "antigen_length_monitor_note": "Monitor-only -- these entries are NOT truncated or excluded, "
                                        f"just flagged for a manual look (antigen > {ANTIGEN_LENGTH_MONITOR_THRESHOLD} residues).",
        "note": "Cross-tabulate exclusions.csv against heavy_species/method before writing anything "
                "in the paper about whether any exclusion reason is selective rather than incidental.",
    }
    with open(os.path.join(args.out_dir, "funnel_report.json"), "w") as f:
        json.dump(funnel_report, f, indent=2)

    print("\n[preprocess] Done.")
    print(json.dumps(funnel_report, indent=2))
    print(f"\n[preprocess] Exclusion detail written to {exclusions_path}")
    print(f"[preprocess] Funnel summary written to {os.path.join(args.out_dir, 'funnel_report.json')}")
    if fv_safety_cap_triggers:
        print(f"[preprocess] WARNING: Fv safety length cap ({SAFETY_LENGTH_CAP}) triggered on "
              f"{fv_safety_cap_triggers} entries -- investigate before trusting their CDR-H3 lengths.")
    if antigen_length_flags:
        print(f"[preprocess] NOTE: {antigen_length_flags} entries have an antigen longer than "
              f"{ANTIGEN_LENGTH_MONITOR_THRESHOLD} residues (monitor-only -- not truncated, "
              f"nothing was excluded; worth a manual look at exclusions.csv-adjacent metadata "
              f"if that count is surprisingly high).")


if __name__ == "__main__":
    main()
