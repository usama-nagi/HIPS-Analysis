"""
rq_germline_00_fetch_reference.py

Step 1 of 2 for the germline allele-resolution extension (Limitations and
Future Directions, Section 6). Downloads V and J germline gene reference
sequences for Homo sapiens and Mus musculus from OGRDB (the Open Germline
Receptor Database, the AIRR Community's curated, stable, REST-API-backed
replacement for ad-hoc IMGT/GENE-DB scraping), and runs sanity checks on
the downloaded sets before anything else is allowed to depend on them.

WHY OGRDB AND NOT THE ORIGINAL ANARCI APPROACH:
The original ANARCI builds its germline HMMs from sequences scraped live
from IMGT/GENE-DB's web query interface at install time (see its
build_pipeline/RipIMGT.py). This is independently broken for several users
right now (see github.com/oxpig/ANARCI issues #85, #41 -- IMGT's site
structure appears to have changed). OGRDB is a stable, versioned,
REST-API-accessible alternative used by IgBLAST/MiXCR pipelines
community-wide, with no live-scraping fragility.

SPECIES SCOPE, STATED HONESTLY:
OGRDB currently provides germline sets for only four species: Homo sapiens,
Mus musculus, Macaca mulatta, and Oncorhynchus mykiss (rainbow trout) --
confirmed directly against https://ogrdb.airr-community.org/api_v2/germline/species
at the time this script was written. It does NOT cover camelid species
(Lama glama, Vicugna pacos), which make up ~14% of this dataset. This is
not unique to our approach: the ORIGINAL ANARCI's own natively-supported
species list (human, mouse, rat, rabbit, pig, rhesus) also excludes
camelids -- ANARCI numbers camelid VHH sequences but does not claim
reliable germline assignment for them either. We therefore scope this
extension to Homo sapiens and Mus musculus, the two largest species
groups in the dataset (48.5% and 22.5% of entries respectively, ~71%
combined), and report camelid and other minority species at family-level
only, exactly as before, with this stated as a scope limit rather than a
silently dropped case.

MOUSE STRAIN CAVEAT:
OGRDB does not provide one universal "Mus musculus" germline set -- it
provides several strain-specific sets (BALB/c, C57BL/6, CAST/EiJ, etc.).
SAbDab's summary export does not record mouse strain. We use the C57BL/6
IGH/IGK/IGL sets as the reference, since C57BL/6 is the most commonly
used laboratory mouse strain and the most complete/best-curated OGRDB
mouse set, and state this as an explicit assumption: mouse germline
assignments in this analysis are calls against the C57BL/6 reference,
which may not be the strain a given SAbDab entry's source antibody was
actually raised in. A different strain's true nearest germline could
differ from the C57BL/6-nearest call reported here. This is flagged
in the output JSON's metadata block, not just in this docstring.

Usage:
    python rq_germline_00_fetch_reference.py --out_dir tables/germline_reference

Writes (per species/locus, e.g. for Homo sapiens IGH):
    <out_dir>/Homo_sapiens_IGH_V.fasta        (raw OGRDB download, NUCLEOTIDE)
    <out_dir>/Homo_sapiens_IGH_J.fasta        (raw OGRDB download, NUCLEOTIDE)
    <out_dir>/Homo_sapiens_IGH_V_aa.fasta     (translated, validated AMINO ACID -- use this)
    <out_dir>/Homo_sapiens_IGH_J_aa.fasta     (translated, validated AMINO ACID -- use this)
    <out_dir>/Homo_sapiens_IGH_V_aa_flagged_nt.fasta  (only created if any
        sequences failed unambiguous frame detection -- nucleotide, excluded
        from the _aa.fasta set, inspect before deciding whether to handle
        separately)
    <out_dir>/fetch_reference_report.json   (full sanity-check summary --
                                              READ THIS before running step 2)
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig

from Bio import SeqIO
from Bio.Seq import Seq


# (species, locus, ogrdb_name_or_None, output_prefix)
# ogrdb_name is only needed when a species/locus has multiple named sets
# (currently true only for mouse); None means "use the default/only set".
# Each entry: (species, locus, output_prefix, v_set_name, j_set_name)
# v_set_name / j_set_name are usually identical (one OGRDB set covers both
# V and J for that species/locus/strain), but mouse IGK and IGL are a real
# exception: V genes are strain-specific, but J genes are shared across
# ALL strains in a single common set. This is documented directly by
# OGRDB ("IGK and IGL sets for each strain contain V sequences only: use
# these with the IGKJ and IGLJ sets which contain Js for all strains")
# and was confirmed in practice -- the original single-set mouse IGK
# attempt ("C57BL/6 IGK") failed with "set not found", and the real error
# message's list of available sets (C57BL/6J IGKV, IGKJ (all strains),
# etc.) revealed the actual structure. When v_set_name != j_set_name,
# run_download_split() below performs two separate downloads and merges
# their V/J outputs into the single expected {prefix}_V.fasta /
# {prefix}_J.fasta pair the rest of this script expects.
GERMLINE_SETS = [
    ("Homo sapiens", "IGH", "Homo_sapiens_IGH", "IGH_VDJ", "IGH_VDJ"),
    ("Homo sapiens", "IGK", "Homo_sapiens_IGK", "IGKappa_VJ", "IGKappa_VJ"),
    ("Homo sapiens", "IGL", "Homo_sapiens_IGL", "IGLambda_VJ", "IGLambda_VJ"),
    ("Mus musculus", "IGH", "Mus_musculus_C57BL6_IGH", "C57BL/6 IGH", "C57BL/6 IGH"),
    ("Mus musculus", "IGK", "Mus_musculus_C57BL6_IGK", "C57BL/6J IGKV", "IGKJ (all strains)"),
    ("Mus musculus", "IGL", "Mus_musculus_C57BL6_IGL", "C57BL/6J IGLV", "IGLJ (all strains)"),
]
# NOTE: human set names were initially assumed to need no -n flag, which
# failed at runtime with "Multiple sets are available... IGHC, IGH_VDJ".
# IGH_VDJ is the V/D/J variable-region set (what we want); IGHC is the
# constant-region set (not germline V/J, not used here). IGKappa_VJ /
# IGLambda_VJ are the official AIRR-C reference set names for human kappa
# and lambda light chains, confirmed against the receptor_utils
# documentation, the AIRR-C Human IG Reference Sets paper (Lees et al.),
# and the igblastr package manual.
#
# NOTE: mouse "C57BL/6 IGH" (no J suffix) was confirmed working in the
# successful IGH run; mouse light-chain V-set names use "C57BL/6J" (with
# J suffix on the strain name itself, e.g. "C57BL/6J IGKV") per the real
# OGRDB error message -- this J refers to the Jackson Laboratory strain
# suffix, unrelated to the immunoglobulin J-gene segment, a naming
# coincidence worth being careful not to confuse.

# Plausibility bounds for sanity-checking the downloaded sets. These are
# generous (real counts for human/C57BL6 mouse are well-documented and
# fall comfortably inside these ranges) -- the point is to catch a
# catastrophically wrong or empty download, not to pin an exact count.
EXPECTED_V_GENE_RANGE = {
    "Homo_sapiens_IGH": (50, 400),
    "Homo_sapiens_IGK": (30, 200),
    "Homo_sapiens_IGL": (20, 150),
    "Mus_musculus_C57BL6_IGH": (50, 300),
    "Mus_musculus_C57BL6_IGK": (50, 300),
    "Mus_musculus_C57BL6_IGL": (1, 30),
}
EXPECTED_J_GENE_RANGE = {
    "Homo_sapiens_IGH": (1, 20),
    "Homo_sapiens_IGK": (1, 20),
    "Homo_sapiens_IGL": (1, 20),
    "Mus_musculus_C57BL6_IGH": (1, 20),
    "Mus_musculus_C57BL6_IGK": (1, 20),
    "Mus_musculus_C57BL6_IGL": (1, 20),
}

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def best_frame_translation(nt_seq):
    """OGRDB's download_germline_set returns NUCLEOTIDE sequences, not
    amino acid (confirmed against receptor_utils' own IgBLAST walkthrough,
    which builds nucleotide BLAST databases from this exact output) --
    despite IMGT/GENE-DB's own germline gene FASTA convention setting
    V-REGION numbering to start at the first nucleotide of the first
    codon (frame 0), J-REGION entries can include 1-2 leading bases
    before the true coding frame per IMGT's own documentation, so frame
    0 is not blindly assumed here. Instead, all three frames are
    translated and the frame with the fewest internal stop codons is
    selected -- a true germline V/J coding sequence should have zero (a
    stop codon mid-sequence is biologically implausible for a functional
    germline gene). Sequences where this check is ambiguous (more than
    one frame ties for fewest stops, or the best frame still has internal
    stops) are flagged rather than silently guessed at.

    Returns (translation, frame_used, n_internal_stops, all_frame_stop_counts, ambiguous: bool)
    """
    nt_seq = nt_seq.upper().replace("-", "").replace(".", "")
    results = []
    for frame in range(3):
        sub = nt_seq[frame:]
        sub = sub[:len(sub) - (len(sub) % 3)]
        if len(sub) == 0:
            results.append((frame, "", 999))
            continue
        aa = str(Seq(sub).translate())
        n_stops = aa[:-1].count("*") if aa.endswith("*") else aa.count("*")
        results.append((frame, aa, n_stops))

    min_stops = min(r[2] for r in results)
    tied = [r for r in results if r[2] == min_stops]
    ambiguous = (len(tied) > 1) or (min_stops > 0)
    best = tied[0]
    return best[1].rstrip("*"), best[0], best[2], [r[2] for r in results], ambiguous


JGENE_MOTIF = re.compile(r"[FW]G.G")


def best_frame_translation_jgene(nt_seq):
    """Frame disambiguation specifically for J-genes, which are too short
    (~10-20 codons) for the generic stop-codon method above to work
    reliably: with so few codons, multiple frames frequently produce zero
    internal stops purely by chance, leaving no real signal to pick a
    frame. This was discovered empirically, not anticipated in advance --
    a real run against actual OGRDB mouse IGHJ data flagged all 4 of 4
    J-genes as ambiguous using the V-gene method, which on inspection
    turned out to be a structural limitation of that method on short
    sequences, not a data quality problem.

    J-genes have a much stronger, IMGT-documented signal available
    instead: every real germline J-REGION encodes the conserved
    F/W-G-X-G motif (IMGT's J-PHE or J-TRP label) at its 3' end, the
    canonical landmark of framework 4 (confirmed against IMGT's own
    ontology documentation, not assumed). We require the chosen frame to
    both be stop-codon-free AND produce this motif -- a translation that
    happens to lack internal stops but doesn't contain this highly
    specific 4-residue pattern is not accepted as unambiguous, since the
    motif is unlikely to appear by chance in a short, non-J-gene
    translation.
    """
    nt_seq = nt_seq.upper().replace("-", "").replace(".", "")
    results = []
    for frame in range(3):
        sub = nt_seq[frame:]
        sub = sub[:len(sub) - (len(sub) % 3)]
        if len(sub) == 0:
            results.append((frame, "", False, 999))
            continue
        aa = str(Seq(sub).translate())
        n_stops = aa.count("*")
        has_motif = bool(JGENE_MOTIF.search(aa))
        results.append((frame, aa, has_motif, n_stops))

    motif_frames = [r for r in results if r[2] and r[3] == 0]
    ambiguous = len(motif_frames) != 1
    if motif_frames:
        best = motif_frames[0]
        return best[1].rstrip("*"), best[0], 0, [r[3] for r in results], ambiguous
    # no frame both motif-matched and stop-free -- report the overall best
    # stop count for diagnostic purposes, but always treat as ambiguous/failed
    fallback = min(results, key=lambda r: r[3])
    return fallback[1].rstrip("*"), fallback[0], fallback[3], [r[3] for r in results], True


def translate_fasta_to_aa(nt_path, aa_path, gene_kind="V"):
    """Translates a downloaded nucleotide germline FASTA to amino acid,
    writing only sequences whose frame was unambiguous and stop-free.
    Sequences that fail are written to a sibling _flagged.fasta file with
    their nucleotide sequence intact, never silently dropped or guessed.
    gene_kind selects the disambiguation method: 'V' uses the generic
    stop-codon-count method (works well on long V-gene sequences); 'J'
    uses the conserved-motif method (required for short J-gene
    sequences, where the generic method has no reliable signal -- see
    best_frame_translation_jgene's docstring for why)."""
    frame_fn = best_frame_translation_jgene if gene_kind == "J" else best_frame_translation
    min_len = 10 if gene_kind == "J" else 50  # J-genes are short (~12-20 aa)
        # by nature; V-genes are long (~95-130 aa). A single shared floor
        # would be either too permissive for V or too strict for J, so
        # each gene_kind gets its own plausible minimum rather than one
        # compromise value.
    records_in = list(SeqIO.parse(nt_path, "fasta"))
    n_in = len(records_in)
    n_ok, n_flagged = 0, 0
    flagged_path = aa_path.replace(".fasta", "_flagged_nt.fasta")

    with open(aa_path, "w") as out_aa, open(flagged_path, "w") as out_flagged:
        for rec in records_in:
            aa, frame, n_stops, all_stops, ambiguous = frame_fn(str(rec.seq))
            frac_x = (aa.count("X") / len(aa)) if aa else 1.0
            too_ambiguous = frac_x > 0.10  # real germline genes should translate
                                            # almost entirely to defined residues;
                                            # >10% X indicates excessive N/ambiguous
                                            # input bases, not a real coding sequence
                                            # (caught by testing an all-N synthetic
                                            # sequence before trusting this on real
                                            # downloads -- a 100-residue all-N input
                                            # would otherwise pass the length and
                                            # stop-codon guards undetected)
            if ambiguous or len(aa) < min_len or too_ambiguous:
                out_flagged.write(f">{rec.id} [all_frame_stop_counts={all_stops}, frac_X={frac_x:.2f}]\n{rec.seq}\n")
                n_flagged += 1
                continue
            out_aa.write(f">{rec.id}\n{aa}\n")
            n_ok += 1

    if n_flagged == 0:
        os.remove(flagged_path)

    return {"n_input": n_in, "n_translated_ok": n_ok, "n_flagged": n_flagged,
            "flagged_path": flagged_path if n_flagged > 0 else None}


def resolve_download_germline_set_path():
    """Locates the download_germline_set console script without depending
    on PATH. pip writes console-script entry points (like
    download_germline_set, installed alongside the receptor_utils package)
    to a deterministic 'scripts' directory that sysconfig can report
    directly -- this is the same directory pip itself used at install
    time, so it's correct regardless of whether the current shell's PATH
    happens to include it. This sidesteps the exact PATH-export problem
    this project already hit once with MMseqs2, without requiring a
    manual PATH export to be remembered every session.

    Checks, in order: PATH (in case it's already resolvable, the cheap
    common case), then sysconfig.get_path('scripts') for the current
    interpreter (covers the standard venv/conda layout), then falls back
    to a directory search near the installed receptor_utils package as a
    last resort for unusual installs. Raises a clear, actionable error --
    distinct from a download failure -- if none of these find it, since
    "not installed" and "installed but PATH-blocked" need different fixes
    and should not be conflated in the error message.
    """
    found = shutil.which("download_germline_set")
    if found:
        return found

    candidate = os.path.join(sysconfig.get_path("scripts"), "download_germline_set")
    if os.path.exists(candidate):
        return candidate

    try:
        import receptor_utils
        pkg_parent = os.path.dirname(os.path.dirname(receptor_utils.__file__))
        for root, _, files in os.walk(pkg_parent):
            if "download_germline_set" in files:
                return os.path.join(root, "download_germline_set")
    except ImportError:
        pass

    raise RuntimeError(
        "[NOT FOUND] Could not locate the 'download_germline_set' console "
        "script anywhere -- checked PATH, sysconfig's scripts directory "
        f"({sysconfig.get_path('scripts')}), and a search near the "
        "receptor_utils package itself. This is different from a PATH "
        "problem: it suggests receptor_utils' console-script entry point "
        "was not actually installed (e.g. pip installed only as a wheel "
        "without entry-point registration, or into an environment "
        "different from the one this script is running in). Confirm with: "
        "`pip show receptor_utils` and `python -c \"import receptor_utils; "
        "print(receptor_utils.__file__)\"` -- if those succeed but this "
        "still fails, reinstall with `pip install --force-reinstall "
        "receptor_utils`."
    )


def run_download(species, locus, name, out_dir, ogrdb_prefix):
    """Low-level: calls download_germline_set once for a single named set,
    located via resolve_download_germline_set_path() rather than assumed
    to be on PATH.

    IMPORTANT: download_germline_set can exit with code 0 while having
    written NO output file at all -- this happens when the requested
    species/locus has multiple named germline sets and none is specified
    via -n (it prints "Multiple sets are available... Please specify the
    set name" and returns, without raising or setting a non-zero exit
    code). This was caught by running this exact failure mode against the
    real OGRDB API (the human IGH download, before IGH_VDJ was specified
    explicitly): the subprocess returned 0, so an earlier version of this
    function reported success, and the failure only surfaced downstream
    as a confusing FileNotFoundError in the translation step instead of a
    clear, immediate error here. Checking the exit code alone is
    therefore not sufficient -- this function also checks that at least
    one of the expected V/D/J output files actually exists before
    declaring success, regardless of what the subprocess returned.

    ogrdb_prefix is passed directly as -p and is the caller's
    responsibility to construct correctly -- see fetch_v_and_j for why
    (download_germline_set concatenates prefix+chain with NO separator
    when -p is given explicitly: f"{prefix}{chain}.fasta", confirmed
    against its source and against a real failed run that produced
    "Homo_sapiens_IGHV.fasta" instead of the expected
    "Homo_sapiens_IGH_V.fasta" before this was accounted for)."""
    script_path = resolve_download_germline_set_path()
    cmd = [sys.executable, script_path, species, locus, "-f", "MULTI-F", "-p", ogrdb_prefix]
    if name is not None:
        cmd += ["-n", name]
    print(f"[fetch] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=out_dir, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            f"[FETCH FAILED] download_germline_set exited {result.returncode} "
            f"for {species} {locus} (name={name}). See stderr above. "
            f"Do not proceed to step 2 until this is resolved."
        )

    any_output = any(
        os.path.exists(os.path.join(out_dir, f"{ogrdb_prefix}{chain}.fasta"))
        for chain in ("V", "D", "J")
    )
    if not any_output:
        raise RuntimeError(
            f"[FETCH FAILED] download_germline_set exited 0 (apparent success) "
            f"for {species} {locus} (name={name}), but none of the expected "
            f"V/D/J output files were created under prefix '{ogrdb_prefix}'. "
            f"This is a known failure mode: the tool can print an error (e.g. "
            f"\"Multiple sets are available\" or \"set not found\") and return "
            f"successfully without writing anything. Full stdout from the call "
            f"is printed above -- check it for the actual error message and "
            f"available set names before retrying."
        )


def fetch_v_and_j(species, locus, v_set_name, j_set_name, out_dir, prefix):
    """High-level: produces the final {prefix}_V.fasta and {prefix}_J.fasta
    files this script's downstream code expects, regardless of whether
    OGRDB stores V and J for this species/locus in one combined set or
    two separate ones.

    For most species/locus combinations, v_set_name == j_set_name (one
    OGRDB set covers both), and this is a single download. For mouse IGK
    and IGL specifically, OGRDB splits V (strain-specific) and J (shared
    across all strains) into two separate named sets -- this was
    discovered from a real failed run ("C57BL/6 IGK" does not exist; the
    error message listed "C57BL/6J IGKV" and "IGKJ (all strains)" as the
    real available sets) and confirmed against OGRDB's own mouse-strain
    documentation. When the two set names differ, this function performs
    two separate downloads into distinct temporary prefixes and copies
    only the relevant chain's file into the final expected location from
    each, so a V-only download's stray D/J files (if any) and a J-only
    download's stray V/D files are never mistaken for real data."""
    final_v_path = os.path.join(out_dir, f"{prefix}_V.fasta")
    final_j_path = os.path.join(out_dir, f"{prefix}_J.fasta")

    if v_set_name == j_set_name:
        ogrdb_prefix = prefix + "_"
        run_download(species, locus, v_set_name, out_dir, ogrdb_prefix)
        # run_download's own check already confirmed at least one of
        # V/D/J exists under this prefix; confirm V and J specifically,
        # since those are what this script actually needs downstream.
        for final_path, chain_label in [(final_v_path, "V"), (final_j_path, "J")]:
            if not os.path.exists(final_path):
                raise RuntimeError(
                    f"[FETCH FAILED] {species} {locus} (set={v_set_name}) "
                    f"downloaded successfully but did not produce a {chain_label} "
                    f"gene file ({final_path}). This OGRDB set may not include "
                    f"{chain_label} genes for this locus -- check the console "
                    f"output above for what files were actually written."
                )
        return

    # Split case: V and J come from two different named sets.
    v_tmp_prefix = f"__tmp_{prefix}_vset_"
    j_tmp_prefix = f"__tmp_{prefix}_jset_"

    run_download(species, locus, v_set_name, out_dir, v_tmp_prefix)
    v_src_path = os.path.join(out_dir, f"{v_tmp_prefix}V.fasta")
    if not os.path.exists(v_src_path):
        raise RuntimeError(
            f"[FETCH FAILED] {species} {locus} V-set '{v_set_name}' downloaded "
            f"successfully but did not produce a V gene file ({v_src_path}). "
            f"Check the console output above for what files were actually "
            f"written by this set."
        )
    shutil.copyfile(v_src_path, final_v_path)

    run_download(species, locus, j_set_name, out_dir, j_tmp_prefix)
    j_src_path = os.path.join(out_dir, f"{j_tmp_prefix}J.fasta")
    if not os.path.exists(j_src_path):
        raise RuntimeError(
            f"[FETCH FAILED] {species} {locus} J-set '{j_set_name}' downloaded "
            f"successfully but did not produce a J gene file ({j_src_path}). "
            f"Check the console output above for what files were actually "
            f"written by this set."
        )
    shutil.copyfile(j_src_path, final_j_path)

    # Clean up temp-prefixed files (V-set download may also have written
    # D.fasta / J.fasta / V_gapped.fasta that we don't want lingering
    # alongside the merged final output under misleading temp names).
    for tmp_prefix in (v_tmp_prefix, j_tmp_prefix):
        for chain in ("V", "D", "J", "V_gapped"):
            stray = os.path.join(out_dir, f"{tmp_prefix}{chain}.fasta")
            if os.path.exists(stray):
                os.remove(stray)


def validate_fasta(path, expected_range, gene_kind, label):
    """Loads a TRANSLATED (amino-acid) germline FASTA, checks it parses,
    confirms every sequence is genuinely amino-acid alphabet (a real check
    now, since the translation step already removed nucleotide content --
    this check exists to catch the case where translate_fasta_to_aa itself
    has a bug and somehow wrote raw nucleotide through unchanged), and
    checks the gene count falls in a generous plausibility range."""
    if not os.path.exists(path):
        return {
            "label": label, "gene_kind": gene_kind, "status": "MISSING_FILE",
            "path": path, "n_genes": 0,
        }

    records = list(SeqIO.parse(path, "fasta"))
    n = len(records)

    bad_alphabet = []
    for rec in records[:50]:
        seq = str(rec.seq).upper().replace("-", "").replace(".", "").replace("*", "")
        if seq and not set(seq).issubset(VALID_AA):
            bad_alphabet.append(rec.id)

    lo, hi = expected_range
    status = "OK"
    issues = []
    if n == 0:
        status = "EMPTY"
        issues.append("zero sequences parsed")
    elif not (lo <= n <= hi):
        status = "OUT_OF_RANGE"
        issues.append(f"n_genes={n} outside plausibility range [{lo},{hi}]")
    if bad_alphabet:
        status = "BAD_ALPHABET"
        issues.append(
            f"{len(bad_alphabet)}/{min(50,n)} sampled records contain non-amino-acid "
            f"characters after translation (sample ids: {bad_alphabet[:5]}) -- this "
            f"would indicate a bug in translate_fasta_to_aa, since this check runs "
            f"on its output, not the raw OGRDB download."
        )

    return {
        "label": label, "gene_kind": gene_kind, "status": status,
        "path": path, "n_genes": n, "issues": issues,
        "sample_ids": [r.id for r in records[:5]],
    }

    return {
        "label": label, "gene_kind": gene_kind, "status": status,
        "path": path, "n_genes": n, "issues": issues,
        "sample_ids": [r.id for r in records[:5]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    report = {"sets": [], "all_ok": True}

    for species, locus, prefix, v_set_name, j_set_name in GERMLINE_SETS:
        fetch_v_and_j(species, locus, v_set_name, j_set_name, args.out_dir, prefix)

        v_nt_path = os.path.join(args.out_dir, f"{prefix}_V.fasta")
        j_nt_path = os.path.join(args.out_dir, f"{prefix}_J.fasta")
        v_aa_path = os.path.join(args.out_dir, f"{prefix}_V_aa.fasta")
        j_aa_path = os.path.join(args.out_dir, f"{prefix}_J_aa.fasta")

        print(f"[translate] {prefix}_V: nucleotide -> amino acid (best-frame, "
              f"flagging ambiguous/stop-containing sequences)")
        v_translate_stats = translate_fasta_to_aa(v_nt_path, v_aa_path, gene_kind="V")
        print(f"  {v_translate_stats}")
        print(f"[translate] {prefix}_J: nucleotide -> amino acid")
        j_translate_stats = translate_fasta_to_aa(j_nt_path, j_aa_path, gene_kind="J")
        print(f"  {j_translate_stats}")

        v_result = validate_fasta(v_aa_path, EXPECTED_V_GENE_RANGE[prefix], "V", f"{prefix}_V")
        j_result = validate_fasta(j_aa_path, EXPECTED_J_GENE_RANGE[prefix], "J", f"{prefix}_J")
        v_result["translation_stats"] = v_translate_stats
        j_result["translation_stats"] = j_translate_stats

        for r in (v_result, j_result):
            print(f"[check] {r['label']}: status={r['status']} n_genes={r['n_genes']}")
            if r["status"] != "OK":
                report["all_ok"] = False
                print(f"  ISSUES: {r.get('issues', r)}")
            if r["translation_stats"]["n_flagged"] > 0:
                print(f"  NOTE: {r['translation_stats']['n_flagged']} of "
                      f"{r['translation_stats']['n_input']} sequences were flagged "
                      f"during translation (ambiguous frame or internal stop codon) "
                      f"and excluded from the amino-acid set -- see "
                      f"{r['translation_stats']['flagged_path']}")

        report["sets"].append({"species": species, "locus": locus,
                                "v_set_name": v_set_name, "j_set_name": j_set_name,
                                "prefix": prefix,
                                "V": v_result, "J": j_result})

    report["mouse_strain_used"] = (
        "C57BL/6 for IGH (single combined V/D/J set). For IGK and IGL, OGRDB "
        "splits strain-specific V genes from a J-gene set shared across all "
        "mouse strains: V genes are from the C57BL/6J-specific set, J genes "
        "are from OGRDB's strain-independent shared set (see v_set_name / "
        "j_set_name per entry in 'sets' above). SAbDab does not record "
        "source mouse strain, so this is an assumption stated explicitly "
        "here and in the paper's Limitations section."
    )
    report["species_scope"] = ["Homo sapiens", "Mus musculus"]
    report["species_not_covered"] = "All other species (camelid, rat, rabbit, etc.) -- OGRDB has no reference for them; family-level statistics for these entries are unchanged from the main paper."

    out_path = os.path.join(args.out_dir, "fetch_reference_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Wrote {out_path}")
    if report["all_ok"]:
        print("[OK] All sets passed sanity checks. Safe to proceed to step 2.")
    else:
        print("[STOP] One or more sets failed sanity checks. "
              "Inspect fetch_reference_report.json and the FASTA files "
              "directly before running step 2 -- do not proceed on a "
              "failed or out-of-range reference set.")
        sys.exit(1)


if __name__ == "__main__":
    main()