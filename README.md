# Hidden in Plain Sight - Reproducibility Package

This repository contains the full analysis pipeline behind *"Hidden in
Plain Sight: Quantifying Compositional Skews and CDR-H3 Redundancy in
SAbDab for Antibody Machine-Learning Benchmarks."* Every number, table,
and figure in the paper is produced by the scripts in `scripts/`, run in
the order below against a snapshot of SAbDab, Thera-SAbDab, and OAS.

This document is a run order derived from the actual read/write
dependencies between scripts, not from memory: a script's inputs are
either listed as another script's outputs below, or come from
`configs/config.yaml` (raw SAbDab/Thera-SAbDab/OAS paths).

---

## 1. Requirements

- Python 3.10+
- `torch`, `numpy`, `pandas`, `scipy`, `biopython`, `pyyaml`
- [MMseqs2](https://github.com/soedinglab/MMseqs2) on `PATH` (`conda install -c bioconda mmseqs2`)
- [ANARCI](https://github.com/oxpig/ANARCI) importable, with `hmmscan` on `PATH`
  (`apt-get install hmmer` on Debian/Ubuntu) - only required for
  `anarci_germline_validation.py`, `scope_chain_mislabel.py`, and the two
  ANARCI-based audits
- Raw data, obtained separately:
  - SAbDab summary table and structure files ([opig.stats.ox.ac.uk/webapps/newsabdab/sabdab](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/))
  - Thera-SAbDab summary table (same portal)
  - An OAS-derived mmap directory with per-sequence CDR3 length and
    source-split metadata (built from [Observed Antibody Space](https://opig.stats.ox.ac.uk/webapps/oas/))
  - For `audit_sabdab2_splits.py` only: SAbDab2's published `ab_split.csv`
    and `abag_split.csv` ([Zenodo record 20083995](https://zenodo.org/records/20083995))

Every script takes `--config configs/config.yaml`. Paths written below are
relative to `work_dir` in that config.

## 2. Configuration

Copy the template below to `configs/config.yaml` and fill in the four
real paths under `paths:`. The other values match what the paper reports;
change them only if you intend to reproduce a different setting.

```yaml
paths:
  work_dir: /path/to/your/working/directory        # tables/, figures/ written here
  sabdab_summary_tsv: /path/to/sabdab_summary_all.tsv
  sabdab_pt_dir: /path/to/preprocessed/pt_files     # output of preprocess_sabdab.py
  thera_sabdab_tsv: /path/to/thera_sabdab_summary.tsv
  oas_mmap_root: /path/to/oas_mmap_directory

filters:
  cdr3_min_len: 3
  cdr3_max_len: 35

mmseqs2:
  cdrh3_min_seq_id: 0.90
  cdrh3_coverage: 0.80
  n_threads: 8

sampling:
  oas_subsample_n_heavy: 2000000
  oas_subsample_n_paired: 2000000
  oas_subsample_seed: 42

scope_notes:
  oas_species_scope: "human-only; verified at the source-file level (verify_oas_species.py)"
  oas_germline_scope: "family-level only; no allele-resolved comparison against OAS"
  germline_resolution: "allele-resolved extension covers human and mouse only (OGRDB coverage)"
```

Antigen-cluster identity (70% identity / 80% coverage) is a separate,
one-off MMseqs2 run and is passed as command-line flags to
`assign_antigen_clusters.py` directly, not read from this file.

## 3. Directory layout

```
scripts/
  00_build_dataset.py                  Stage 1
  preprocess_sabdab.py                 Stage 0
  assign_antigen_clusters.py           Stage 0
  scope_chain_mislabel.py              Stage 2
  patch_apply_chain_mislabel_exclusions.py   Stage 2
  rq1_sequence_structural_bias.py      Stage 3
  migrate_master_csv.py                Stage 4
  rq_germline_00_fetch_reference.py    Stage 5
  rq_germline_01_assign.py             Stage 5
  anarci_germline_validation.py        Stage 5
  validate_antigen_classifier_sample.py    Stage 6
  validate_antigen_classifier_score.py     Stage 6
  rq2_oas_comparison.py                Stage 8
  rq2_composition_analysis.py          Stage 8
  rq3_redundancy_and_recommendations.py    Stage 9
  rq3_dedup_threshold_sensitivity.py   Stage 9
  make_figures.py                      Stage 10
  common.py                            shared utilities, imported everywhere
  validate_antigen_classification.py   deprecated, kept for reference (see Part 3)
  audits/                              Stage 7, supporting robustness checks
  anarci/                              vendored third-party ANARCI library
```

---

## Part 1 - Core reproducible pipeline

Run once, top to bottom, on a fresh checkout to go from raw data to every
number and figure in the paper.

### Stage 0 - Structural preprocessing (raw data → retained `.pt` files)
```
python scripts/preprocess_sabdab.py \
    --summary    <raw_data>/sabdab_summary_all.tsv \
    --struct_dir <raw_data>/all_structures \
    --out_dir    <processed_dir> \
    --n_workers  8
```
Produces: `<processed_dir>/*.pt`, `tables/funnel_report.json`,
`tables/exclusions.csv`. This is the 20,968 → 20,037 filter.

```
python scripts/assign_antigen_clusters.py \
    --sabdab_dir <processed_dir> \
    --out_dir    <processed_dir> \
    --min_seq_id 0.70 --coverage 0.80 --n_threads 16
```
Adds `antigen_cluster_id` to the `.pt` files written above.

### Stage 1 - Master table construction
```
python scripts/00_build_dataset.py --config configs/config.yaml
```
Reads: Stage 0's `.pt` files + raw SAbDab/Thera-SAbDab TSVs.
Produces: `tables/master_antibodies.csv` (20,037 rows at this point),
`tables/build_report.json`.

### Stage 2 - Chain-mislabel quality filter (20,037 → 19,848)
```
python scripts/scope_chain_mislabel.py --config configs/config.yaml
```
Read-only investigation. Produces: `tables/chain_mislabel_scope.json`.
```
python scripts/patch_apply_chain_mislabel_exclusions.py \
    --master-csv tables/master_antibodies.csv \
    --out-master-csv tables/master_antibodies_clean.csv
```
Reads: `tables/master_antibodies.csv` (20,037 rows), `tables/exclusions.csv`.
Produces: `tables/master_antibodies_clean.csv` (19,848 rows), updated
`tables/exclusions.csv`. Self-verifies row counts before exiting.

Then, only after confirming the integrity check passed:
```
cp tables/master_antibodies.csv tables/master_antibodies.csv.bak_pre_chain_mislabel_patch
mv tables/master_antibodies_clean.csv tables/master_antibodies.csv
```
From here on, `master_antibodies.csv` is the final 19,848-row analysis
population referenced everywhere else in this document and in the paper.

### Stage 3 - RQ1 core analysis
```
python scripts/rq1_sequence_structural_bias.py --config configs/config.yaml
```
Reads: `tables/master_antibodies.csv` + `.pt` files.
Produces (Sections A–E): `tables/rq1_cdrh3_clusters.tsv`,
`tables/rq1_antigen_landscape.json`,
`tables/rq1_antigen_class_distribution.csv`,
`tables/rq1_antigen_class_vs_therapeutic.csv`,
`tables/rq1_sequence_metadata_bias.json`,
`tables/rq1_length_distribution.csv`, `tables/rq1_method_vs_length.csv`,
`tables/rq1_year_distribution.csv`,
`tables/rq1_heavy_germline_family_distribution.csv`,
`tables/rq1_heavy_species_distribution.csv`,
`tables/rq1_structural_redundancy_paratope.json`,
`tables/rq1_backbone_redundancy.json`, `tables/rq1_paratope_all_cdrs.json`.

This is the single canonical source of `classify_antigen()` and
`extract_cdrh3_sequences()` - every other script that needs either
imports them from here rather than keeping a local copy, so results
cannot drift out of sync across the pipeline. Can be run with
`--only A`/`B`/`C`/`D`/`E` individually; Section D requires Section B to
have run at least once first (reads its `rq1_cdrh3_clusters.tsv`).

### Stage 4 - Patch derived columns onto the master table
Requires Stage 3's `rq1_cdrh3_clusters.tsv` and `rq1_antigen_landscape.json`
for its self-verification checks.
```
python scripts/migrate_master_csv.py --config configs/config.yaml --only h3_seq
python scripts/migrate_master_csv.py --config configs/config.yaml --only cluster_rep
python scripts/migrate_master_csv.py --config configs/config.yaml --only antigen_class
```
Or all three (plus the read-only `diagnose` op) in one safe-ordered call:
```
python scripts/migrate_master_csv.py --config configs/config.yaml --only all
```
Each op backs up `master_antibodies.csv` before its first run
(`.bak_pre_<op>_patch`). `--only antigen_class` exits with status 1 and
prints `[MISMATCH -- DO NOT TRUST THIS COLUMN YET]` if its output doesn't
exactly reproduce `rq1_antigen_landscape.json`'s `antigen_class_counts` -
treat that as a hard stop, not a warning.

### Stage 5 - Germline allele-resolution extension
```
python scripts/rq_germline_00_fetch_reference.py --config configs/config.yaml
python scripts/rq_germline_01_assign.py --config configs/config.yaml
python scripts/anarci_germline_validation.py --config configs/config.yaml \
    --germline_json tables/rq_germline_allele_assignment.json \
    --master_csv tables/master_antibodies.csv --n_sample 300 --seed 0
```
Produces: `tables/fetch_reference_report.json`,
`tables/rq_germline_allele_assignment.json`,
`tables/anarci_germline_cross_validation.json`.

### Stage 6 - Antigen classifier validation
Requires Stage 4's `antigen_class` column.
```
python scripts/validate_antigen_classifier_sample.py --config configs/config.yaml \
    --n_random 500 --n_rare_class 30 --n_other_protein_audit 100
```
→ hand-label `tables/antigen_classifier_sample_for_review.csv` (manual,
cannot be automated; a `tables/antigen_classifier_sample_meta.json`
sidecar records the sampling parameters used) →
```
python scripts/validate_antigen_classifier_score.py --config configs/config.yaml
```
Produces: `tables/antigen_classifier_validation.json` - the source of the
paper's classifier-validation appendix.

### Stage 7 - RQ1 supporting audits
Each is independently runnable once Stage 3 (+ Stage 4 for the two that
need `antigen_class`) has completed; no ordering between them except
where noted.
```
python scripts/audits/audit_exclusion_funnel.py --config configs/config.yaml
python scripts/audits/audit_h3_boundary.py --config configs/config.yaml
python scripts/audits/audit_antigen_construction.py --config configs/config.yaml \
    --struct_dir <raw_data>/all_structures
python scripts/audits/audit_antigen_bound_denominators.py --config configs/config.yaml
python scripts/audits/audit_dedup_sensitivity.py --config configs/config.yaml --n_seeds 200
python scripts/audits/audit_rmsd_weighting.py --config configs/config.yaml --max_cluster_size 1000
python scripts/audits/audit_paratope_h3.py --config configs/config.yaml \
    --struct_dir <raw_data>/all_structures --n_sample 2000 --n_boot 2000
python scripts/audits/audit_paratope_all_loops.py --config configs/config.yaml \
    --struct_dir <raw_data>/all_structures --n_sample 2000 --n_boot 2000
python scripts/audits/audit_nonnumeric_models.py --config configs/config.yaml \
    --struct_dir <raw_data>/all_structures --pdb_ids 2kh2,2ltq,7ssh,7st3,7stg,7ums
python scripts/audits/audit_sars_cov2_sensitivity.py --config configs/config.yaml
python scripts/audits/audit_therapeutic_tiers.py --config configs/config.yaml \
    --dedup_circularity_json tables/dedup_circularity_check.json   # run after audit_dedup_sensitivity.py
python scripts/audits/audit_sabdab2_splits.py --config configs/config.yaml \
    --splits_dir <sabdab2_splits_dir>   # run after Stage 4 (needs cluster_rep + antigen_class);
                                        # requires the separate SAbDab2/Zenodo download, see Section 1
```
`audit_paratope_h3.py`'s output (`paratope_contact_redefinition.json`) is
a hard input to `make_figures.py` in Stage 10, not just a standalone
audit result - it must run before Stage 10.

Each audit above writes one JSON to `tables/`, named after the script
(e.g. `audit_exclusion_funnel.py` → `row_entry_accounting.json`;
`audit_h3_boundary.py` → `h3_boundary_length_distributions.json` and
`h3_boundary_anarci_cross_check.json`; `audit_antigen_construction.py` →
`antigen_construction_audit.json`; `audit_antigen_bound_denominators.py`
→ `antigen_bound_denominator_audit.json`; `audit_dedup_sensitivity.py` →
`dedup_circularity_check.json`; `audit_rmsd_weighting.py` →
`rmsd_pair_vs_cluster_weighted.json`; `audit_paratope_h3.py` →
`paratope_contact_redefinition.json`; `audit_paratope_all_loops.py` →
`paratope_all_loops_contact_redefinition.json`;
`audit_nonnumeric_models.py` → `nonnumeric_model_investigation.json`;
`audit_sars_cov2_sensitivity.py` → `audit_sars_cov2_sensitivity.json`;
`audit_therapeutic_tiers.py` → `therapeutic_tier_audit.json`;
`audit_sabdab2_splits.py` → `audit_sabdab2_splits.json`) - each script
also prints its output path on completion.

### Stage 8 - RQ2
```
python scripts/rq2_oas_comparison.py --config configs/config.yaml
python scripts/rq2_composition_analysis.py --config configs/config.yaml
```
Produces: `tables/rq2_length_distributions.csv`,
`tables/rq2_oas_comparison_summary.json`,
`tables/rq2_extended_diversity.json` (from `rq2_oas_comparison.py`), and
`tables/composition_by_weighting_scheme.json` (from
`rq2_composition_analysis.py`, which also reads the `rq2_*.json` outputs
above for its OAS-heavy comparison).

### Stage 9 - RQ3
```
python scripts/rq3_redundancy_and_recommendations.py --config configs/config.yaml
python scripts/rq3_dedup_threshold_sensitivity.py --config configs/config.yaml \
    --thresholds 0.80 0.85 0.90 0.95
```
Produces: `tables/rq3_before_after_dedup.json`,
`tables/rq3_deduplicated_master.csv` (the cluster-representative subset),
and `tables/rq3_dedup_threshold_sensitivity.json`.
The threshold-sensitivity script imports `extract_cdrh3_sequences`/
`run_mmseqs2_cluster` directly from `rq1_sequence_structural_bias.py`
rather than keeping a local copy, for the same reason given in Stage 3.

### Stage 10 - Figures
```
python scripts/make_figures.py --config configs/config.yaml
```
Reads the full set of `rq1_*`/`rq2_*`/`rq3_*`/`paratope_contact_redefinition.json`
outputs above. Run last.

---

## Part 2 - One-off scripts

- `scope_chain_mislabel.py` - read-only; safe to re-run any time for a
  fresh look, never mutates data.
- `patch_apply_chain_mislabel_exclusions.py` - a one-time data migration
  (folded into Part 1, Stage 2, since later stages depend on its output).
  Re-running it against an already-patched master table is guarded by its
  own self-check and will abort rather than double-apply.

Every other script under `audits/`, `rq_*`, `validate_*`, and
`anarci_germline_validation.py` writes an output that either the paper
cites directly or another script in Part 1 reads - treat all of them as
part of the reproducible chain, safe to re-run as needed.

