# Benchmark Design Checklist

Generated from quantitative analysis — each item cites its supporting statistic file.

## 1. Cluster-aware CDR-H3 splitting

Split train/test by CDR-H3 sequence cluster, not by PDB ID. At 90% identity threshold, 19848 SAbDab entries collapse into only 5404 clusters (compression 3.71x; 35.6% are singletons). A PDB-ID-only split allows near-duplicate CDR-H3 loops into both train and test.

*Source: `rq1_structural_redundancy_paratope.json:cdrh3_redundancy`*

## 2. Antigen-cluster-balanced splitting

Use antigen_cluster_id (already computed in the existing preprocessing pipeline via MMseqs2) as the split key for antigen-disjoint evaluation. The top 10 antigen clusters account for 29.9% of antigen-bound complexes (Gini=0.72 across 2292 clusters) — a random split risks leaking near-identical antigens across train/test.

*Source: `rq1_structural_redundancy_paratope.json:antigen_redundancy_reused_from_existing_pipeline`*

## 3. Report diversity metrics, not just size

Any SAbDab-derived benchmark should report CDR-H3 length entropy and Gini alongside sample count. This dataset's full (non-deduplicated) length distribution has entropy=4.12 bits (normalized=0.82) and Gini=0.58 — a benchmark subset reporting substantially lower entropy than this is measurably less diverse than the source dataset, and that should be disclosed.

*Source: `rq1_sequence_metadata_bias.json:length`*

## 4. Disclose germline-family balance (at available resolution)

The top 5 heavy-chain germline FAMILIES (IGHV-level only — allele/J-gene not available in SAbDab's summary export) account for 91.0% of entries (Gini=0.71 across 14 families). Benchmarks should report this family-level breakdown explicitly and disclose that finer-grained allele bias is NOT captured by this statistic.

*Source: `rq1_sequence_metadata_bias.json:heavy_germline_family`*

**Limitation:** family-level only (e.g. IGHV1), no allele or J-gene - not present in SAbDab summary TSV

## 5. Apply cluster-aware deduplication before computing benchmark statistics

Cluster-aware deduplication (one representative per CDR-H3 cluster) removed 73.0% of entries in this dataset and changed measured diversity metrics by the amounts in rq3_before_after_dedup.json. Any benchmark statistic (entropy, Gini, class balance) computed BEFORE this dedup step is measuring redundancy, not diversity, to a quantifiable degree — report both, or report only the deduplicated numbers.

*Source: `rq3_before_after_dedup.json:delta`*

## 6. State species scope explicitly when benchmarking against repertoire data (e.g. OAS)

When using OAS or similar repertoire data as a diversity reference, state the species scope explicitly. This OAS mmap was built with an explicit human-only sample filter applied at construction time, confirmed against the preprocessing pipeline (not merely a loader-default assumption) -- so the heavy_species == 'homo sapiens' restriction on the SAbDab side is a genuine like-for-like species match, not a one-sided restriction against an uncharacterized comparator. For broader context: a separate, independently-verified IgLM-derived reprocessing of OAS (588,488,146 sequences across the full train+test release) shows the unfiltered OAS pool is 82.14% human / 17.86% non-human (mouse, rat, rabbit, camelid, rhesus) -- this figure describes OAS generally and supersedes this project's earlier 'assumed human-only by convention' caveat, but it is not itself evidence about this specific mmap, whose human-only composition is independently confirmed by construction. The length-distribution divergence measured here (JS=0.02155258294513146) is only valid under that scope restriction — re-introducing other species on the SAbDab side without a matching OAS comparison would silently break the comparison's validity.

*Source: `rq2_oas_comparison_summary.json`*
