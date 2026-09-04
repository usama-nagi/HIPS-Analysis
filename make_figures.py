#!/usr/bin/env python3
"""
make_figures.py
=====================

Usage
-----
    python scripts/make_figures.py --config configs/config.yaml
"""

import os
import sys
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, log

# Shared style
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
GRAY = "#4d4d4d"
LIGHT_GRID = "#dddddd"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": LIGHT_GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "font.size": 9,
})


def _save(fig, work_dir, name):
    out_path = os.path.join(work_dir, "figures", name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"Wrote {out_path}")

def _barh_sorted(ax, labels, values, color=BLUE, top_n=None, fmt="{:,.0f}", fontsize=7.5):
    """Horizontal bar chart, sorted descending, value-labeled. Avoids rotated
    x-tick labels entirely, which is the single biggest legibility problem
    in the v1 species/antigen-class bars."""
    order = np.argsort(values)[::-1]
    if top_n:
        order = order[:top_n]
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    y = np.arange(len(labels))[::-1]  # largest at top
    ax.barh(y, values, color=color, height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=fontsize)
    xmax = max(values) if values else 1
    for yi, v in zip(y, values):
        ax.text(v + xmax * 0.015, yi, fmt.format(v), va="center", fontsize=fontsize - 0.5, color=GRAY)
    ax.set_xlim(0, xmax * 1.18)
    return ax

def _bar_sorted_vertical(ax, labels, values, color=BLUE, fmt="{:,.0f}", fontsize=7.5, rotation=0):
    order = np.argsort(values)[::-1]
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    x = np.arange(len(labels))
    ax.bar(x, values, color=color, width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fontsize, rotation=rotation, ha="right" if rotation else "center")
    ymax = max(values) if values else 1
    for xi, v in zip(x, values):
        ax.text(xi, v + ymax * 0.02, fmt.format(v), ha="center", fontsize=fontsize - 0.5, color=GRAY)
    ax.set_ylim(0, ymax * 1.15)
    return ax

def figure1_dataset_composition(work_dir):
    tables = os.path.join(work_dir, "tables")
    summary_path = os.path.join(tables, "rq1_sequence_metadata_bias.json")
    if not os.path.exists(summary_path):
        log("SKIP figure1: rq1_sequence_metadata_bias.json not found")
        return
    with open(summary_path) as f:
        s = json.load(f)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # (a) length distribution
    len_csv = os.path.join(tables, "rq1_length_distribution.csv")
    if os.path.exists(len_csv):
        ld = pd.read_csv(len_csv, index_col=0)
        axes[0, 0].bar(ld.index.astype(str), ld.iloc[:, 0], color=BLUE)
        axes[0, 0].set_title("CDR-H3 length")
        axes[0, 0].set_xlabel("length (aa)")
        axes[0, 0].tick_params(axis="x", rotation=90, labelsize=6)

    # (b) heavy germline family - sorted, value-labeled, top 15
    hg_csv = os.path.join(tables, "rq1_heavy_germline_family_distribution.csv")
    if os.path.exists(hg_csv):
        hg = pd.read_csv(hg_csv, index_col=0).sort_values("count", ascending=False).head(15)
        _bar_sorted_vertical(axes[0, 1], list(hg.index.astype(str)), list(hg.iloc[:, 0]),
                              color=BLUE, rotation=90, fontsize=6.5)
        axes[0, 1].set_title("Heavy germline family (top 15)")

    # (c) species - horizontal, sorted, labeled (replaces rotated vertical bars)
    sp_csv = os.path.join(tables, "rq1_heavy_species_distribution.csv")
    if os.path.exists(sp_csv):
        sp = pd.read_csv(sp_csv, index_col=0).sort_values("count", ascending=False).head(10)
        _barh_sorted(axes[0, 2], list(sp.index.astype(str)), list(sp.iloc[:, 0]), color=BLUE, fontsize=7)
        axes[0, 2].set_title("Heavy-chain species (top 10)")

    # (d) therapeutic fraction - horizontal bar instead of pie
    ther = s.get("therapeutic", {})
    if ther:
        n_ther = ther.get("n_therapeutic", 0)
        n_non = ther.get("n_non_therapeutic", 0)
        total = n_ther + n_non
        pct_ther = 100 * n_ther / total if total else 0
        axes[1, 0].barh([0, 1], [pct_ther, 100 - pct_ther], color=[ORANGE, BLUE])
        axes[1, 0].set_yticks([0, 1])
        axes[1, 0].set_yticklabels([f"Therapeutic\n(n={n_ther:,})", f"Non-therapeutic\n(n={n_non:,})"], fontsize=8)
        axes[1, 0].set_xlabel("% of entries")
        axes[1, 0].set_xlim(0, 100)
        for yi, v in zip([0, 1], [pct_ther, 100 - pct_ther]):
            axes[1, 0].text(v + 1.5, yi, f"{v:.1f}%", va="center", fontsize=8)
        axes[1, 0].set_title("Therapeutic enrichment")

    # (e) year trend
    yr_csv = os.path.join(tables, "rq1_year_distribution.csv")
    if os.path.exists(yr_csv):
        yr = pd.read_csv(yr_csv, index_col=0)
        axes[1, 1].plot(yr.index, yr.iloc[:, 0], marker="o", markersize=3, color=BLUE)
        axes[1, 1].set_title("Depositions per year")
        axes[1, 1].set_xlabel("year")

    # (f) method vs length confound
    ml_csv = os.path.join(tables, "rq1_method_vs_length.csv")
    if os.path.exists(ml_csv):
        ml = pd.read_csv(ml_csv, index_col=0)
        if "mean" in ml.columns:
            ml_sorted = ml.sort_values("mean", ascending=False)
            # Source data (SAbDab `method` field) is ALL CAPS (e.g.
            # "ELECTRON MICROSCOPY", "X-RAY DIFFRACTION"); apply .title()
            # casing for display only, here.
            display_labels = ml_sorted.index.astype(str).str.title()
            axes[1, 2].bar(display_labels, ml_sorted["mean"],
                            yerr=ml_sorted.get("std"), color=BLUE)
            axes[1, 2].set_title("Mean CDR-H3 length by method")
            axes[1, 2].tick_params(axis="x", labelsize=6.5)
            plt.setp(axes[1, 2].get_xticklabels(), rotation=30, ha="right")

    fig.suptitle("Figure: SAbDab dataset composition", fontsize=13)
    fig.tight_layout()
    _save(fig, work_dir, "figure1_dataset_composition.png")

def figure2_redundancy_paratope(work_dir):
    """
    Left panel: cluster-size bars, PLUS an on-panel annotation of
    exact_duplicate_fraction, 

    Right panel reads paratope_contact_redefinition.json:
    fold-change is contacting/non-contacting (a true partition, not nested
    sets) at a 10A distance, WITH 95% CI error bars from 2,000 CDR-H3-cluster
    bootstrap resamples
    """
    tables = os.path.join(work_dir, "tables")
    summary_path = os.path.join(tables, "rq1_structural_redundancy_paratope.json")
    if not os.path.exists(summary_path):
        log("SKIP figure2: rq1_structural_redundancy_paratope.json not found")
        return
    with open(summary_path) as f:
        s = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    redund = s.get("cdrh3_redundancy")
    if redund:
        top10 = redund.get("top10_largest_clusters", {})
        if top10:
            vals = list(top10.values())
            axes[0].bar(range(len(vals)), vals, color=BLUE)
            axes[0].set_xticks(range(len(vals)))
            axes[0].set_xticklabels([f"#{i+1}" for i in range(len(vals))], fontsize=7)
            for i, v in enumerate(vals):
                axes[0].text(i, v + max(vals) * 0.015, f"{v}", ha="center", fontsize=7, color=GRAY)
            axes[0].set_title("Top 10 CDR-H3 clusters (90% identity)")
            axes[0].set_ylabel("cluster size")
            edf = redund.get("exact_duplicate_fraction")
            n_seq = redund.get("n_sequences")
            if edf is not None:
                label_n = f"{n_seq:,}" if n_seq else "all"
                axes[0].text(
                    0.97, 0.95,
                    f"{edf*100:.1f}% of {label_n} entries\nshare an exact CDR-H3\nwith another entry",
                    transform=axes[0].transAxes, ha="right", va="top", fontsize=8,
                    color=GRAY, bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec=LIGHT_GRID),
                )

    contact_path = os.path.join(tables, "paratope_contact_redefinition.json")
    if os.path.exists(contact_path):
        with open(contact_path) as f:
            c = json.load(f)
        t10 = c.get("by_threshold", {}).get("10.0", {})
        comp = t10.get("composition", {})
        ci = t10.get("cluster_bootstrap_ci_fold_change", {})
        entries = [(aa, v["fold_change"], ci.get(aa)) for aa, v in comp.items()
                   if v.get("fold_change") is not None]
        entries.sort(key=lambda e: e[1])
        labels = [e[0] for e in entries]
        vals = [e[1] for e in entries]
        lo_err = [max(v - (e[2]["ci_2.5"] if e[2] else v), 0) for v, e in zip(vals, entries)]
        hi_err = [max((e[2]["ci_97.5"] if e[2] else v) - v, 0) for v, e in zip(vals, entries)]
        significant = [e[2] is not None and not (e[2]["ci_2.5"] <= 1.0 <= e[2]["ci_97.5"])
                       for e in entries]
        colors = [
            (GREEN if v >= 1.0 else ORANGE) if sig else LIGHT_GRID
            for v, sig in zip(vals, significant)
        ]
        y = np.arange(len(labels))
        axes[1].barh(y, vals, xerr=[lo_err, hi_err], color=colors,
                     error_kw=dict(ecolor=GRAY, elinewidth=0.8, capsize=2))
        axes[1].axvline(1.0, color=GRAY, linewidth=1, linestyle="--")
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(labels, fontsize=8)
        for i, v in enumerate(vals):
            axes[1].text(v + hi_err[i] + 0.15, i, f"{v:.2f}\u00d7",
                          ha="left", va="center", fontsize=7, color=GRAY)
        axes[1].set_xlabel("fold-change (contacting / non-contacting, 10\u00c5)")
        axes[1].set_title("CDR-H3 antigen-contact enrichment by residue\n"
                           "(95% CI, gray = CI includes 1)")

    fig.suptitle("Figure: CDR-H3 redundancy is exact repetition; contact enrichment is a fold-change story", fontsize=12)
    fig.tight_layout()
    _save(fig, work_dir, "figure2_redundancy_paratope.png")


def figure3_antigen_landscape(work_dir):
    """
    written by 00_build_dataset.py) -- a one-line groupby, not a new
    aggregation script. This also independently recomputes Gini from raw
    per-entry cluster assignments at render time, which is a useful
    cross-check: if the value shown here ever drifts from the Gini reported
    elsewhere (e.g. Table 3, Section 4.2.4), that is worth flagging rather
    than assumed to be a rendering artifact.
    """
    tables = os.path.join(work_dir, "tables")
    cls_csv = os.path.join(tables, "rq1_antigen_class_distribution.csv")
    master_csv = os.path.join(tables, "master_antibodies.csv")
    if not os.path.exists(cls_csv):
        log("SKIP figure3: rq1_antigen_class_distribution.csv not found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    cls = pd.read_csv(cls_csv, index_col=0).sort_values("count", ascending=False)
    _barh_sorted(axes[0], list(cls.index.astype(str)), list(cls.iloc[:, 0]), color=BLUE, fontsize=8)
    axes[0].set_title("Antigen class (keyword heuristic)")

    if os.path.exists(master_csv):
        m = pd.read_csv(master_csv, usecols=["antigen_cluster_id"])
        sizes = m["antigen_cluster_id"].dropna().value_counts().sort_values(ascending=True).values
        n_clusters = len(sizes)
        n_complexes = int(sizes.sum())
        if n_clusters > 0 and n_complexes > 0:
            cum_clusters = np.concatenate([[0], np.arange(1, n_clusters + 1) / n_clusters])
            cum_complexes = np.concatenate([[0], np.cumsum(sizes) / n_complexes])
            # gini = abs(1 - 2 * np.trapz(cum_complexes, cum_clusters))
            # gini = abs(1 - 2 * np.trapezoid(cum_complexes, cum_clusters))
            gini = abs(1 - 2 * np.trapz(cum_complexes, cum_clusters))

            axes[1].plot([0, 1], [0, 1], color=GRAY, linewidth=1, linestyle="--", label="perfect equality")
            axes[1].plot(cum_clusters, cum_complexes, color=ORANGE, linewidth=2,
                         label="antigen clusters (smallest first)")
            axes[1].fill_between(cum_clusters, cum_complexes, cum_clusters, color=ORANGE, alpha=0.15)

            if n_clusters >= 10:
                idx10 = n_clusters - 10
                x10, y10 = cum_clusters[idx10], cum_complexes[idx10]
                top10_share = 1.0 - y10
                axes[1].plot([x10], [y10], marker="o", color=ORANGE, markersize=6, zorder=5)
                axes[1].annotate(
                    f"top 10 clusters =\n{top10_share*100:.1f}% of complexes",
                    xy=(x10, y10), xytext=(x10 - 0.55, y10 - 0.05), fontsize=8, color=GRAY,
                    arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6),
                )
            axes[1].set_xlim(0, 1)
            axes[1].set_ylim(0, 1)
            axes[1].set_xlabel("cumulative share of antigen clusters")
            axes[1].set_ylabel("cumulative share of antigen-bound complexes")
            axes[1].legend(fontsize=7, frameon=False, loc="lower right")
            axes[1].set_title(f"Antigen cluster concentration (Lorenz curve)\n({n_clusters} unique clusters, Gini={gini:.2f})")

    fig.suptitle("Figure: Antigen landscape", fontsize=13)
    fig.tight_layout()
    _save(fig, work_dir, "figure3_antigen_landscape.png")


def figure4_oas_comparison(work_dir):
    """
    adds a residual panel (SAbDab_frac - OAS_heavy_frac) beneath the
    original overlay, which shows the same "how do these compare" question
    the reviewer wanted, but quantifies exactly how small the gap is and
    exactly which CDR-H3 lengths it concentrates at, rather than asserting a
    generic before/after transformation. The largest single deviation is
    auto-annotated (found via argmax on the real data at render time, not
    hardcoded), so this stays correct even if the underlying distributions
    are ever regenerated.
    """
    tables = os.path.join(work_dir, "tables")
    dist_csv = os.path.join(tables, "rq2_length_distributions.csv")
    summary_path = os.path.join(tables, "rq2_oas_comparison_summary.json")
    if not os.path.exists(dist_csv):
        log("SKIP figure4: rq2_length_distributions.csv not found")
        return

    df = pd.read_csv(dist_csv)
    with open(summary_path) as f:
        s = json.load(f)

    def _norm(col):
        total = df[col].sum()
        return df[col] / total if total > 0 else df[col]

    sabdab_frac = _norm("sabdab_human_count")
    has_heavy = "oas_heavy_count" in df.columns and df["oas_heavy_count"].sum() > 0
    has_paired = "oas_paired_count" in df.columns and df["oas_paired_count"].sum() > 0

    fig, (ax_top, ax_res) = plt.subplots(
        2, 1, figsize=(8, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_top.plot(df["cdr3_length"], sabdab_frac, marker="o", color=BLUE, markersize=4,
                label=f"SAbDab (human, n={s.get('sabdab_human_n', '?')})")
    oas_heavy_frac = None
    if has_heavy:
        oas_heavy_frac = _norm("oas_heavy_count")
        ax_top.plot(df["cdr3_length"], oas_heavy_frac, marker="s", color=ORANGE, markersize=4,
                    label=f"OAS heavy (n={s.get('oas_heavy_sampled_n', '?')})")
    if has_paired:
        ax_top.plot(df["cdr3_length"], _norm("oas_paired_count"), marker="^", color=GREEN, markersize=4,
                    label=f"OAS paired (n={s.get('oas_paired_sampled_n', '?')})")

    jsd = s.get("length_jsd_sabdab_vs_oas_heavy")
    title = "Figure: SAbDab (human) vs. OAS \u2014 CDR-H3 length distribution"
    if jsd is not None:
        title += f"  (JS divergence vs. OAS heavy = {jsd:.3f} bits)"
    ax_top.set_title(title, fontsize=11)
    ax_top.set_ylabel("normalized frequency")
    ax_top.legend(fontsize=8, frameon=False)

    if oas_heavy_frac is not None:
        residual = (sabdab_frac - oas_heavy_frac).values
        colors = [BLUE if r >= 0 else ORANGE for r in residual]
        ax_res.bar(df["cdr3_length"], residual, color=colors, width=0.8)
        ax_res.axhline(0, color=GRAY, linewidth=0.8)
        ax_res.set_ylabel("SAbDab \u2212 OAS\n(residual)", fontsize=8)
        peak_idx = int(np.argmax(np.abs(residual)))
        peak_len = df["cdr3_length"].iloc[peak_idx]
        peak_val = residual[peak_idx]
        ax_res.annotate(
            f"largest deviation:\nlen={peak_len}, \u0394={peak_val:+.3f}",
            xy=(peak_len, peak_val),
            xytext=(0, 12 if peak_val >= 0 else -12), textcoords="offset points",
            ha="center", fontsize=7, color=GRAY,
            arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6),
        )
    ax_res.set_xlabel("CDR-H3 length (aa)")

    fig.tight_layout()
    _save(fig, work_dir, "figure4_oas_comparison.png")


def figure5_dedup_before_after(work_dir):
    """
    Create Figure 5: Comparison of deduplication effects before and after.
    """
    tables = os.path.join(work_dir, "tables")
    summary_path = os.path.join(tables, "rq3_before_after_dedup.json")
    if not os.path.exists(summary_path):
        log("SKIP figure5: rq3_before_after_dedup.json not found")
        return
    with open(summary_path) as f:
        s = json.load(f)

    metrics = [k for k in s["before"] if k != "n_antibodies"
               and isinstance(s["before"][k], (int, float))]
    before_vals = [s["before"][m] for m in metrics]
    after_vals = [s["after"][m] for m in metrics]
    pct_change = [
        100 * (a - b) / b if b else 0
        for b, a in zip(before_vals, after_vals)
    ]

    n_metrics = len(metrics)
    n_cols = 4
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 2.6 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for i, (m, bv, av, pct) in enumerate(zip(metrics, before_vals, after_vals, pct_change)):
        ax = axes_flat[i]
        bars = ax.bar([0, 1], [bv, av], color=[BLUE, ORANGE], width=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Before", "After"], fontsize=7.5)
        # own y-axis per subplot -- this is the actual fix: each metric gets
        # scaled to its own range instead of sharing one axis with every
        # other metric regardless of magnitude
        vmax = max(bv, av) or 1
        ax.set_ylim(0, vmax * 1.28)
        for bar_obj, v in zip(bars, [bv, av]):
            label = f"{v:,.3f}" if abs(v) < 10 else f"{v:,.1f}" if abs(v) < 1000 else f"{v:,.0f}"
            ax.text(bar_obj.get_x() + bar_obj.get_width() / 2, v + vmax * 0.04,
                     label, ha="center", fontsize=6.5, color=GRAY)
        sign = "+" if pct >= 0 else ""
        ax.set_title(f"{m}\n{sign}{pct:.1f}%", fontsize=7.5, fontweight="bold")
        ax.tick_params(axis="y", labelsize=6)

    # hide any unused trailing subplot cells
    for j in range(n_metrics, len(axes_flat)):
        axes_flat[j].axis("off")

    n_before = s["before"]["n_antibodies"]
    n_after = s["after"]["n_antibodies"]
    fig.suptitle(
        f"Figure: Diversity metrics, before vs. after cluster-aware dedup\n"
        f"(n={n_before} \u2192 {n_after}, "
        f"{s['delta']['n_antibodies']['reduction_fraction']*100:.1f}% removed) "
        f"-- each panel on its own scale",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, work_dir, "figure5_dedup_before_after.png")


def figure6_threshold_sensitivity(work_dir):
    """New figure. Reads rq3_dedup_threshold_sensitivity.json (confirmed
    filename and structure - same file already used for Table 4) directly,
    no new computation. JSON shape:
        {
          "before": {..., "antigen_cluster_gini": <float>, ...},
          "by_threshold": {
            "0.4":  {"n_clusters": int, "compression_ratio": float,
                      "after": {..., "antigen_cluster_gini": float, ...},
                      "delta": {"n_antibodies": {"reduction_fraction": float, ...}, ...}},
            "0.45": {...}, ... "0.95": {...}
          }
        }
    Threshold keys are string-encoded fractions (e.g. "0.4" = 40% identity)."""
    tables = os.path.join(work_dir, "tables")
    sens_json = os.path.join(tables, "rq3_dedup_threshold_sensitivity.json")
    if not os.path.exists(sens_json):
        log("SKIP figure6: rq3_dedup_threshold_sensitivity.json not found")
        return

    with open(sens_json) as f:
        d = json.load(f)

    thr_keys = sorted(d["by_threshold"].keys(), key=float)
    threshold = [float(k) * 100 for k in thr_keys]
    reduction_pct = [d["by_threshold"][k]["delta"]["n_antibodies"]["reduction_fraction"] * 100
                      for k in thr_keys]
    antigen_gini_after = [d["by_threshold"][k]["after"]["antigen_cluster_gini"] for k in thr_keys]
    before_antigen_gini = d["before"]["antigen_cluster_gini"]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
    l1, = ax1.plot(threshold, reduction_pct, marker="o", color=BLUE, linewidth=2, markersize=5,
                    label="Reduction fraction (%)")
    ax1.set_xlabel("CDR-H3 clustering threshold (% identity)")
    ax1.set_ylabel("Reduction fraction after dedup (%)", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)

    ax2 = ax1.twinx()
    ax2.grid(False)
    l2, = ax2.plot(threshold, antigen_gini_after, marker="s", color=ORANGE,
                    linewidth=2, markersize=5, label="Antigen-cluster Gini (after)")
    ax2.set_ylabel("Antigen-cluster Gini, after dedup", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)

    if 90.0 in threshold:
        ax1.axvline(90, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        primary_idx = threshold.index(90.0)
        ax1.annotate("90% (primary)", xy=(90, reduction_pct[primary_idx]),
                      xytext=(threshold[0] + 0.6 * (90 - threshold[0]), min(reduction_pct) - 0.6),
                      fontsize=8.5, color="gray",
                      arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    ax1.text(threshold[0] + 0.3, max(reduction_pct) + 0.5,
              f"Before dedup: antigen-cluster Gini = {before_antigen_gini:.3f}\n"
              f"(off-scale above; shown for reference only)",
              fontsize=7.5, color="dimgray", va="top")

    ax1.set_xticks(threshold)
    ax1.set_xticklabels([f"{int(t)}" for t in threshold], fontsize=8)

    lines = [l1, l2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower left", fontsize=8.5, frameon=True)
    ax1.set_title("Deduplication effect is stable across the clustering threshold", fontsize=11)
    fig.tight_layout()
    _save(fig, work_dir, "figure6_threshold_sensitivity.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    work_dir = cfg["paths"]["work_dir"]
    os.makedirs(os.path.join(work_dir, "figures"), exist_ok=True)

    figure1_dataset_composition(work_dir)
    figure2_redundancy_paratope(work_dir)
    figure3_antigen_landscape(work_dir)
    figure4_oas_comparison(work_dir)
    figure5_dedup_before_after(work_dir)
    figure6_threshold_sensitivity(work_dir)   # new, needs threshold-sweep CSV
    log("Figure generation complete (any SKIP lines above mean the "
        "corresponding analysis output hasn't been found yet).")

if __name__ == "__main__":
    main()