"""
Hard Thresholding Analysis for Gene Co-Expression Benchmark
Accepts 3 similarity matrices and produces threshold selection plots + thresholded networks.

Methods based on:
- Couto et al. (2017) Mol. BioSyst. 13:2024-2035
- Zhang & Horvath (2005) Stat. Appl. Genet. Mol. Biol. 4:1128

Usage:
    python src/hard_threshold_analysis.py \
        --inputs data/similarity/pearson_similarity.csv \
                 data/similarity/spearman_similarity.csv \
                 data/similarity/rho_similarity.csv \
        --names Pearson Spearman Proportionality \
        --output results/figures/hard_threshold
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sp_connected_components

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Shared style ───────────────────────────────────────
NAVY   = "#1B2A4A"
TEAL   = "#0D9488"
SLATE  = "#64748B"
RED    = "#EF4444"
AMBER  = "#F59E0B"
GREEN  = "#10B981"
PURPLE = "#8B5CF6"

COLORS = [TEAL, RED, PURPLE]


# ══════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════
def load_matrix(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    logger.info("Loading: %s", path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path, index_col=0)
    elif path.suffix in {".tsv", ".txt"}:
        df = pd.read_csv(path, sep="\t", index_col=0)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
    return df.astype("float32")


def get_upper_triangle(sim: pd.DataFrame) -> np.ndarray:
    vals = sim.values
    return vals[np.triu_indices_from(vals, k=1)]


# ══════════════════════════════════════════════════════
# THRESHOLD METRICS
# ══════════════════════════════════════════════════════
def compute_threshold_metrics(sim: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    """
    For each threshold, compute network topology metrics.
    Signed network: only positive correlations >= tau are kept.
    Uses sparse matrices to avoid memory overflow.
    Based on Couto et al. (2017).
    """
    n_genes = sim.shape[0]
    tri     = get_upper_triangle(sim)
    n_pairs = len(tri)

    rows = []
    for tau in thresholds:
        vals   = sim.values
        degree = ((vals >= tau) & ~np.eye(n_genes, dtype=bool)).sum(axis=1)

        n_edges     = int((tri >= tau).sum())
        pct_edges   = 100 * n_edges / n_pairs if n_pairs > 0 else 0
        mean_corr   = float(tri[tri >= tau].mean()) if (tri >= tau).sum() > 0 else 0.0
        mean_degree = float(degree.mean())
        std_degree  = float(degree.std())

        # Degree entropy
        counts, _ = np.histogram(degree, bins=min(50, n_genes))
        counts    = counts[counts > 0]
        probs     = counts / counts.sum()
        deg_entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))

        # Power-law fit
        deg_nonzero = degree[degree > 0]
        if len(deg_nonzero) > 10:
            try:
                hist, bin_edges = np.histogram(deg_nonzero, bins=20)
                bin_centers     = (bin_edges[:-1] + bin_edges[1:]) / 2
                mask            = hist > 0
                if mask.sum() >= 3:
                    slope, _, r_value, _, _ = stats.linregress(
                        np.log10(bin_centers[mask]),
                        np.log10(hist[mask] / hist[mask].sum())
                    )
                    powerlaw_r2    = r_value ** 2
                    powerlaw_slope = slope
                else:
                    powerlaw_r2, powerlaw_slope = 0.0, 0.0
            except Exception:
                powerlaw_r2, powerlaw_slope = 0.0, 0.0
        else:
            powerlaw_r2, powerlaw_slope = 0.0, 0.0

        # Giant component — use scipy sparse to avoid full dense matrix
        bool_adj = (vals >= tau)
        np.fill_diagonal(bool_adj, False)
        sparse = csr_matrix(bool_adj)
        n_comp, labels = sp_connected_components(sparse, directed=False)
        giant_size = int(np.bincount(labels).max())
        giant_pct  = 100 * giant_size / n_genes

        clustering = float("nan")  # computed later in topology comparison

        rows.append({
            "tau":            tau,
            "n_edges":        n_edges,
            "pct_edges":      pct_edges,
            "mean_corr":      mean_corr,
            "mean_degree":    mean_degree,
            "std_degree":     std_degree,
            "deg_entropy":    deg_entropy,
            "powerlaw_r2":    powerlaw_r2,
            "powerlaw_slope": powerlaw_slope,
            "giant_size":     giant_size,
            "giant_pct":      giant_pct,
            "n_components":   n_comp,
            "clustering":     clustering,
        })

        del bool_adj, sparse

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════
# THRESHOLD SELECTION METHODS
# ══════════════════════════════════════════════════════
def select_threshold_powerlaw(
    df: pd.DataFrame,
    min_giant_pct: float = 50.0,
    tau_min: float = 0.3,
) -> float:
    """
    Best power-law fit where:
    - tau >= tau_min (avoids trivial fully-connected network at tau=0)
    - giant_pct >= min_giant_pct (avoids fragmented networks)
    Based on Couto et al. (2017) Method 4.
    """
    filtered = df[(df["tau"] >= tau_min) & (df["giant_pct"] >= min_giant_pct)]
    if len(filtered) == 0:
        filtered = df[df["tau"] >= tau_min]  # relax giant constraint
    if len(filtered) == 0:
        filtered = df  # final fallback
    idx = filtered["powerlaw_r2"].idxmax()
    return float(df.loc[idx, "tau"])


def select_threshold_entropy(df: pd.DataFrame) -> float:
    """Method 2 from Couto et al.: maximum degree entropy."""
    idx = df["deg_entropy"].idxmax()
    return float(df.loc[idx, "tau"])


def select_threshold_giant_pct(df: pd.DataFrame, target_pct: float = 80.0) -> float:
    """Method 3 from Couto et al.: threshold retaining ~target_pct% of genes in giant component."""
    diffs = (df["giant_pct"] - target_pct).abs()
    idx   = diffs.idxmin()
    return float(df.loc[idx, "tau"])


def select_threshold_avg_degree(df: pd.DataFrame, target_degree: float = 100.0) -> float:
    """Method 5 from Couto et al.: threshold giving ~target average degree."""
    diffs = (df["mean_degree"] - target_degree).abs()
    idx   = diffs.idxmin()
    return float(df.loc[idx, "tau"])


# ══════════════════════════════════════════════════════
# FIGURE 1: Threshold Selection Dashboard
# ══════════════════════════════════════════════════════
def plot_threshold_dashboard(
    metrics_list: list[pd.DataFrame],
    names: list[str],
    out_dir: Path,
    min_giant_pct: float = 50.0,
) -> None:
    """
    Multi-panel dashboard showing all threshold selection criteria.
    One column per metric, rows = different criteria plots.
    """
    logger.info("Plotting threshold selection dashboard...")

    n_metrics = len(metrics_list)
    fig = plt.figure(figsize=(6 * n_metrics, 28))
    fig.suptitle(
        "Hard Threshold Selection Dashboard\n"
        "Based on Couto et al. (2017) Mol. BioSyst. 13:2024-2035",
        fontsize=14, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(7, n_metrics, figure=fig, hspace=0.45, wspace=0.35)

    for col, (df, name, color) in enumerate(zip(metrics_list, names, COLORS)):

        # ── Row 0: Network density ──────────────────────────────────
        ax = fig.add_subplot(gs[0, col])
        ax.plot(df["tau"], df["pct_edges"], color=color, lw=2)
        ax.set_title(f"{name}\nNetwork Density vs τ", fontsize=11, fontweight="bold")
        ax.set_xlabel("Hard Threshold (τ)")
        ax.set_ylabel("% Pairs Retained")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
        for tau_mark in [0.5, 0.7, 0.8]:
            pct = float(df.loc[(df["tau"] - tau_mark).abs().idxmin(), "pct_edges"])
            ax.axvline(tau_mark, color=SLATE, linestyle=":", lw=1, alpha=0.7)
            ax.text(tau_mark + 0.01, pct + 1, f"{pct:.1f}%", fontsize=7, color=SLATE)

        # ── Row 1: Mean degree ──────────────────────────────────────
        ax = fig.add_subplot(gs[1, col])
        ax.plot(df["tau"], df["mean_degree"], color=color, lw=2)
        for target, ls in [(100, "--"), (1000, ":")]:
            tau_target = select_threshold_avg_degree(df, target_degree=target)
            ax.axvline(tau_target, color=AMBER, linestyle=ls, lw=1.5,
                       label=f"avg_deg={target} → τ={tau_target:.2f}")
        ax.set_title("Mean Degree vs τ\n(Method 5: Specified Average Degree)", fontsize=10)
        ax.set_xlabel("Hard Threshold (τ)")
        ax.set_ylabel("Mean Degree")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # ── Row 2: Giant component ──────────────────────────────────
        ax = fig.add_subplot(gs[2, col])
        ax.plot(df["tau"], df["giant_pct"], color=color, lw=2)
        for pct_target, c in [(80, GREEN), (60, AMBER), (40, RED)]:
            tau_target = select_threshold_giant_pct(df, target_pct=pct_target)
            ax.axhline(pct_target, color=c, linestyle="--", lw=1.2, alpha=0.8)
            ax.axvline(tau_target, color=c, linestyle="--", lw=1.2, alpha=0.8,
                       label=f"{pct_target}% → τ={tau_target:.2f}")
        ax.set_title("Giant Component vs τ\n(Method 3: Giant Component Size)", fontsize=10)
        ax.set_xlabel("Hard Threshold (τ)")
        ax.set_ylabel("% Genes in Giant Component")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # ── Row 3: Degree entropy ───────────────────────────────────
        ax = fig.add_subplot(gs[3, col])
        ax.plot(df["tau"], df["deg_entropy"], color=color, lw=2)
        tau_entropy = select_threshold_entropy(df)
        ax.axvline(tau_entropy, color=RED, linestyle="--", lw=2,
                   label=f"Max entropy → τ={tau_entropy:.2f}")
        ax.set_title("Degree Entropy vs τ\n(Method 2: Maximum Entropy)", fontsize=10)
        ax.set_xlabel("Hard Threshold (τ)")
        ax.set_ylabel("Shannon Entropy (bits)")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # ── Row 4: Power-law R² ─────────────────────────────────────
        ax = fig.add_subplot(gs[4, col])
        ax.plot(df["tau"], df["powerlaw_r2"], color=color, lw=2)
        tau_powerlaw = select_threshold_powerlaw(df, min_giant_pct=min_giant_pct)
        ax.axvline(tau_powerlaw, color=RED, linestyle="--", lw=2,
                   label=f"Best power-law → τ={tau_powerlaw:.2f}")
        ax.axhline(0.8, color=SLATE, linestyle=":", lw=1.5, label="R²=0.8")
        ax.set_title("Power-Law R² vs τ\n(Method 4: Power-Law Degree Dist.)", fontsize=10)
        ax.set_xlabel("Hard Threshold (τ)")
        ax.set_ylabel("Scale-Free R²")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # ── Row 5: Clustering coefficient ──────────────────────────
        ax = fig.add_subplot(gs[5, col])
        ax.plot(df["tau"], df["clustering"], color=color, lw=2)
        ax.set_title("Clustering Coefficient vs τ", fontsize=10)
        ax.set_xlabel("Hard Threshold (τ)")
        ax.set_ylabel("Avg Clustering Coefficient")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)

        # ── Row 6: Summary table ────────────────────────────────────
        ax = fig.add_subplot(gs[6, col])
        ax.axis("off")
        tau_pl   = select_threshold_powerlaw(df, min_giant_pct=min_giant_pct)
        tau_ent  = select_threshold_entropy(df)
        tau_g80  = select_threshold_giant_pct(df, 80)
        tau_g60  = select_threshold_giant_pct(df, 60)
        tau_d100 = select_threshold_avg_degree(df, 100)

        table_data = [
            ["Method",           "τ",              "Criterion"],
            ["Power-law (rec.)", f"{tau_pl:.2f}",  "Best scale-free fit"],
            ["Max entropy",      f"{tau_ent:.2f}", "Max info content"],
            ["Giant comp. 80%",  f"{tau_g80:.2f}", "80% genes connected"],
            ["Giant comp. 60%",  f"{tau_g60:.2f}", "60% genes connected"],
            ["Avg degree ≈100",  f"{tau_d100:.2f}", "~100 edges/gene"],
        ]

        table = ax.table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            loc="center",
            cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)

        for j in range(3):
            table[1, j].set_facecolor("#D1FAE5")
            table[1, j].set_text_props(fontweight="bold")

        ax.set_title(f"Recommended Thresholds for {name}", fontsize=10, fontweight="bold")

    plt.savefig(out_dir / "fig1_threshold_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → %s", out_dir / "fig1_threshold_dashboard.png")


# ══════════════════════════════════════════════════════
# FIGURE 2: Cross-metric comparison
# ══════════════════════════════════════════════════════
def plot_cross_metric_comparison(
    metrics_list: list[pd.DataFrame],
    names: list[str],
    out_dir: Path,
) -> None:
    """Overlay all 3 metrics on the same axes for direct comparison."""
    logger.info("Plotting cross-metric comparison...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Cross-Metric Threshold Comparison\n"
        "Pearson vs Spearman vs Proportionality",
        fontsize=13, fontweight="bold"
    )

    plot_configs = [
        ("pct_edges",   "% Pairs Retained",          "Network Density"),
        ("mean_degree", "Mean Degree",                "Mean Degree"),
        ("giant_pct",   "% Genes in Giant Component", "Giant Component Size"),
        ("deg_entropy", "Shannon Entropy (bits)",      "Degree Entropy"),
        ("powerlaw_r2", "Scale-Free R²",              "Power-Law Fit (R²)"),
        ("clustering",  "Avg Clustering Coefficient", "Clustering Coefficient"),
    ]

    for ax, (col, ylabel, title) in zip(axes.flat, plot_configs):
        for df, name, color in zip(metrics_list, names, COLORS):
            ax.plot(df["tau"], df[col], color=color, lw=2, label=name)
        ax.set_xlabel("Hard Threshold (τ)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        if col == "powerlaw_r2":
            ax.axhline(0.8, color=SLATE, linestyle=":", lw=1.5, label="R²=0.8")
            ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(out_dir / "fig2_cross_metric_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → %s", out_dir / "fig2_cross_metric_comparison.png")


# ══════════════════════════════════════════════════════
# FIGURE 3: Degree distribution at recommended threshold
# ══════════════════════════════════════════════════════
def plot_degree_distributions(
    sim_list: list[pd.DataFrame],
    names: list[str],
    out_dir: Path,
    metrics_list: list[pd.DataFrame],
    min_giant_pct: float = 50.0,
) -> None:
    """
    Plot degree distribution at the recommended (power-law) threshold
    for each metric, with power-law fit overlaid.
    """
    logger.info("Plotting degree distributions at recommended thresholds...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Degree Distribution at Recommended Threshold (Power-Law Method)\n"
        "Log-log scale — straight line indicates scale-free topology",
        fontsize=13, fontweight="bold"
    )

    for ax, sim, df, name, color in zip(axes, sim_list, metrics_list, names, COLORS):
        tau = select_threshold_powerlaw(df, min_giant_pct=min_giant_pct)

        adj    = (sim >= tau).astype(int)
        np.fill_diagonal(adj.values, 0)
        degree = adj.values.sum(axis=1)
        degree = degree[degree > 0]

        counts, bin_edges = np.histogram(degree, bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mask = counts > 0

        ax.scatter(bin_centers[mask], counts[mask] / counts[mask].sum(),
                   color=color, alpha=0.7, s=30, zorder=3)

        if mask.sum() >= 3:
            slope, intercept, r_value, _, _ = stats.linregress(
                np.log10(bin_centers[mask]),
                np.log10(counts[mask] / counts[mask].sum())
            )
            x_fit = np.logspace(
                np.log10(bin_centers[mask].min()),
                np.log10(bin_centers[mask].max()), 100
            )
            y_fit = 10 ** (intercept + slope * np.log10(x_fit))
            ax.plot(x_fit, y_fit, color=NAVY, lw=2, linestyle="--",
                    label=f"Power-law fit\nγ={abs(slope):.2f}, R²={r_value**2:.3f}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Degree (k)", fontsize=11)
        ax.set_ylabel("P(k)", fontsize=11)
        ax.set_title(f"{name}\nτ = {tau:.2f}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which="both")
        ax.text(0.05, 0.05,
                f"n genes = {len(degree):,}\nmean degree = {degree.mean():.1f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_dir / "fig3_degree_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → %s", out_dir / "fig3_degree_distributions.png")


# ══════════════════════════════════════════════════════
# FIGURE 4: Threshold summary heatmap
# ══════════════════════════════════════════════════════
def plot_threshold_summary_heatmap(
    metrics_list: list[pd.DataFrame],
    names: list[str],
    out_dir: Path,
    min_giant_pct: float = 50.0,
) -> dict:
    """
    Heatmap showing recommended thresholds per method per metric.
    Based on Couto et al. (2017) Fig 8.
    """
    logger.info("Plotting threshold summary heatmap...")

    methods = [
        "Power-law (rec.)",
        "Max entropy",
        "Giant comp. 80%",
        "Giant comp. 60%",
        "Giant comp. 40%",
        "Avg degree ≈100",
    ]

    recommended = {}
    data = []
    for df, name in zip(metrics_list, names):
        row = [
            select_threshold_powerlaw(df, min_giant_pct=min_giant_pct),
            select_threshold_entropy(df),
            select_threshold_giant_pct(df, 80),
            select_threshold_giant_pct(df, 60),
            select_threshold_giant_pct(df, 40),
            select_threshold_avg_degree(df, 100),
        ]
        data.append(row)
        recommended[name] = row[0]

    data_arr = np.array(data)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        "Recommended Thresholds by Method and Metric\n"
        "Green = Power-law (recommended per Couto et al. 2017)",
        fontsize=12, fontweight="bold"
    )

    im = ax.imshow(data_arr, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Threshold value τ")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)

    for i in range(len(names)):
        for j in range(len(methods)):
            val = data_arr[i, j]
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center", fontsize=11,
                    fontweight="bold" if j == 0 else "normal",
                    color="black")

    ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, len(names),
                                fill=False, edgecolor=GREEN, lw=3))

    plt.tight_layout()
    plt.savefig(out_dir / "fig4_threshold_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → %s", out_dir / "fig4_threshold_heatmap.png")

    return recommended


# ══════════════════════════════════════════════════════
# SAVE THRESHOLDED NETWORKS
# ══════════════════════════════════════════════════════
def save_thresholded_networks(
    sim_list: list[pd.DataFrame],
    names: list[str],
    recommended: dict[str, float],
    out_dir: Path,
) -> None:
    """
    Apply recommended threshold and save adjacency matrices.
    Uses scipy sparse to avoid memory overflow on large matrices.
    """
    networks_dir = out_dir / "networks"
    networks_dir.mkdir(exist_ok=True)

    summary_rows = []
    for sim, name in zip(sim_list, names):
        tau = recommended[name]
        logger.info("Saving thresholded network for %s at τ=%.2f", name, tau)

        # Signed network: only keep positive correlations >= tau
        adj = (sim >= tau).astype(int)
        np.fill_diagonal(adj.values, 0)

        # Save adjacency matrix
        adj_path = networks_dir / f"{name.lower()}_adjacency_tau{tau:.2f}.csv"
        adj.to_csv(adj_path)

        # Network stats using scipy sparse — avoids networkx memory overhead
        adj_sparse = csr_matrix(adj.values)
        n_comp, labels = sp_connected_components(adj_sparse, directed=False)
        giant_size = int(np.bincount(labels).max())
        degree_seq = np.array(adj.values.sum(axis=1)).flatten()
        n_edges    = int(adj.values.sum() / 2)
        density    = n_edges / (sim.shape[0] * (sim.shape[0] - 1) / 2)

        summary_rows.append({
            "metric":       name,
            "threshold":    tau,
            "n_genes":      sim.shape[0],
            "n_edges":      n_edges,
            "density":      density,
            "giant_size":   giant_size,
            "giant_pct":    100 * giant_size / sim.shape[0],
            "mean_degree":  float(degree_seq.mean()),
            "n_components": n_comp,
            "adj_path":     str(adj_path),
        })
        logger.info("  Saved → %s", adj_path)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "network_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Network summary → %s", summary_path)
    print("\n" + "="*60)
    print("NETWORK SUMMARY (Power-Law Threshold)")
    print("="*60)
    print(summary_df.to_string(index=False))


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hard threshold analysis for gene co-expression benchmark."
    )
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Paths to similarity matrices (3 expected).")
    parser.add_argument("--names", nargs="+", required=True,
                        help="Names for each matrix e.g. Pearson Spearman Proportionality.")
    parser.add_argument("--output", required=True,
                        help="Output directory for figures and networks.")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=list(np.arange(0.0, 1.01, 0.05)),
                        help="Threshold values to test (default: 0.0 to 1.0 step 0.05).")
    parser.add_argument("--min-giant-pct", type=float, default=50.0,
                        help="Minimum giant component %% for power-law threshold selection (default: 50).")

    args = parser.parse_args()

    if len(args.inputs) != len(args.names):
        raise ValueError("Number of --inputs must match number of --names.")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = np.array(args.thresholds)

    # Load matrices
    sim_list = [load_matrix(p) for p in args.inputs]

    # Compute metrics for each
    logger.info("Computing threshold metrics for all matrices...")
    metrics_list = []
    for sim, name in zip(sim_list, args.names):
        logger.info("  Processing %s (%d x %d)...", name, sim.shape[0], sim.shape[1])
        df = compute_threshold_metrics(sim, thresholds)
        metrics_list.append(df)
        df.to_csv(out_dir / f"{name.lower()}_threshold_metrics.csv", index=False)

    # Generate figures
    plot_threshold_dashboard(metrics_list, args.names, out_dir,
                             min_giant_pct=args.min_giant_pct)
    plot_cross_metric_comparison(metrics_list, args.names, out_dir)
    plot_degree_distributions(sim_list, args.names, out_dir, metrics_list,
                              min_giant_pct=args.min_giant_pct)
    recommended = plot_threshold_summary_heatmap(metrics_list, args.names, out_dir,
                                                  min_giant_pct=args.min_giant_pct)

    # Save thresholded networks
    save_thresholded_networks(sim_list, args.names, recommended, out_dir)

    logger.info("All outputs saved to: %s", out_dir)
    print(f"\nDone! All figures and networks saved to: {out_dir}")


if __name__ == "__main__":
    main()