"""
Similarity matrix visualization for gene co-expression benchmarking.
Modular design — accepts any similarity matrix as input.

Usage:
    python src/visualization_similarity.py \
        --input data/similarity/pearson_similarity.csv \
        --name Pearson \
        --output results/figures/pearson
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Shared style ───────────────────────────────────────
NAVY  = "#1B2A4A"
TEAL  = "#0D9488"
SLATE = "#64748B"
RED   = "#EF4444"
AMBER = "#F59E0B"
GREEN = "#10B981"


# ══════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════
def load_similarity_matrix(input_path: str | Path) -> pd.DataFrame:
    input_path = Path(input_path)
    logger.info("Loading similarity matrix from: %s", input_path)

    if input_path.suffix.lower() == ".parquet":
        sim = pd.read_parquet(input_path)
    elif input_path.suffix.lower() in {".csv"}:
        sim = pd.read_csv(input_path, index_col=0)
    elif input_path.suffix.lower() in {".tsv", ".txt"}:
        sim = pd.read_csv(input_path, sep="\t", index_col=0)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    logger.info("Loaded: %d genes x %d genes", sim.shape[0], sim.shape[1])
    return sim


def get_upper_triangle(sim: pd.DataFrame) -> np.ndarray:
    """Extract upper triangle values (no diagonal)."""
    vals = sim.values
    return vals[np.triu_indices_from(vals, k=1)]


# ══════════════════════════════════════════════════════
# FIGURE 1: Distribution of Pairwise Correlations
# ══════════════════════════════════════════════════════
def plot_correlation_distribution(
    sim: pd.DataFrame,
    metric_name: str,
    out_dir: Path,
) -> None:
    """
    Plot histogram of all pairwise correlation values.
    Justified by: Johnson & Krishnan (2022), Ballouz et al. (2015)
    """
    logger.info("Plotting correlation distribution...")
    tri = get_upper_triangle(sim)
    valid = tri[~np.isnan(tri)]

    mean_val   = np.mean(valid)
    median_val = np.median(valid)
    std_val    = np.std(valid)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"{metric_name} — Distribution of Pairwise Correlations\n"
        f"{sim.shape[0]:,} genes  |  {len(valid):,} gene pairs",
        fontsize=13, fontweight="bold"
    )

    # Left: full distribution
    ax = axes[0]
    ax.hist(valid, bins=100, color=TEAL, edgecolor="none", alpha=0.85)
    ax.axvline(mean_val,   color=RED,   linestyle="--", lw=1.5,
               label=f"Mean = {mean_val:.3f}")
    ax.axvline(median_val, color=AMBER, linestyle="--", lw=1.5,
               label=f"Median = {median_val:.3f}")
    ax.set_xlabel(f"{metric_name} Correlation", fontsize=11)
    ax.set_ylabel("Number of Gene Pairs", fontsize=11)
    ax.set_title("Full Distribution", fontsize=12)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x/1e6)}M" if x >= 1e6 else f"{int(x/1e3)}K")
    )

    # Right: positive tail with threshold candidates
    ax = axes[1]
    pos = valid[valid > 0]
    ax.hist(pos, bins=100, color=NAVY, edgecolor="none", alpha=0.85)
    for tau, color, ls in [(0.5, RED, "--"), (0.7, AMBER, "--"), (0.8, GREEN, "--")]:
        pct = 100 * (valid > tau).mean()
        ax.axvline(tau, color=color, linestyle=ls, lw=1.5,
                   label=f"τ={tau} → {pct:.1f}% pairs")
    ax.set_xlabel(f"{metric_name} Correlation", fontsize=11)
    ax.set_ylabel("Number of Gene Pairs", fontsize=11)
    ax.set_title("Positive Tail — Threshold Candidates", fontsize=12)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x/1e6)}M" if x >= 1e6 else f"{int(x/1e3)}K")
    )

    # Stats annotation
    stats_text = (
        f"n pairs = {len(valid):,}\n"
        f"mean = {mean_val:.3f}\n"
        f"median = {median_val:.3f}\n"
        f"std = {std_val:.3f}\n"
        f"min = {valid.min():.3f}\n"
        f"max = {valid.max():.3f}"
    )
    axes[0].text(
        0.02, 0.97, stats_text,
        transform=axes[0].transAxes,
        fontsize=8, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()
    out_path = out_dir / "fig1_correlation_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → %s", out_path)


# ══════════════════════════════════════════════════════
# FIGURE 2: Gene-Gene Correlation Heatmap
# ══════════════════════════════════════════════════════
def plot_gene_heatmap(
    sim: pd.DataFrame,
    metric_name: str,
    out_dir: Path,
    n_top: int = 100,
) -> None:
    """
    Clustered gene-gene heatmap of top N most variable genes.
    Justified by: Langfelder & Horvath (2008), Johnson & Krishnan (2022)
    """
    logger.info("Plotting gene-gene heatmap (top %d genes)...", n_top)

    # Select top N genes by variance of their similarity scores
    # Load expression matrix to get gene variances
    expr = pd.read_csv("data/processed/preprocessed_counts.csv", index_col=0)
    gene_std = expr.std(axis=1)
    top_genes = gene_std.nlargest(n_top).index

    # Then subset similarity matrix
    sub = sim.loc[top_genes, top_genes]

    fig = sns.clustermap(
        sub,
        cmap="RdBu_r",
        center=0, vmin=-1, vmax=1,
        xticklabels=False,
        yticklabels=False,
        figsize=(10, 10),
        cbar_kws={"label": f"{metric_name} r", "shrink": 0.6},
        dendrogram_ratio=0.12,
        colors_ratio=0.02,
    )
    fig.ax_heatmap.set_title(
        f"{metric_name} — Gene Co-Expression Heatmap\n"
        f"Top {n_top} Most Variable Genes",
        fontsize=13, pad=20
    )
    fig.ax_heatmap.set_xlabel("Genes", fontsize=11)
    fig.ax_heatmap.set_ylabel("Genes", fontsize=11)

    out_path = out_dir / "fig2_gene_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → %s", out_path)


# ══════════════════════════════════════════════════════
# FIGURE 3: Scale-Free Topology Fit (Soft Thresholding)
# ══════════════════════════════════════════════════════
def plot_scale_free_topology(
    sim: pd.DataFrame,
    metric_name: str,
    out_dir: Path,
    powers: list[int] | None = None,
    r2_threshold: float = 0.80,
) -> dict:
    """
    Plot scale-free topology R² and mean connectivity vs beta power.
    Justified by: Zhang & Horvath (2005), Langfelder & Horvath (2008)

    Returns dict with recommended beta power.
    """
    if powers is None:
        powers = list(range(1, 21))

    logger.info("Computing scale-free topology fit for %d powers...", len(powers))

    # Use absolute values for signed hybrid network
    abs_sim = sim.abs().values
    np.fill_diagonal(abs_sim, 0)  # remove self-connections

    results = []
    for beta in powers:
        logger.info("  Testing beta = %d...", beta)

        # Apply soft threshold: raise to power beta
        adj = abs_sim ** beta

        # Connectivity = sum of edge weights per gene
        k = adj.sum(axis=1)
        mean_k = np.mean(k)

        # Scale-free fit: log(P(k)) ~ log(k)
        # Bin connectivity values and fit linear regression
        k_nonzero = k[k > 0]
        if len(k_nonzero) < 10:
            results.append({"power": beta, "r2": 0.0, "slope": 0.0, "mean_k": mean_k})
            continue

        # Create histogram of connectivity
        counts, bin_edges = np.histogram(k_nonzero, bins=20)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Keep only bins with non-zero counts
        mask = counts > 0
        if mask.sum() < 3:
            results.append({"power": beta, "r2": 0.0, "slope": 0.0, "mean_k": mean_k})
            continue

        log_k  = np.log10(bin_centers[mask])
        log_pk = np.log10(counts[mask] / counts[mask].sum())

        # Linear regression
        slope, intercept, r_value, p_value, se = stats.linregress(log_k, log_pk)
        r2 = r_value ** 2

        results.append({
            "power":  beta,
            "r2":     r2,
            "slope":  slope,
            "mean_k": mean_k
        })

    df_sft = pd.DataFrame(results)

    # Find recommended power
    above_threshold = df_sft[df_sft["r2"] >= r2_threshold]
    if len(above_threshold) > 0:
        recommended_power = int(above_threshold.iloc[0]["power"])
        logger.info("Recommended soft threshold power: β = %d (R² = %.3f)",
                    recommended_power,
                    above_threshold.iloc[0]["r2"])
    else:
        recommended_power = int(df_sft.loc[df_sft["r2"].idxmax(), "power"])
        logger.warning("R² never exceeded %.2f. Best power: β = %d (R² = %.3f)",
                       r2_threshold, recommended_power,
                       df_sft["r2"].max())

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"{metric_name} — Scale-Free Topology Fit\n"
        f"Recommended β = {recommended_power}  "
        f"(R² threshold = {r2_threshold})",
        fontsize=13, fontweight="bold"
    )

    # Left: R² vs power
    ax = axes[0]
    ax.plot(df_sft["power"], df_sft["r2"],
            color=TEAL, marker="o", linewidth=2, markersize=6)
    ax.axhline(r2_threshold, color=RED, linestyle="--", lw=1.5,
               label=f"R² threshold = {r2_threshold}")
    ax.axvline(recommended_power, color=NAVY, linestyle="--", lw=1.5,
               label=f"Selected β = {recommended_power}")
    ax.set_xlabel("Soft Threshold Power (β)", fontsize=11)
    ax.set_ylabel("Scale-Free Topology Fit R²", fontsize=11)
    ax.set_title("R² vs Soft Threshold Power", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right: Mean connectivity vs power
    ax = axes[1]
    ax.plot(df_sft["power"], df_sft["mean_k"],
            color=NAVY, marker="o", linewidth=2, markersize=6)
    ax.axvline(recommended_power, color=RED, linestyle="--", lw=1.5,
               label=f"Selected β = {recommended_power}")
    ax.set_xlabel("Soft Threshold Power (β)", fontsize=11)
    ax.set_ylabel("Mean Connectivity", fontsize=11)
    ax.set_title("Mean Connectivity vs Soft Threshold Power", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Annotate recommended point on both plots
    r2_at_rec = df_sft.loc[df_sft["power"] == recommended_power, "r2"].values[0]
    k_at_rec  = df_sft.loc[df_sft["power"] == recommended_power, "mean_k"].values[0]
    axes[0].annotate(
        f"β={recommended_power}\nR²={r2_at_rec:.3f}",
        xy=(recommended_power, r2_at_rec),
        xytext=(recommended_power + 1, r2_at_rec - 0.1),
        fontsize=9, color=NAVY,
        arrowprops=dict(arrowstyle="->", color=NAVY)
    )
    axes[1].annotate(
        f"β={recommended_power}\nk={k_at_rec:.1f}",
        xy=(recommended_power, k_at_rec),
        xytext=(recommended_power + 1, k_at_rec + k_at_rec * 0.1),
        fontsize=9, color=NAVY,
        arrowprops=dict(arrowstyle="->", color=NAVY)
    )

    plt.tight_layout()
    out_path = out_dir / "fig3_scale_free_topology.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → %s", out_path)

    # Save SFT table
    table_path = out_dir / "sft_table.csv"
    df_sft.to_csv(table_path, index=False)
    logger.info("Saved SFT table → %s", table_path)

    return {"recommended_power": recommended_power, "sft_table": df_sft}


# ══════════════════════════════════════════════════════
# FIGURE 4: Network Density vs Hard Threshold
# ══════════════════════════════════════════════════════
def plot_network_density(
    sim: pd.DataFrame,
    metric_name: str,
    out_dir: Path,
) -> dict:
    """
    Plot % pairs retained at each hard threshold τ.
    Justified by: Perkins & Langston (2009), Bleker et al. (2024)
    """
    logger.info("Computing network density vs hard threshold...")
    tri = get_upper_triangle(sim)
    valid = tri[~np.isnan(tri)]
    abs_valid = np.abs(valid)

    thresholds = np.arange(0, 1.01, 0.01)
    pct_retained = [100 * (abs_valid > t).mean() for t in thresholds]

    # Build threshold table
    table_rows = []
    for tau in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        pct = 100 * (abs_valid > tau).mean()
        n_edges = int((abs_valid > tau).sum())
        table_rows.append({"tau": tau, "pct_retained": pct, "n_edges": n_edges})
    df_table = pd.DataFrame(table_rows)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"{metric_name} — Network Density vs Hard Threshold",
        fontsize=13, fontweight="bold"
    )

    # Left: full range
    ax = axes[0]
    ax.plot(thresholds, pct_retained, color=RED, linewidth=2)
    ax.set_xlabel("Hard Threshold (τ)", fontsize=11)
    ax.set_ylabel("% of Pairs Retained", fontsize=11)
    ax.set_title("Network Density vs τ (Full Range)", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)

    # Right: zoomed into 0.5–1.0 range
    ax = axes[1]
    mask = thresholds >= 0.5
    ax.plot(thresholds[mask], np.array(pct_retained)[mask],
            color=NAVY, linewidth=2)
    ax.set_xlabel("Hard Threshold (τ)", fontsize=11)
    ax.set_ylabel("% of Pairs Retained", fontsize=11)
    ax.set_title("Network Density vs τ (Zoomed: 0.5–1.0)", fontsize=12)
    ax.set_xlim(0.5, 1.0)
    ax.grid(alpha=0.3)

    # Add threshold table as text
    table_str = "  τ    | % kept |   edges\n"
    table_str += "-------|--------|--------\n"
    for _, row in df_table.iterrows():
        table_str += f" {row['tau']:.2f}  | {row['pct_retained']:5.2f}% | {int(row['n_edges']):,}\n"
    axes[1].text(
        0.97, 0.97, table_str,
        transform=axes[1].transAxes,
        fontsize=7.5, verticalalignment="top", horizontalalignment="right",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    plt.tight_layout()
    out_path = out_dir / "fig4_network_density.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → %s", out_path)

    # Save density table
    table_path = out_dir / "density_table.csv"
    df_table.to_csv(table_path, index=False)
    logger.info("Saved density table → %s", table_path)

    return {"density_table": df_table}
# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a gene co-expression similarity matrix."
    )
    parser.add_argument("--input",   required=True,
                        help="Path to similarity matrix (CSV, TSV, or Parquet).")
    parser.add_argument("--name",    required=True,
                        help="Name of the metric e.g. Pearson, Spearman, Proportionality.")
    parser.add_argument("--output",  required=True,
                        help="Output directory for figures.")
    parser.add_argument("--n-top",   type=int, default=100,
                        help="Number of top variable genes for heatmap (default: 100).")
    parser.add_argument("--powers",  type=int, nargs="+", default=list(range(2, 21)),
                        help="Beta powers to test for scale-free topology (default: 2-20).")
    parser.add_argument("--r2",      type=float, default=0.80,
                        help="R² threshold for scale-free topology (default: 0.80).")
    parser.add_argument("--density-min", type=float, default=0.01,
                        help="Minimum target network density (default: 0.01 = 1%%).")
    parser.add_argument("--density-max", type=float, default=0.05,
                        help="Maximum target network density (default: 0.05 = 5%%).")
    parser.add_argument("--skip-sft", action="store_true",
                        help="Skip scale-free topology plot (slow for large matrices).")

    args = parser.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    sim = load_similarity_matrix(args.input)

    # Run all figures
    plot_correlation_distribution(sim, args.name, out_dir)
    plot_gene_heatmap(sim, args.name, out_dir, n_top=args.n_top)
    plot_network_density(sim, args.name, out_dir)

    if not args.skip_sft:
        plot_scale_free_topology(sim, args.name, out_dir,
                                 powers=args.powers,
                                 r2_threshold=args.r2)

    logger.info("All figures saved to: %s", out_dir)


if __name__ == "__main__":
    main()