"""
Soft Thresholding Network Construction for Gene Co-Expression Benchmark.
Accepts 3 similarity matrices, applies soft threshold (beta power),
and produces adjacency matrices + topology plots.

Memory optimized for large matrices (17,578 x 17,578):
- In-place operations to minimize copies
- Scipy sparse for connected components
- Upper triangle sampling for distributions
- Parquet output instead of CSV

Methods based on:
- Zhang & Horvath (2005) Stat. Appl. Genet. Mol. Biol. 4:1128
- Langfelder & Horvath (2008) BMC Bioinformatics 9:559

Usage:
    python src/soft_threshold_network.py \
        --inputs data/similarity/pearson_similarity.csv \
                 data/similarity/spearman_similarity.csv \
                 data/similarity/rho_similarity.csv \
        --names Pearson Spearman Proportionality \
        --betas 13 11 5 \
        --output results/figures/soft_threshold
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


# ══════════════════════════════════════════════════════
# SOFT THRESHOLD — memory optimized
# ══════════════════════════════════════════════════════
def apply_soft_threshold(sim: pd.DataFrame, beta: int) -> pd.DataFrame:
    """
    Apply soft threshold power to signed similarity matrix.
    adj(i,j) = sign(sim) * |sim|^beta

    In-place operations to minimize memory — only one copy of the matrix.
    Per Zhang & Horvath (2005): signed hybrid approach.
    """
    logger.info("Applying soft threshold beta=%d...", beta)
    vals = sim.values.copy()       # one copy only
    np.fill_diagonal(vals, 0)      # remove self-connections in-place
    sign = np.sign(vals)           # preserve sign
    np.abs(vals, out=vals)         # in-place abs
    np.power(vals, beta, out=vals) # in-place power
    vals *= sign                   # restore sign in-place
    return pd.DataFrame(vals, index=sim.index, columns=sim.columns)


# ══════════════════════════════════════════════════════
# TOPOLOGY METRICS — memory optimized
# ══════════════════════════════════════════════════════
def compute_network_topology(adj: pd.DataFrame, name: str, beta: int) -> dict:
    """
    Compute key topology metrics on soft-thresholded adjacency matrix.
    Uses sparse matrices and avoids networkx for memory efficiency.
    Clustering and betweenness are computed later in network_topology_comparison.py.
    Based on Zhang & Horvath (2005) and Couto et al. (2017).
    """
    logger.info("Computing topology: %s (beta=%d)...", name, beta)

    vals     = adj.values
    abs_vals = np.abs(vals).copy()
    np.fill_diagonal(abs_vals, 0)

    # ── Weighted connectivity ───────────────────────────
    connectivity = abs_vals.sum(axis=1)

    # ── Scale-free topology fit on connectivity ─────────
    k_nonzero = connectivity[connectivity > 0]
    if len(k_nonzero) >= 10:
        counts, bin_edges = np.histogram(k_nonzero, bins=20)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mask = counts > 0
        if mask.sum() >= 3:
            slope, intercept, r_value, _, _ = stats.linregress(
                np.log10(bin_centers[mask]),
                np.log10(counts[mask] / counts[mask].sum())
            )
            sft_r2    = r_value ** 2
            sft_slope = slope
        else:
            sft_r2, sft_slope = 0.0, 0.0
    else:
        sft_r2, sft_slope = 0.0, 0.0

    # ── Giant component — scipy sparse ──────────────────
    binary_sparse = csr_matrix((abs_vals > 0).astype(np.int8))
    n_comp, labels = sp_connected_components(binary_sparse, directed=False)
    giant_size  = int(np.bincount(labels).max())
    giant_pct   = 100 * giant_size / adj.shape[0]

    # ── Degree ──────────────────────────────────────────
    degree_seq  = np.array((abs_vals > 0).sum(axis=1)).flatten()
    mean_degree = float(degree_seq.mean())

    # Free large intermediate arrays
    del abs_vals, binary_sparse

    return {
        "metric":            name,
        "beta":              beta,
        "n_genes":           adj.shape[0],
        "mean_connectivity": float(connectivity.mean()),
        "std_connectivity":  float(connectivity.std()),
        "sft_r2":            sft_r2,
        "sft_slope":         sft_slope,
        "giant_size":        giant_size,
        "giant_pct":         giant_pct,
        "n_components":      n_comp,
        "mean_degree":       mean_degree,
        "connectivity":      connectivity,  # kept for plotting
        "degree_seq":        degree_seq,    # kept for plotting
    }


# ══════════════════════════════════════════════════════
# FIGURE 1: SFT Validation
# ══════════════════════════════════════════════════════
def plot_sft_validation(
    topology_list: list[dict],
    names: list[str],
    out_dir: Path,
) -> None:
    """
    Validate scale-free topology at the chosen beta for each metric.
    Shows connectivity distribution in log-log space.
    """
    logger.info("Plotting SFT validation...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Scale-Free Topology Validation at Recommended β\n"
        "Log-log connectivity distribution — straight line = scale-free",
        fontsize=13, fontweight="bold"
    )

    for ax, topo, name, color in zip(axes, topology_list, names, COLORS):
        k = topo["connectivity"]
        k_nonzero = k[k > 0]

        counts, bin_edges = np.histogram(k_nonzero, bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mask = counts > 0

        ax.scatter(bin_centers[mask], counts[mask] / counts[mask].sum(),
                   color=color, alpha=0.7, s=30, zorder=3, label="Observed")

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
        ax.set_xlabel("Connectivity (k)", fontsize=11)
        ax.set_ylabel("P(k)", fontsize=11)
        ax.set_title(
            f"{name}  |  β = {topo['beta']}\n"
            f"SFT R² = {topo['sft_r2']:.3f}",
            fontsize=11, fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which="both")
        ax.text(0.05, 0.05,
                f"mean k = {topo['mean_connectivity']:.1f}\n"
                f"n genes = {topo['n_genes']:,}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_dir / "fig1_sft_validation.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → fig1_sft_validation.png")


# ══════════════════════════════════════════════════════
# FIGURE 2: Connectivity distribution comparison
# ══════════════════════════════════════════════════════
def plot_connectivity_comparison(
    topology_list: list[dict],
    names: list[str],
    out_dir: Path,
) -> None:
    """Overlay connectivity distributions for all 3 metrics."""
    logger.info("Plotting connectivity comparison...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Weighted Connectivity Distribution Comparison\n"
        "After Soft Thresholding at Recommended β",
        fontsize=13, fontweight="bold"
    )

    ax = axes[0]
    for topo, name, color in zip(topology_list, names, COLORS):
        k = topo["connectivity"]
        ax.hist(k, bins=50, color=color, alpha=0.5, label=name, density=True)
        ax.axvline(k.mean(), color=color, linestyle="--", lw=1.5)
    ax.set_xlabel("Weighted Connectivity", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Full Distribution", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for topo, name, color in zip(topology_list, names, COLORS):
        k = topo["connectivity"]
        k_nonzero = k[k > 0]
        counts, bin_edges = np.histogram(k_nonzero, bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mask = counts > 0
        ax.scatter(bin_centers[mask], counts[mask] / counts[mask].sum(),
                   color=color, alpha=0.6, s=20,
                   label=f"{name} (β={topo['beta']})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Connectivity (k)", fontsize=11)
    ax.set_ylabel("P(k)", fontsize=11)
    ax.set_title("Log-Log Overlay", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(out_dir / "fig2_connectivity_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → fig2_connectivity_comparison.png")


# ══════════════════════════════════════════════════════
# FIGURE 3: Topology summary
# ══════════════════════════════════════════════════════
def plot_topology_summary(
    topology_list: list[dict],
    names: list[str],
    out_dir: Path,
) -> None:
    """Bar chart comparison of key topology metrics across all 3 networks."""
    logger.info("Plotting topology summary...")

    metrics = [
        ("sft_r2",            "Scale-Free R²"),
        ("mean_connectivity", "Mean Connectivity"),
        ("giant_pct",         "Giant Component (%)"),
        ("mean_degree",       "Mean Degree"),
        ("n_components",      "Number of Components"),
        ("std_connectivity",  "Connectivity Std Dev"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "Network Topology Summary\n"
        "Soft-Thresholded Networks at Recommended β",
        fontsize=13, fontweight="bold"
    )

    for ax, (metric, label) in zip(axes.flat, metrics):
        values = [topo[metric] for topo in topology_list]
        bars   = ax.bar(names, values, color=COLORS, alpha=0.85,
                        edgecolor="white", lw=1.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    f"{val:.3f}" if val < 10 else f"{val:,.0f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(
            [f"{n}\n(β={topo['beta']})" for n, topo in zip(names, topology_list)],
            fontsize=9
        )

        if metric == "sft_r2":
            ax.axhline(0.8, color=RED, linestyle="--", lw=1.5, label="R²=0.8")
            ax.legend(fontsize=9)
            ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(out_dir / "fig3_topology_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → fig3_topology_summary.png")


# ══════════════════════════════════════════════════════
# FIGURE 4: Adjacency weight distribution — sampled
# ══════════════════════════════════════════════════════
def plot_adjacency_distribution(
    adj_list: list[pd.DataFrame],
    names: list[str],
    betas: list[int],
    out_dir: Path,
    sample_size: int = 1_000_000,
) -> None:
    """
    Distribution of adjacency weights after soft thresholding.
    Samples upper triangle only to avoid loading 309M values into memory.
    """
    logger.info("Plotting adjacency weight distributions (sampled)...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Adjacency Weight Distribution After Soft Thresholding\n"
        f"Sampled from upper triangle (n={sample_size:,} pairs)",
        fontsize=13, fontweight="bold"
    )

    for ax, adj, name, beta, color in zip(axes, adj_list, names, betas, COLORS):
        n = adj.shape[0]
        tri_idx    = np.triu_indices(n, k=1)
        total_pairs = len(tri_idx[0])
        n_sample   = min(sample_size, total_pairs)

        idx  = np.random.choice(total_pairs, n_sample, replace=False)
        vals = adj.values[tri_idx[0][idx], tri_idx[1][idx]]
        vals = vals[vals != 0]

        if len(vals) == 0:
            ax.set_title(f"{name}  |  β = {beta}\nNo non-zero values", fontsize=11)
            continue

        ax.hist(vals, bins=100, color=color, alpha=0.8, edgecolor="none")
        ax.axvline(vals.mean(), color=NAVY, linestyle="--", lw=2,
                   label=f"Mean = {vals.mean():.3f}")
        ax.axvline(np.median(vals), color=RED, linestyle="--", lw=2,
                   label=f"Median = {np.median(vals):.3f}")
        ax.set_xlabel("Adjacency Weight", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{name}  |  β = {beta}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        stats_text = (
            f"min = {vals.min():.4f}\n"
            f"max = {vals.max():.4f}\n"
            f"% > 0.1 = {100*(vals > 0.1).mean():.1f}%\n"
            f"% > 0.5 = {100*(vals > 0.5).mean():.1f}%"
        )
        ax.text(0.97, 0.97, stats_text,
                transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_dir / "fig4_adjacency_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → fig4_adjacency_distribution.png")


# ══════════════════════════════════════════════════════
# SAVE NETWORKS
# ══════════════════════════════════════════════════════
def save_networks(
    adj_list: list[pd.DataFrame],
    topology_list: list[dict],
    names: list[str],
    betas: list[int],
    out_dir: Path,
) -> None:
    """
    Save adjacency matrices as parquet (~10x smaller than CSV)
    and topology summary CSV.
    """
    networks_dir = out_dir / "networks"
    networks_dir.mkdir(exist_ok=True)

    summary_rows = []
    for adj, topo, name, beta in zip(adj_list, topology_list, names, betas):
        adj_path = networks_dir / f"{name.lower()}_soft_adj_beta{beta}.parquet"
        adj.to_parquet(adj_path)
        logger.info("Saved adjacency → %s", adj_path)

        row = {k: v for k, v in topo.items()
               if k not in {"connectivity", "degree_seq"}}
        row["adj_path"] = str(adj_path)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "soft_network_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "="*60)
    print("SOFT-THRESHOLDED NETWORK SUMMARY")
    print("="*60)
    cols = ["metric", "beta", "n_genes", "mean_connectivity",
            "sft_r2", "giant_pct", "mean_degree", "n_components"]
    print(summary_df[cols].to_string(index=False))
    logger.info("Summary → %s", summary_path)


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Soft threshold network construction for co-expression benchmark."
    )
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Paths to similarity matrices.")
    parser.add_argument("--names", nargs="+", required=True,
                        help="Names e.g. Pearson Spearman Proportionality.")
    parser.add_argument("--betas", nargs="+", type=int, required=True,
                        help="Recommended beta power per matrix (from SFT analysis).")
    parser.add_argument("--output", required=True,
                        help="Output directory for figures and networks.")
    parser.add_argument("--sample-size", type=int, default=1_000_000,
                        help="Pairs to sample for adjacency distribution plot (default: 1M).")

    args = parser.parse_args()

    if not (len(args.inputs) == len(args.names) == len(args.betas)):
        raise ValueError("--inputs, --names, and --betas must have the same number of values.")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load matrices
    sim_list = [load_matrix(p) for p in args.inputs]

    # Apply soft threshold — in-place ops, minimal memory
    logger.info("Applying soft thresholds...")
    adj_list = []
    for sim, name, beta in zip(sim_list, args.names, args.betas):
        logger.info("  %s: β = %d", name, beta)
        adj = apply_soft_threshold(sim, beta)
        adj_list.append(adj)
        del sim  # free original after thresholding
    del sim_list

    # Compute topology
    topology_list = []
    for adj, name, beta in zip(adj_list, args.names, args.betas):
        topo = compute_network_topology(adj, name, beta)
        topology_list.append(topo)

    # Figures
    plot_sft_validation(topology_list, args.names, out_dir)
    plot_connectivity_comparison(topology_list, args.names, out_dir)
    plot_topology_summary(topology_list, args.names, out_dir)
    plot_adjacency_distribution(adj_list, args.names, args.betas, out_dir,
                                sample_size=args.sample_size)

    # Save
    save_networks(adj_list, topology_list, args.names, args.betas, out_dir)

    logger.info("All outputs saved to: %s", out_dir)
    print(f"\nDone! All figures and networks saved to: {out_dir}")


if __name__ == "__main__":
    main()