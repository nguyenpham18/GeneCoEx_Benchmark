"""
Network Topology Comparison for Gene Co-Expression Benchmark.
Compares topology properties across 6 networks:
    - 3 metrics (Pearson, Spearman, Proportionality)
    - 2 thresholding methods (Hard, Soft)

Properties based on:
- Couto et al. (2017) Mol. BioSyst. 13:2024-2035
- Zhang & Horvath (2005) Stat. Appl. Genet. Mol. Biol. 4:1128
- Barabasi & Oltvai (2004) Nat. Rev. Genet. 5:101-113

Usage:
    python src/network_topology_comparison.py \
        --hard-inputs  results/figures/hard_threshold/networks/pearson_adjacency_tau0.70.csv \
                       results/figures/hard_threshold/networks/spearman_adjacency_tau0.70.csv \
                       results/figures/hard_threshold/networks/proportionality_adjacency_tau0.70.csv \
        --soft-inputs  results/figures/soft_threshold/networks/pearson_soft_adj_beta13.csv \
                       results/figures/soft_threshold/networks/spearman_soft_adj_beta11.csv \
                       results/figures/soft_threshold/networks/proportionality_soft_adj_beta5.csv \
        --names Pearson Spearman Proportionality \
        --output results/figures/topology_comparison
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from scipy import stats

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

METRIC_COLORS  = [TEAL, RED, PURPLE]       # Pearson, Spearman, Proportionality
METHOD_COLORS  = [NAVY, AMBER]             # Hard, Soft
METHOD_HATCHES = ["", "///"]


# ══════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════
def load_matrix(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    logger.info("Loading: %s", path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix == ".csv":
        return pd.read_csv(path, index_col=0)
    elif path.suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", index_col=0)
    raise ValueError(f"Unsupported format: {path.suffix}")


# ══════════════════════════════════════════════════════
# TOPOLOGY COMPUTATION
# ══════════════════════════════════════════════════════
def compute_topology(adj: pd.DataFrame, name: str, method: str) -> dict:
    """
    Compute all topology properties from Couto et al. (2017):
    - Degree statistics
    - Local transitivity / clustering coefficient
    - Betweenness centrality
    - Giant component size
    - Number of edges
    - Degree assortativity
    - Average shortest path length (on giant component)
    - Diameter (on giant component)
    - Modularity
    - Scale-free R²
    """
    logger.info("Computing topology: %s (%s)...", name, method)

    # For soft-thresholded (weighted) networks, binarize for graph metrics
    vals        = adj.values.copy()
    abs_vals    = np.abs(vals)
    np.fill_diagonal(abs_vals, 0)

    # Binarize: edge exists if weight > 0
    binary      = (abs_vals > 0).astype(int)
    n_genes     = adj.shape[0]

    G = nx.from_numpy_array(binary)

    # ── Degree ─────────────────────────────────────────
    degree_seq  = np.array([d for _, d in G.degree()])

    # ── Giant component ─────────────────────────────────
    components  = sorted(nx.connected_components(G), key=len, reverse=True)
    giant_nodes = components[0] if components else set()
    giant_size  = len(giant_nodes)
    giant_pct   = 100 * giant_size / n_genes
    n_comp      = nx.number_connected_components(G)
    giant_G     = G.subgraph(giant_nodes).copy()

    # ── Clustering coefficient ──────────────────────────
    clustering_dict = nx.clustering(giant_G)
    clustering_vals = np.array(list(clustering_dict.values()))

    # ── Betweenness centrality (sampled for speed on large graphs) ──
    if giant_size > 2000:
        k_sample = min(500, giant_size)
        between_dict = nx.betweenness_centrality(giant_G, k=k_sample, normalized=True)
    else:
        between_dict = nx.betweenness_centrality(giant_G, normalized=True)
    between_vals = np.array(list(between_dict.values()))

    # ── Shortest path & diameter (sampled for large graphs) ─────────
    if giant_size > 1000:
        # Sample nodes for efficiency
        sample_nodes = list(giant_nodes)[:500]
        path_lengths = []
        for source in sample_nodes[:100]:
            lengths = nx.single_source_shortest_path_length(giant_G, source)
            path_lengths.extend(lengths.values())
        avg_path_length = float(np.mean(path_lengths)) if path_lengths else float("nan")
        diameter        = float("nan")  # too slow for large graphs
    else:
        avg_path_length = nx.average_shortest_path_length(giant_G) if nx.is_connected(giant_G) else float("nan")
        diameter        = nx.diameter(giant_G) if nx.is_connected(giant_G) else float("nan")

    # ── Degree assortativity ────────────────────────────
    try:
        assortativity = nx.degree_assortativity_coefficient(G)
    except Exception:
        assortativity = float("nan")

    # ── Modularity (Louvain-like via greedy) ────────────
    try:
        communities   = nx.community.greedy_modularity_communities(giant_G)
        modularity    = nx.community.modularity(giant_G, communities)
        n_communities = len(communities)
        avg_community_size = giant_size / n_communities if n_communities > 0 else 0
    except Exception:
        modularity         = float("nan")
        n_communities      = 0
        avg_community_size = 0.0

    # ── Scale-free topology fit ─────────────────────────
    deg_nonzero = degree_seq[degree_seq > 0]
    if len(deg_nonzero) >= 10:
        counts, bin_edges = np.histogram(deg_nonzero, bins=20)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mask = counts > 0
        if mask.sum() >= 3:
            slope, _, r_val, _, _ = stats.linregress(
                np.log10(bin_centers[mask]),
                np.log10(counts[mask] / counts[mask].sum())
            )
            sft_r2    = r_val ** 2
            sft_gamma = abs(slope)
        else:
            sft_r2, sft_gamma = 0.0, 0.0
    else:
        sft_r2, sft_gamma = 0.0, 0.0

    # ── Weighted connectivity (for soft networks) ────────
    connectivity = abs_vals.sum(axis=1)

    return {
        # identifiers
        "name":               name,
        "method":             method,
        # degree
        "degree_mean":        float(degree_seq.mean()),
        "degree_std":         float(degree_seq.std()),
        "degree_min":         float(degree_seq.min()),
        "degree_max":         float(degree_seq.max()),
        "degree_seq":         degree_seq,         # kept for plotting
        # clustering
        "clustering_mean":    float(clustering_vals.mean()) if len(clustering_vals) else 0.0,
        "clustering_std":     float(clustering_vals.std())  if len(clustering_vals) else 0.0,
        "clustering_vals":    clustering_vals,
        # betweenness
        "between_mean":       float(between_vals.mean()) if len(between_vals) else 0.0,
        "between_std":        float(between_vals.std())  if len(between_vals) else 0.0,
        "between_vals":       between_vals,
        # giant component
        "giant_size":         giant_size,
        "giant_pct":          giant_pct,
        "n_components":       n_comp,
        # network level
        "n_edges":            G.number_of_edges(),
        "density":            nx.density(G),
        "assortativity":      assortativity,
        "avg_path_length":    avg_path_length,
        "diameter":           diameter,
        "modularity":         modularity,
        "n_communities":      n_communities,
        "avg_community_size": avg_community_size,
        # scale-free
        "sft_r2":             sft_r2,
        "sft_gamma":          sft_gamma,
        # weighted connectivity
        "connectivity_mean":  float(connectivity.mean()),
        "connectivity_std":   float(connectivity.std()),
    }


# ══════════════════════════════════════════════════════
# FIGURE 1: Degree Distribution (Couto et al. Fig 5 equivalent)
# ══════════════════════════════════════════════════════
def plot_degree_distributions(topologies: list[dict], out_dir: Path) -> None:
    """
    Degree distribution for all 6 networks in log-log space.
    2 rows (Hard, Soft) x 3 cols (Pearson, Spearman, Proportionality)
    """
    logger.info("Plotting degree distributions...")

    names   = [t["name"]   for t in topologies[:3]]
    methods = ["Hard", "Soft"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Degree Distribution — All 6 Networks\n"
        "Log-log scale: straight line indicates scale-free topology",
        fontsize=13, fontweight="bold"
    )

    for row, method in enumerate(methods):
        row_topos = [t for t in topologies if t["method"] == method]
        for col, (topo, color) in enumerate(zip(row_topos, METRIC_COLORS)):
            ax  = axes[row, col]
            deg = topo["degree_seq"]
            deg_nonzero = deg[deg > 0]

            counts, bin_edges = np.histogram(deg_nonzero, bins=30)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            mask = counts > 0

            ax.scatter(bin_centers[mask], counts[mask] / counts[mask].sum(),
                       color=color, alpha=0.7, s=25, zorder=3)

            # Power-law fit
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
                        label=f"γ={abs(slope):.2f}, R²={r_value**2:.3f}")

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Degree (k)", fontsize=10)
            ax.set_ylabel("P(k)", fontsize=10)
            ax.set_title(
                f"{topo['name']} — {method}\n"
                f"mean k={topo['degree_mean']:.1f}, SFT R²={topo['sft_r2']:.3f}",
                fontsize=10, fontweight="bold"
            )
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(out_dir / "fig1_degree_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → fig1_degree_distributions.png")


# ══════════════════════════════════════════════════════
# FIGURE 2: Topology Bar Chart Comparison (Couto et al. Table S3 equivalent)
# ══════════════════════════════════════════════════════
def plot_topology_bars(topologies: list[dict], out_dir: Path) -> None:
    """
    Grouped bar charts for all key topology metrics.
    Groups = metrics, bars within group = Hard vs Soft.
    Based on Table S3 from Couto et al. (2017).
    """
    logger.info("Plotting topology bar charts...")

    metrics_config = [
        ("degree_mean",        "Mean Degree",                False),
        ("degree_std",         "Degree Std Dev",             False),
        ("clustering_mean",    "Clustering Coefficient",     False),
        ("between_mean",       "Betweenness Centrality",     False),
        ("giant_pct",          "Giant Component (%)",        False),
        ("n_components",       "Number of Components",       False),
        ("assortativity",      "Degree Assortativity",       False),
        ("modularity",         "Modularity",                 False),
        ("avg_community_size", "Avg Community Size",         False),
        ("sft_r2",             "Scale-Free R²",              True),
        ("density",            "Network Density",            False),
        ("n_edges",            "Number of Edges",            False),
    ]

    names   = sorted(set(t["name"]   for t in topologies))
    methods = sorted(set(t["method"] for t in topologies))

    n_plots = len(metrics_config)
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle(
        "Network Topology Properties — Hard vs Soft Thresholding\n"
        "Based on Couto et al. (2017) Mol. BioSyst. 13:2024-2035",
        fontsize=13, fontweight="bold"
    )

    x      = np.arange(len(names))
    width  = 0.35

    for ax, (metric, label, add_ref) in zip(axes.flat, metrics_config):
        for i, (method, color, hatch) in enumerate(zip(methods, METHOD_COLORS, METHOD_HATCHES)):
            values = []
            for name in names:
                match = next((t for t in topologies
                              if t["name"] == name and t["method"] == method), None)
                values.append(match[metric] if match and not np.isnan(match[metric]) else 0)

            bars = ax.bar(x + i * width - width / 2, values,
                          width, label=method, color=color,
                          hatch=hatch, alpha=0.85, edgecolor="white")

            # Annotate
            for bar, val in zip(bars, values):
                if val != 0:
                    fmt = f"{val:.3f}" if abs(val) < 100 else f"{val:,.0f}"
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.01,
                            fmt, ha="center", va="bottom",
                            fontsize=7, rotation=45)

        if add_ref:
            ax.axhline(0.8, color=RED, linestyle="--", lw=1.5,
                       alpha=0.7, label="R²=0.8 threshold")

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "fig2_topology_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → fig2_topology_bars.png")


# ══════════════════════════════════════════════════════
# FIGURE 3: Clustering Coefficient Distribution
# ══════════════════════════════════════════════════════
def plot_clustering_distributions(topologies: list[dict], out_dir: Path) -> None:
    """
    Distribution of local clustering coefficients for all 6 networks.
    """
    logger.info("Plotting clustering distributions...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(
        "Local Clustering Coefficient Distribution\n"
        "Measures tendency of genes to form tightly connected groups",
        fontsize=13, fontweight="bold"
    )

    methods = ["Hard", "Soft"]
    for row, method in enumerate(methods):
        row_topos = [t for t in topologies if t["method"] == method]
        for col, (topo, color) in enumerate(zip(row_topos, METRIC_COLORS)):
            ax   = axes[row, col]
            vals = topo["clustering_vals"]

            ax.hist(vals, bins=50, color=color, alpha=0.8, edgecolor="none", density=True)
            ax.axvline(vals.mean(), color=NAVY, linestyle="--", lw=2,
                       label=f"Mean = {vals.mean():.3f}")
            ax.axvline(np.median(vals), color=RED, linestyle="--", lw=1.5,
                       label=f"Median = {np.median(vals):.3f}")
            ax.set_xlabel("Clustering Coefficient", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.set_title(
                f"{topo['name']} — {method}\n"
                f"std = {vals.std():.3f}",
                fontsize=10, fontweight="bold"
            )
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "fig3_clustering_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → fig3_clustering_distributions.png")


# ══════════════════════════════════════════════════════
# FIGURE 4: Betweenness Centrality Distribution
# ══════════════════════════════════════════════════════
def plot_betweenness_distributions(topologies: list[dict], out_dir: Path) -> None:
    """
    Distribution of betweenness centrality — identifies hub genes.
    """
    logger.info("Plotting betweenness distributions...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(
        "Betweenness Centrality Distribution\n"
        "High betweenness = hub/bridge genes connecting different modules",
        fontsize=13, fontweight="bold"
    )

    methods = ["Hard", "Soft"]
    for row, method in enumerate(methods):
        row_topos = [t for t in topologies if t["method"] == method]
        for col, (topo, color) in enumerate(zip(row_topos, METRIC_COLORS)):
            ax   = axes[row, col]
            vals = topo["between_vals"]

            ax.hist(vals, bins=50, color=color, alpha=0.8,
                    edgecolor="none", density=True)
            ax.axvline(vals.mean(), color=NAVY, linestyle="--", lw=2,
                       label=f"Mean = {vals.mean():.4f}")

            # Highlight top 1% hub genes
            p99 = np.percentile(vals, 99)
            ax.axvline(p99, color=RED, linestyle="--", lw=1.5,
                       label=f"Top 1% hub: >{p99:.4f}")
            ax.set_xlabel("Betweenness Centrality", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.set_title(
                f"{topo['name']} — {method}",
                fontsize=10, fontweight="bold"
            )
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "fig4_betweenness_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → fig4_betweenness_distributions.png")


# ══════════════════════════════════════════════════════
# FIGURE 5: Topology Heatmap (Couto et al. Fig 9b equivalent)
# ══════════════════════════════════════════════════════
def plot_topology_heatmap(topologies: list[dict], out_dir: Path) -> None:
    """
    Heatmap of normalized topology properties across all 6 networks.
    Equivalent to Couto et al. (2017) Fig 9b comparison matrix.
    """
    logger.info("Plotting topology heatmap...")

    metrics = [
        ("degree_mean",        "Mean Degree"),
        ("clustering_mean",    "Clustering"),
        ("between_mean",       "Betweenness"),
        ("giant_pct",          "Giant Comp %"),
        ("n_components",       "N Components"),
        ("assortativity",      "Assortativity"),
        ("modularity",         "Modularity"),
        ("avg_community_size", "Avg Comm. Size"),
        ("sft_r2",             "SFT R²"),
        ("density",            "Density"),
        ("sft_gamma",          "γ (power-law)"),
    ]

    network_labels = [f"{t['name']}\n({t['method']})" for t in topologies]
    data = []
    for metric_key, _ in metrics:
        row = []
        for topo in topologies:
            val = topo[metric_key]
            row.append(val if not np.isnan(val) else 0.0)
        data.append(row)

    data_arr = np.array(data, dtype=float)

    # Normalize each row to [0, 1] for visualization
    data_norm = np.zeros_like(data_arr)
    for i, row in enumerate(data_arr):
        row_min, row_max = row.min(), row.max()
        if row_max > row_min:
            data_norm[i] = (row - row_min) / (row_max - row_min)
        else:
            data_norm[i] = 0.5

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(
        "Topology Property Comparison Across All 6 Networks\n"
        "Left: Raw values | Right: Normalized (0=min, 1=max per property)",
        fontsize=13, fontweight="bold"
    )

    metric_labels = [label for _, label in metrics]

    # Left: raw values
    ax = axes[0]
    im = ax.imshow(data_arr, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Raw value")
    ax.set_xticks(range(len(network_labels)))
    ax.set_xticklabels(network_labels, fontsize=9)
    ax.set_yticks(range(len(metric_labels)))
    ax.set_yticklabels(metric_labels, fontsize=10)
    ax.set_title("Raw Values", fontsize=11, fontweight="bold")
    for i in range(len(metrics)):
        for j in range(len(topologies)):
            val = data_arr[i, j]
            fmt = f"{val:.3f}" if abs(val) < 10 else f"{val:,.0f}"
            ax.text(j, i, fmt, ha="center", va="center",
                    fontsize=7, color="black")

    # Right: normalized
    ax = axes[1]
    im2 = ax.imshow(data_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im2, ax=ax, label="Normalized value")
    ax.set_xticks(range(len(network_labels)))
    ax.set_xticklabels(network_labels, fontsize=9)
    ax.set_yticks(range(len(metric_labels)))
    ax.set_yticklabels(metric_labels, fontsize=10)
    ax.set_title("Normalized (per property)", fontsize=11, fontweight="bold")
    for i in range(len(metrics)):
        for j in range(len(topologies)):
            ax.text(j, i, f"{data_norm[i, j]:.2f}",
                    ha="center", va="center", fontsize=7, color="black")

    # Vertical lines separating Hard vs Soft
    for ax_ in axes:
        ax_.axvline(2.5, color=SLATE, lw=2, linestyle="--", alpha=0.5)
        ax_.text(1.0, -1.0, "Hard Threshold",
                 ha="center", fontsize=9, color=SLATE,
                 transform=ax_.transData)
        ax_.text(4.0, -1.0, "Soft Threshold",
                 ha="center", fontsize=9, color=SLATE,
                 transform=ax_.transData)

    plt.tight_layout()
    plt.savefig(out_dir / "fig5_topology_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → fig5_topology_heatmap.png")


# ══════════════════════════════════════════════════════
# FIGURE 6: Hub Gene Overlap
# ══════════════════════════════════════════════════════
def plot_hub_overlap(
    adj_list: list[pd.DataFrame],
    topologies: list[dict],
    out_dir: Path,
    top_n: int = 50,
) -> None:
    """
    Compare top hub genes (by degree) across all 6 networks.
    Shows overlap between metrics and thresholding methods.
    """
    logger.info("Plotting hub gene overlap...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Top {top_n} Hub Genes by Degree — Overlap Across Methods\n"
        "Shared hubs suggest robust co-expression relationships",
        fontsize=13, fontweight="bold"
    )

    methods = ["Hard", "Soft"]
    hub_sets = {}

    for row, method in enumerate(methods):
        row_topos  = [t for t in topologies if t["method"] == method]
        row_adjs   = [a for a, t in zip(adj_list, topologies) if t["method"] == method]

        for col, (topo, adj, color) in enumerate(zip(row_topos, row_adjs, METRIC_COLORS)):
            ax = axes[row, col]

            abs_adj  = adj.abs()
            np.fill_diagonal(abs_adj.values, 0)
            degree   = abs_adj.values.sum(axis=1)
            gene_ids = adj.index

            top_idx  = np.argsort(degree)[::-1][:top_n]
            top_genes = set(gene_ids[top_idx])
            hub_sets[f"{topo['name']}_{method}"] = top_genes

            top_degrees = degree[top_idx]
            ax.barh(range(min(20, top_n)), top_degrees[:20][::-1],
                    color=color, alpha=0.8)
            ax.set_yticks(range(min(20, top_n)))
            ax.set_yticklabels([gene_ids[i] for i in top_idx[:20]][::-1],
                               fontsize=7)
            ax.set_xlabel("Degree", fontsize=10)
            ax.set_title(
                f"{topo['name']} — {method}\nTop 20 hub genes",
                fontsize=10, fontweight="bold"
            )
            ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "fig6_hub_genes.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → fig6_hub_genes.png")

    # Save hub overlap table
    hub_names = list(hub_sets.keys())
    overlap_matrix = np.zeros((len(hub_names), len(hub_names)))
    for i, k1 in enumerate(hub_names):
        for j, k2 in enumerate(hub_names):
            overlap = len(hub_sets[k1] & hub_sets[k2])
            overlap_matrix[i, j] = overlap

    overlap_df = pd.DataFrame(overlap_matrix,
                               index=hub_names, columns=hub_names)
    overlap_df.to_csv(out_dir / "hub_overlap.csv")

    # Plot overlap heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(overlap_matrix, cmap="Blues",
                   vmin=0, vmax=top_n)
    plt.colorbar(im, ax=ax, label=f"Shared hub genes (out of top {top_n})")
    ax.set_xticks(range(len(hub_names)))
    ax.set_xticklabels(hub_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(hub_names)))
    ax.set_yticklabels(hub_names, fontsize=9)
    ax.set_title(
        f"Hub Gene Overlap (Top {top_n} genes by degree)\n"
        "Diagonal = self-overlap (= top_n)",
        fontsize=12, fontweight="bold"
    )
    for i in range(len(hub_names)):
        for j in range(len(hub_names)):
            ax.text(j, i, f"{int(overlap_matrix[i,j])}",
                    ha="center", va="center", fontsize=9,
                    fontweight="bold" if i == j else "normal")

    plt.tight_layout()
    plt.savefig(out_dir / "fig7_hub_overlap_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved → fig7_hub_overlap_heatmap.png")


# ══════════════════════════════════════════════════════
# SAVE SUMMARY TABLE
# ══════════════════════════════════════════════════════
def save_summary_table(topologies: list[dict], out_dir: Path) -> None:
    """Save full topology summary table."""
    exclude = {"degree_seq", "clustering_vals", "between_vals", "connectivity"}
    rows = []
    for topo in topologies:
        row = {k: v for k, v in topo.items() if k not in exclude}
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "topology_summary.csv", index=False)

    print("\n" + "="*70)
    print("TOPOLOGY SUMMARY — ALL 6 NETWORKS")
    print("="*70)
    cols = ["name", "method", "n_edges", "degree_mean", "clustering_mean",
            "giant_pct", "modularity", "sft_r2", "assortativity"]
    print(df[cols].to_string(index=False))
    logger.info("Saved → topology_summary.csv")


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare topology of 6 co-expression networks."
    )
    parser.add_argument("--hard-inputs", nargs="+", required=True,
                        help="3 hard-thresholded adjacency matrix CSVs.")
    parser.add_argument("--soft-inputs", nargs="+", required=True,
                        help="3 soft-thresholded adjacency matrix CSVs.")
    parser.add_argument("--names", nargs="+", required=True,
                        help="Names e.g. Pearson Spearman Proportionality.")
    parser.add_argument("--output", required=True,
                        help="Output directory.")
    parser.add_argument("--top-hubs", type=int, default=50,
                        help="Number of top hub genes to compare (default: 50).")

    args = parser.parse_args()

    if not (len(args.hard_inputs) == len(args.soft_inputs) == len(args.names)):
        raise ValueError("--hard-inputs, --soft-inputs, and --names must have same length.")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all 6 adjacency matrices
    hard_adjs = [load_matrix(p) for p in args.hard_inputs]
    soft_adjs = [load_matrix(p) for p in args.soft_inputs]

    all_adjs = hard_adjs + soft_adjs

    # Compute topology for all 6
    logger.info("Computing topology for all 6 networks...")
    topologies = []
    for adj, name in zip(hard_adjs, args.names):
        topo = compute_topology(adj, name, "Hard")
        topologies.append(topo)
    for adj, name in zip(soft_adjs, args.names):
        topo = compute_topology(adj, name, "Soft")
        topologies.append(topo)

    # Generate all figures
    plot_degree_distributions(topologies, out_dir)
    plot_topology_bars(topologies, out_dir)
    plot_clustering_distributions(topologies, out_dir)
    plot_betweenness_distributions(topologies, out_dir)
    plot_topology_heatmap(topologies, out_dir)
    plot_hub_overlap(all_adjs, topologies, out_dir, top_n=args.top_hubs)

    # Save summary
    save_summary_table(topologies, out_dir)

    logger.info("All outputs saved to: %s", out_dir)
    print(f"\nDone! All figures saved to: {out_dir}")


if __name__ == "__main__":
    main()