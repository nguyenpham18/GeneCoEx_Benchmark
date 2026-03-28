# src/visualize_matrices.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUT_DIR = "results/figures/matrices"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════
# EXPRESSION MATRIX VISUALIZATIONS
# ══════════════════════════════════════════════════════
print("Loading expression matrix...")
expr = pd.read_csv("data/processed/preprocessed_counts.csv", index_col=0)
print(f"  Shape: {expr.shape}")

# ── Figure 1: Expression heatmap (top 50 variable genes x all samples)
print("Plotting expression heatmap...")
gene_std = expr.std(axis=1)
top50 = gene_std.nlargest(50).index
sub_expr = expr.loc[top50]

fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(sub_expr,
            ax=ax,
            cmap="viridis",
            xticklabels=False,
            yticklabels=True,
            cbar_kws={"label": "log1p(CTF)", "shrink": 0.6},
            linewidths=0)
ax.set_title("Expression Matrix — Top 50 Most Variable Genes\n17,578 genes × 125 samples (log1p CTF-normalized)", fontsize=13)
ax.set_xlabel("Samples (125)", fontsize=11)
ax.set_ylabel("Genes", fontsize=11)
ax.tick_params(axis='y', labelsize=7)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/expr_heatmap_top50.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved expr_heatmap_top50.png")

# ── Figure 2: Expression heatmap (top 50 genes x top 50 samples by variance)
print("Plotting expression submatrix...")
sample_std = expr.std(axis=0)
top50_samples = sample_std.nlargest(50).index
sub_expr2 = expr.loc[top50, top50_samples]

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(sub_expr2,
            ax=ax,
            cmap="viridis",
            xticklabels=False,
            yticklabels=True,
            cbar_kws={"label": "log1p(CTF)", "shrink": 0.6},
            annot=False)
ax.set_title("Expression Submatrix\nTop 50 Variable Genes × Top 50 Variable Samples", fontsize=13)
ax.set_xlabel("Samples", fontsize=11)
ax.set_ylabel("Genes", fontsize=11)
ax.tick_params(axis='y', labelsize=7)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/expr_heatmap_50x50.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved expr_heatmap_50x50.png")

# ── Figure 3: Per-sample expression distribution (boxplot)
print("Plotting per-sample distribution...")
sample_subset = expr.iloc[:, :50]  # first 50 samples for readability
fig, ax = plt.subplots(figsize=(16, 5))
sample_subset.boxplot(ax=ax, rot=90, fontsize=6,
                      boxprops=dict(color="#4C72B0"),
                      medianprops=dict(color="red", linewidth=1.5),
                      whiskerprops=dict(color="#4C72B0"),
                      capprops=dict(color="#4C72B0"),
                      flierprops=dict(marker='.', markersize=1, alpha=0.3))
ax.set_title("Per-Sample Expression Distribution (First 50 Samples)\nlog1p CTF-normalized", fontsize=13)
ax.set_xlabel("Samples", fontsize=11)
ax.set_ylabel("log1p(CTF expression)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/expr_per_sample_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved expr_per_sample_boxplot.png")

# ── Figure 4: Per-gene expression distribution
print("Plotting per-gene distribution...")
fig, ax = plt.subplots(figsize=(8, 5))
gene_means = expr.mean(axis=1)
gene_stds  = expr.std(axis=1)
ax.scatter(gene_means, gene_stds, alpha=0.2, s=3, color="#4C72B0")
ax.set_xlabel("Mean Expression (log1p CTF)", fontsize=12)
ax.set_ylabel("Std Dev of Expression", fontsize=12)
ax.set_title("Mean vs Std Dev per Gene\n(17,578 genes)", fontsize=13)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/expr_mean_vs_std.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved expr_mean_vs_std.png")

# ══════════════════════════════════════════════════════
# SIMILARITY MATRIX VISUALIZATIONS
# ══════════════════════════════════════════════════════
print("\nLoading similarity matrix...")
sim = pd.read_csv("data/similarity/pearson_similarity.csv", index_col=0)
print(f"  Shape: {sim.shape}")

# ── Figure 5: Similarity heatmap (top 100 variable genes)
print("Plotting similarity heatmap top 100...")
top100 = gene_std.nlargest(100).index
sub_sim = sim.loc[top100, top100]

fig, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(sub_sim,
            ax=ax,
            cmap="RdBu_r",
            center=0, vmin=-1, vmax=1,
            xticklabels=False,
            yticklabels=False,
            cbar_kws={"label": "Pearson r", "shrink": 0.8})
ax.set_title("Pearson Similarity Matrix\n(Top 100 Most Variable Genes)", fontsize=13)
ax.set_xlabel("Genes", fontsize=11)
ax.set_ylabel("Genes", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/sim_heatmap_top100.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved sim_heatmap_top100.png")

# ── Figure 6: Similarity heatmap sorted by hierarchical clustering
print("Plotting clustered similarity heatmap...")
top50_sim = sim.loc[top50, top50]

fig = sns.clustermap(top50_sim,
                     cmap="RdBu_r",
                     center=0, vmin=-1, vmax=1,
                     xticklabels=False,
                     yticklabels=True,
                     figsize=(12, 12),
                     cbar_kws={"label": "Pearson r"},
                     dendrogram_ratio=0.1,
                     colors_ratio=0.02)
fig.ax_heatmap.set_title("Clustered Similarity Matrix\n(Top 50 Most Variable Genes)", fontsize=13, pad=20)
fig.ax_heatmap.tick_params(axis='y', labelsize=7)
plt.savefig(f"{OUT_DIR}/sim_clustermap_top50.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved sim_clustermap_top50.png")

# ── Figure 7: Similarity row means (hub gene candidates)
print("Plotting hub gene candidates...")
row_means = sim.mean(axis=1)
top20_hubs = row_means.nlargest(20)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(20), top20_hubs.values[::-1], color="#0D9488")
ax.set_yticks(range(20))
ax.set_yticklabels(top20_hubs.index[::-1], fontsize=9)
ax.set_xlabel("Mean Pearson Correlation with All Other Genes", fontsize=11)
ax.set_title("Top 20 Hub Gene Candidates\n(Highest mean correlation across all gene pairs)", fontsize=13)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/sim_hub_genes_top20.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved sim_hub_genes_top20.png")

print(f"\nAll figures saved to {OUT_DIR}/")