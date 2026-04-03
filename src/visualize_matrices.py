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

