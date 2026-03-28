import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Paths ──────────────────────────────────────────────
PEARSON  = "data/similarity/pearson_similarity.csv"
OUT_DIR  = "results/figures/pearson"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

print("Loading matrix...")
df = pd.read_csv(PEARSON, index_col=0)
vals = df.values
tri = vals[np.triu_indices_from(vals, k=1)]

# ══════════════════════════════════════════════════════
# Figure 1: Distribution of All Pairwise Correlations
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(tri, bins=100, color="#4C72B0", edgecolor="none", alpha=0.85)
ax.axvline(np.mean(tri),   color="red",    linestyle="--", linewidth=1.5, label=f"Mean = {np.mean(tri):.3f}")
ax.axvline(np.median(tri), color="orange", linestyle="--", linewidth=1.5, label=f"Median = {np.median(tri):.3f}")
ax.set_xlabel("Pearson Correlation", fontsize=12)
ax.set_ylabel("Number of Gene Pairs", fontsize=12)
ax.set_title("Distribution of All Pairwise Correlations\nGTEx Brain Cortex — 17,578 genes × 125 samples", fontsize=13)
ax.legend(fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1e6)}M"))
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig1_distribution_all.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig1_distribution_all.png")

# ══════════════════════════════════════════════════════
# Figure 2: Distribution of Positive Correlations
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(tri[tri > 0], bins=100, color="#55A868", edgecolor="none", alpha=0.85)
ax.axvline(0.5, color="red",    linestyle="--", linewidth=1.5, label=f"τ = 0.5 → {100*(tri>0.5).mean():.1f}% pairs")
ax.axvline(0.7, color="orange", linestyle="--", linewidth=1.5, label=f"τ = 0.7 → {100*(tri>0.7).mean():.1f}% pairs")
ax.axvline(0.8, color="purple", linestyle="--", linewidth=1.5, label=f"τ = 0.8 → {100*(tri>0.8).mean():.1f}% pairs")
ax.set_xlabel("Pearson Correlation", fontsize=12)
ax.set_ylabel("Number of Gene Pairs", fontsize=12)
ax.set_title("Distribution of Positive Correlations\n(Threshold candidates shown)", fontsize=13)
ax.legend(fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1e6)}M"))
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig2_distribution_positive.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2_distribution_positive.png")

# ══════════════════════════════════════════════════════
# Figure 3: CDF
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
sorted_tri = np.sort(tri)
cdf = np.arange(1, len(sorted_tri) + 1) / len(sorted_tri)
ax.plot(sorted_tri, cdf, color="#4C72B0", linewidth=1.5)
ax.axvline(0.5, color="red",    linestyle="--", linewidth=1.2, label=f"τ=0.5 → {100*(tri>0.5).mean():.1f}% pairs retained")
ax.axvline(0.7, color="orange", linestyle="--", linewidth=1.2, label=f"τ=0.7 → {100*(tri>0.7).mean():.1f}% pairs retained")
ax.axvline(0.8, color="purple", linestyle="--", linewidth=1.2, label=f"τ=0.8 → {100*(tri>0.8).mean():.1f}% pairs retained")
ax.set_xlabel("Pearson Correlation", fontsize=12)
ax.set_ylabel("Cumulative Proportion of Pairs", fontsize=12)
ax.set_title("Cumulative Distribution Function (CDF)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig3_cdf.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig3_cdf.png")

# ══════════════════════════════════════════════════════
# Figure 4: Network Density vs Hard Threshold
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
thresholds = np.arange(0, 1.01, 0.01)
pct_retained = [100 * (tri > t).mean() for t in thresholds]
ax.plot(thresholds, pct_retained, color="#C44E52", linewidth=2)
ax.axhline(5, color="red",    linestyle="--", linewidth=1.2, label="5% density")
ax.axhline(1, color="orange", linestyle="--", linewidth=1.2, label="1% density")
ax.fill_between(thresholds, pct_retained, 0,
                where=[p <= 5 for p in pct_retained],
                alpha=0.15, color="green", label="Target zone (1-5%)")
ax.set_xlabel("Hard Threshold (τ)", fontsize=12)
ax.set_ylabel("% of Pairs Retained", fontsize=12)
ax.set_title("Network Density vs Hard Threshold\n(Target: 1–5% density)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig4_density_vs_threshold.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig4_density_vs_threshold.png")

# ══════════════════════════════════════════════════════
# Figure 5: Heatmap (Top 100 Variable Genes)
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 7))
gene_std = df.std(axis=1)
top100 = gene_std.nlargest(100).index
sub = df.loc[top100, top100]
sns.heatmap(sub, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            xticklabels=False, yticklabels=False,
            cbar_kws={"label": "Pearson r", "shrink": 0.8})
ax.set_title("Correlation Heatmap\n(Top 100 Most Variable Genes)", fontsize=13)
ax.set_xlabel("Genes", fontsize=12)
ax.set_ylabel("Genes", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig5_heatmap_top100.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig5_heatmap_top100.png")

# ══════════════════════════════════════════════════════
# Figure 6: Gene Pair Counts by Category
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
categories = ["> 0.8\n(Strong+)", "> 0.5\n(Moderate+)", "0 to 0.5\n(Weak+)", "< 0\n(Negative)"]
counts = [
    (tri > 0.8).sum(),
    ((tri > 0.5) & (tri <= 0.8)).sum(),
    ((tri > 0) & (tri <= 0.5)).sum(),
    (tri < 0).sum()
]
colors = ["#2ecc71", "#3498db", "#95a5a6", "#e74c3c"]
bars = ax.bar(categories, [c/1e6 for c in counts], color=colors, edgecolor="white", linewidth=0.5)
for bar, count in zip(bars, counts):
    pct = 100 * count / len(tri)
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f"{pct:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Number of Gene Pairs (Millions)", fontsize=12)
ax.set_title("Gene Pair Counts by Correlation Category", fontsize=13)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig6_pair_counts.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig6_pair_counts.png")

print(f"\nAll figures saved to {OUT_DIR}/")