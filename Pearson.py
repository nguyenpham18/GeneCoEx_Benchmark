""" 

Pearson similarity + adjacency matrix construction.
 
Phase 2 - Input:  cleaned expression matrix (data/processed/preprocessed_counts.csv)
          Output: Pearson gene-gene correlation matrix (data/similarity/pearson_similarity.parquet)
 
Phase 3 - Input:  Pearson similarity matrix
          Output: Adjacency matrix via soft or hard thresholding (data/processed/)

"""

from __future__ import annotations
 
import argparse
import logging
from pathlib import Path
 
import numpy as np
import pandas as pd
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
 
# ── Default paths matching the project structure ──────────────────────────────
DEFAULT_INPUT      = "data/processed/preprocessed_counts.csv"
DEFAULT_SIMILARITY = "data/similarity/pearson_similarity.parquet"
DEFAULT_OUTPUT_SOFT = "data/processed/adjacency_soft.parquet"
DEFAULT_OUTPUT_HARD = "data/processed/adjacency_hard.parquet"
 
 
# ── Phase 2: Pearson similarity ───────────────────────────────────────────────
 
def load_expression_matrix(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    logger.info("Loading expression matrix from: %s", path)
 
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, index_col=0)
    elif path.suffix.lower() in {".tsv", ".txt"}:
        df = pd.read_csv(path, sep="\t", index_col=0)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
 
    logger.info("Loaded: %d genes x %d samples", df.shape[0], df.shape[1])
    return df
 
 
def clean_expression_matrix(expr: pd.DataFrame) -> pd.DataFrame:
    expr = expr.copy()
    expr = expr.apply(pd.to_numeric, errors="coerce")
 
    n_dupes = expr.index.duplicated(keep="first").sum()
    if n_dupes > 0:
        logger.warning("Dropping %d duplicated gene(s).", n_dupes)
    expr = expr.loc[~expr.index.duplicated(keep="first")]
 
    expr = expr.dropna(axis=0, how="all")
    expr = expr.dropna(axis=1, how="all")
 
    logger.info("After cleaning: %d genes x %d samples", expr.shape[0], expr.shape[1])
    return expr
 
 
def compute_pearson_similarity(expr: pd.DataFrame, min_periods: int = 3) -> pd.DataFrame:
    logger.info("Computing Pearson correlation for %d genes...", expr.shape[0])
 
    # Transpose so genes become columns, as required by .corr()
    corr = expr.T.corr(method="pearson", min_periods=min_periods)
 
    # Guard against floating-point drift on the diagonal
    corr_array = corr.to_numpy().copy()
    np.fill_diagonal(corr_array, 1.0)
    corr = pd.DataFrame(corr_array, index=corr.index, columns=corr.columns)
 
    return corr
 
 
def summarize_similarity(corr: pd.DataFrame) -> dict:
    values = corr.values
    tri    = values[np.triu_indices_from(values, k=1)]  # upper triangle, no diagonal
    valid  = tri[~np.isnan(tri)]
 
    return {
        "num_genes":         int(corr.shape[0]),
        "num_pairs_total":   int(len(tri)),
        "num_pairs_nan":     int(np.isnan(tri).sum()),
        "num_pairs_valid":   int(len(valid)),
        "mean_correlation":  float(np.mean(valid)),
        "median_correlation":float(np.median(valid)),
        "min_correlation":   float(np.min(valid)),
        "max_correlation":   float(np.max(valid)),
    }
 
 
# ── Phase 3: Adjacency construction ──────────────────────────────────────────
 
def soft_threshold(corr: pd.DataFrame, beta: int) -> pd.DataFrame:
    """Soft thresholding: Aij = |sij|^beta.
 
    Keeps all edges but raises them to a power. Weak correlations fade
    toward zero, strong ones stay strong. Produces a weighted matrix.
    Beta is typically chosen between 6 and 12.
    """
    logger.info("Applying soft thresholding (beta=%d).", beta)
 
    arr = np.abs(corr.to_numpy().copy())  # unsigned: drop sign, keep strength
    arr = np.power(arr, beta)             # raise every value to the power beta
    np.fill_diagonal(arr, 0.0)            # no self-connections in adjacency
 
    return pd.DataFrame(arr, index=corr.index, columns=corr.columns)
 
 
def hard_threshold(corr: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Hard thresholding: Aij = 1 if |sij| >= tau, else 0.
 
    Applies a firm cutoff. Pairs at or above tau become 1 (connected),
    everything below becomes 0. Produces a binary matrix.
    Tau is typically chosen between 0.5 and 0.9.
    """
    logger.info("Applying hard thresholding (tau=%.3f).", tau)
 
    arr = np.abs(corr.to_numpy().copy())  # unsigned: drop sign, keep strength
    arr = (arr >= tau).astype(float)      # 1.0 if above threshold, 0.0 if below
    np.fill_diagonal(arr, 0.0)            # no self-connections in adjacency
 
    return pd.DataFrame(arr, index=corr.index, columns=corr.columns)
 
 
def summarize_adjacency(adj: pd.DataFrame, method: str) -> dict:
    arr   = adj.to_numpy()
    n     = arr.shape[0]
    tri   = arr[np.triu_indices_from(arr, k=1)]
    valid = tri[~np.isnan(tri)]
 
    if method == "soft":
        return {
            "method":             "soft",
            "num_genes":          n,
            "num_pairs":          int(len(valid)),
            "mean_edge_weight":   round(float(np.mean(valid)), 6),
            "median_edge_weight": round(float(np.median(valid)), 6),
            "edges_above_0.1":    int(np.sum(valid > 0.1)),
            "edges_above_0.5":    int(np.sum(valid > 0.5)),
            "pct_strong_edges":   round(float(100 * np.sum(valid > 0.5) / len(valid)), 4),
        }
    else:
        n_edges  = int(np.sum(valid))
        density  = float(n_edges / len(valid))
        degree   = arr.sum(axis=1)  # diagonal is 0 so no adjustment needed
        isolated = int(np.sum(degree == 0))
        return {
            "method":               "hard",
            "num_genes":            n,
            "total_possible_edges": int(len(valid)),
            "num_edges":            n_edges,
            "network_density":      round(density, 6),
            "mean_degree":          round(float(np.mean(degree)), 4),
            "max_degree":           float(np.max(degree)),
            "isolated_nodes":       isolated,
            "pct_isolated":         round(100 * isolated / n, 2),
        }
 
 
def save_matrix(matrix: pd.DataFrame, output_path: str | Path, label: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
 
    if suffix == ".parquet":
        matrix.to_parquet(output_path)
    elif suffix == ".csv":
        matrix.to_csv(output_path)
    elif suffix in {".tsv", ".txt"}:
        matrix.to_csv(output_path, sep="\t")
    else:
        output_path = output_path.with_suffix(".parquet")
        logger.warning("Unrecognized extension; saving as parquet: %s", output_path)
        matrix.to_parquet(output_path)
 
    logger.info("Saved %s to: %s", label, output_path)
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pearson similarity + adjacency matrix construction."
    )
 
    # Phase 2 args
    parser.add_argument("--input",      default=DEFAULT_INPUT,      help=f"Expression matrix path (default: {DEFAULT_INPUT}).")
    parser.add_argument("--similarity", default=DEFAULT_SIMILARITY, help=f"Where to save/load similarity matrix (default: {DEFAULT_SIMILARITY}).")
    parser.add_argument("--min-periods",type=int, default=3,        help="Min overlapping values per gene pair (default: 3).")
    parser.add_argument("--skip-similarity", action="store_true",
                        help="Skip recomputing similarity if the file already exists.")
 
    # Phase 3 args
    method_group = parser.add_mutually_exclusive_group(required=True)
    method_group.add_argument("--soft", action="store_true", help="Apply soft thresholding (weighted network).")
    method_group.add_argument("--hard", action="store_true", help="Apply hard thresholding (binary network).")
 
    parser.add_argument("--beta", type=int,   default=6,   help="Power for soft thresholding (default: 6).")
    parser.add_argument("--tau",  type=float, default=0.7, help="Cutoff for hard thresholding (default: 0.7).")
    parser.add_argument("--output", default=None,
                        help="Output path for adjacency matrix. Defaults to data/processed/adjacency_soft/hard.parquet.")
 
    args = parser.parse_args()
 
    # ── Phase 2 ──
    sim_path = Path(args.similarity)
 
    if args.skip_similarity and sim_path.exists():
        logger.info("--skip-similarity set and file exists. Loading saved similarity matrix.")
        corr = pd.read_parquet(sim_path) if sim_path.suffix == ".parquet" else pd.read_csv(sim_path, index_col=0)
        logger.info("Loaded similarity matrix: %d x %d", corr.shape[0], corr.shape[1])
    else:
        expr = load_expression_matrix(args.input)
        expr = clean_expression_matrix(expr)
        corr = compute_pearson_similarity(expr, min_periods=args.min_periods)
 
        sim_summary = summarize_similarity(corr)
        logger.info("--- Similarity Summary ---")
        for key, value in sim_summary.items():
            logger.info("  %s: %s", key, value)
 
        if sim_summary["num_pairs_nan"] > 0:
            nan_pct = 100 * sim_summary["num_pairs_nan"] / sim_summary["num_pairs_total"]
            logger.warning("%.1f%% of gene pairs have NaN correlation.", nan_pct)
 
        save_matrix(corr, sim_path, "Pearson similarity matrix")
 
    # ── Phase 3 ──
    if args.soft:
        adj         = soft_threshold(corr, beta=args.beta)
        adj_summary = summarize_adjacency(adj, method="soft")
        output_path = args.output or DEFAULT_OUTPUT_SOFT
    else:
        adj         = hard_threshold(corr, tau=args.tau)
        adj_summary = summarize_adjacency(adj, method="hard")
        output_path = args.output or DEFAULT_OUTPUT_HARD
 
    logger.info("--- Adjacency Summary ---")
    for key, value in adj_summary.items():
        logger.info("  %s: %s", key, value)
 
    save_matrix(adj, output_path, "adjacency matrix")
 
 
if __name__ == "__main__":
    main()
