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
from scipy import sparse
from scipy.sparse import save_npz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_INPUT = "data/processed/preprocessed_counts.csv"
DEFAULT_SIMILARITY = "data/similarity/pearson_similarity.parquet"
DEFAULT_OUTPUT_SOFT = "data/processed/adjacency_soft.parquet"
DEFAULT_OUTPUT_HARD = "data/processed/adjacency_hard.npz"


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
    expr = expr.replace([np.inf, -np.inf], np.nan)

    n_dupes = expr.index.duplicated(keep="first").sum()
    if n_dupes > 0:
        logger.warning("Dropping %d duplicated gene(s).", n_dupes)
    expr = expr.loc[~expr.index.duplicated(keep="first")]

    expr = expr.dropna(axis=0, how="all")
    expr = expr.dropna(axis=1, how="all")

    logger.info("After cleaning: %d genes x %d samples", expr.shape[0], expr.shape[1])
    return expr


def compute_pearson_similarity(
    expr: pd.DataFrame,
    min_periods: int = 3,
    chunk_size: int = 2000,
) -> pd.DataFrame:
    logger.info(
        "Computing Pearson correlation for %d genes (chunk_size=%d)...",
        expr.shape[0],
        chunk_size,
    )

    genes = expr.index.tolist()
    n = len(genes)

    expr_vals = expr.to_numpy(dtype=np.float64, copy=True)

    means = expr_vals.mean(axis=1, keepdims=True)
    stds = expr_vals.std(axis=1, ddof=1, keepdims=True)
    stds[stds == 0] = np.nan
    expr_std = (expr_vals - means) / stds

    n_samples = expr_vals.shape[1]
    n_chunks = (n + chunk_size - 1) // chunk_size

    corr_array = np.full((n, n), np.nan, dtype=np.float32)

    for i, start in enumerate(range(0, n, chunk_size)):
        end = min(start + chunk_size, n)
        chunk_corr = np.dot(expr_std[start:end], expr_std.T) / (n_samples - 1)
        corr_array[start:end, :] = chunk_corr.astype(np.float32)
        logger.info("  Chunk %d/%d complete (genes %d-%d)", i + 1, n_chunks, start, end)

    np.fill_diagonal(corr_array, 1.0)
    np.clip(corr_array, -1.0, 1.0, out=corr_array)

    return pd.DataFrame(corr_array, index=genes, columns=genes)


def summarize_similarity(corr: pd.DataFrame) -> dict:
    arr = corr.to_numpy(dtype=np.float32, copy=False)
    n = arr.shape[0]
    total_pairs = n * (n - 1) // 2

    count = 0
    nan_count = 0
    sum_val = 0.0
    min_val = np.inf
    max_val = -np.inf

    for i in range(n - 1):
        row = arr[i, i + 1:]
        valid = row[~np.isnan(row)]

        nan_count += int(row.size - valid.size)
        if valid.size:
            count += int(valid.size)
            sum_val += float(valid.sum(dtype=np.float64))
            min_val = min(min_val, float(valid.min()))
            max_val = max(max_val, float(valid.max()))

    return {
        "num_genes": int(n),
        "num_pairs_total": int(total_pairs),
        "num_pairs_nan": int(nan_count),
        "num_pairs_valid": int(count),
        "mean_correlation": float(sum_val / count) if count else float("nan"),
        "min_correlation": float(min_val) if count else float("nan"),
        "max_correlation": float(max_val) if count else float("nan"),
    }


def soft_threshold(corr: pd.DataFrame, beta: int, chunk_size: int = 2000) -> pd.DataFrame:
    logger.info("Applying soft thresholding (beta=%d).", beta)

    genes = corr.index.tolist()
    n = len(genes)
    n_chunks = (n + chunk_size - 1) // chunk_size
    arr = np.zeros((n, n), dtype=np.float32)

    for i, start in enumerate(range(0, n, chunk_size)):
        end = min(start + chunk_size, n)
        chunk = np.abs(corr.iloc[start:end].to_numpy(dtype=np.float32, copy=True))
        arr[start:end, :] = np.power(chunk, beta).astype(np.float32)
        logger.info("  Chunk %d/%d complete (genes %d-%d)", i + 1, n_chunks, start, end)

    np.fill_diagonal(arr, 0.0)
    return pd.DataFrame(arr, index=genes, columns=genes)


def hard_threshold(corr: pd.DataFrame, tau: float) -> sparse.csr_matrix:
    logger.info("Applying hard thresholding (tau=%.3f).", tau)

    arr = np.abs(corr.to_numpy(dtype=np.float32, copy=False))
    mask = arr >= tau
    np.fill_diagonal(mask, False)

    return sparse.csr_matrix(mask.astype(np.uint8))


def summarize_soft_adjacency(adj: pd.DataFrame) -> dict:
    arr = adj.to_numpy(dtype=np.float32, copy=False)
    n = arr.shape[0]
    count = 0
    sum_w = 0.0
    min_w = np.inf
    max_w = -np.inf
    above_01 = 0
    above_05 = 0

    for i in range(n - 1):
        row = arr[i, i + 1:]
        count += int(row.size)
        sum_w += float(row.sum(dtype=np.float64))
        min_w = min(min_w, float(row.min()))
        max_w = max(max_w, float(row.max()))
        above_01 += int(np.count_nonzero(row > 0.1))
        above_05 += int(np.count_nonzero(row > 0.5))

    return {
        "method": "soft",
        "num_genes": int(n),
        "num_pairs": int(count),
        "mean_edge_weight": float(sum_w / count) if count else float("nan"),
        "min_edge_weight": float(min_w) if count else float("nan"),
        "max_edge_weight": float(max_w) if count else float("nan"),
        "edges_above_0.1": int(above_01),
        "edges_above_0.5": int(above_05),
        "pct_strong_edges": float(100 * above_05 / count) if count else float("nan"),
    }


def summarize_hard_adjacency(adj_sparse: sparse.csr_matrix) -> dict:
    n = adj_sparse.shape[0]
    total_possible_edges = n * (n - 1) // 2

    num_edges = int(adj_sparse.nnz // 2)
    density = float(num_edges / total_possible_edges) if total_possible_edges else float("nan")

    degree = np.asarray(adj_sparse.sum(axis=1)).ravel()
    isolated = int(np.sum(degree == 0))

    return {
        "method": "hard",
        "num_genes": int(n),
        "total_possible_edges": int(total_possible_edges),
        "num_edges": int(num_edges),
        "network_density": round(density, 6),
        "mean_degree": round(float(np.mean(degree)), 4),
        "max_degree": float(np.max(degree)),
        "isolated_nodes": int(isolated),
        "pct_isolated": round(100 * isolated / n, 2),
    }


def save_matrix(matrix, output_path: str | Path, label: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if isinstance(matrix, sparse.spmatrix):
        if suffix != ".npz":
            output_path = output_path.with_suffix(".npz")
        save_npz(output_path, matrix)
        logger.info("Saved %s to: %s", label, output_path)
        return

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Pearson similarity + adjacency matrix construction.")

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Path to expression matrix (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--similarity",
        default=DEFAULT_SIMILARITY,
        help=f"Path to save/load similarity matrix (default: {DEFAULT_SIMILARITY}).",
    )
    parser.add_argument(
        "--min-periods",
        type=int,
        default=3,
        help="Min overlapping values per gene pair (default: 3).",
    )
    parser.add_argument(
        "--skip-similarity",
        action="store_true",
        help="Skip recomputing similarity if the file already exists.",
    )

    method_group = parser.add_mutually_exclusive_group(required=True)
    method_group.add_argument("--soft", action="store_true", help="Apply soft thresholding (weighted network).")
    method_group.add_argument("--hard", action="store_true", help="Apply hard thresholding (binary network).")

    parser.add_argument("--beta", type=int, default=6, help="Power for soft thresholding (default: 6).")
    parser.add_argument("--tau", type=float, default=0.7, help="Cutoff for hard thresholding (default: 0.7).")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for adjacency matrix. Defaults to data/processed/adjacency_soft.parquet or adjacency_hard.npz.",
    )

    args = parser.parse_args()

    sim_path = Path(args.similarity)

    if args.skip_similarity and sim_path.exists():
        logger.info("--skip-similarity set and file exists. Loading saved similarity matrix.")
        if sim_path.suffix.lower() == ".parquet":
            corr = pd.read_parquet(sim_path)
        elif sim_path.suffix.lower() == ".csv":
            corr = pd.read_csv(sim_path, index_col=0)
        else:
            raise ValueError(f"Unsupported similarity format: {sim_path.suffix}")
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

    if args.soft:
        adj = soft_threshold(corr, beta=args.beta)
        adj_summary = summarize_soft_adjacency(adj)
        output_path = args.output or DEFAULT_OUTPUT_SOFT
        logger.info("--- Adjacency Summary ---")
        for key, value in adj_summary.items():
            logger.info("  %s: %s", key, value)
        save_matrix(adj, output_path, "adjacency matrix")
    else:
        adj_sparse = hard_threshold(corr, tau=args.tau)
        adj_summary = summarize_hard_adjacency(adj_sparse)
        output_path = args.output or DEFAULT_OUTPUT_HARD
        logger.info("--- Adjacency Summary ---")
        for key, value in adj_summary.items():
            logger.info("  %s: %s", key, value)
        save_matrix(adj_sparse, output_path, "adjacency matrix")


if __name__ == "__main__":
    main()
