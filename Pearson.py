""" Pearson similarity-metric setup for gene co-expression benchmarking.

Input: cleaned gene expression matrix (genes as rows, samples as columns by default).
Output: Pearson gene-gene correlation matrix.


"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_expression_matrix(
    input_path: str | Path,
    sep: Optional[str] = None,
    index_col: int = 0,
    genes_as_rows: bool = True,
) -> pd.DataFrame:
    input_path = Path(input_path)
    logger.info("Loading expression matrix from: %s", input_path)

    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
        if not genes_as_rows:
            df = df.T
        logger.info("Loaded parquet: %d genes x %d samples", df.shape[0], df.shape[1])
        return df

    if sep is None:
        suffix = input_path.suffix.lower()
        if suffix == ".csv":
            sep = ","
        elif suffix in {".tsv", ".txt"}:
            sep = "\t"
        else:
            raise ValueError(
                f"Could not infer separator from file extension '{input_path.suffix}'. "
                "Please pass --sep explicitly."
            )

    df = pd.read_csv(input_path, sep=sep, index_col=index_col)
    if not genes_as_rows:
        df = df.T

    logger.info("Loaded text file: %d genes x %d samples", df.shape[0], df.shape[1])
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
    logger.info(
        "Computing Pearson correlation for %d genes (min_periods=%d).",
        expr.shape[0],
        min_periods,
    )

    # Transpose so genes become columns, as required by .corr()
    corr = expr.T.corr(method="pearson", min_periods=min_periods)

    # Guard against floating-point drift on the diagonal
    corr_array = corr.to_numpy().copy()
    np.fill_diagonal(corr_array, 1.0)
    corr = pd.DataFrame(corr_array, index=corr.index, columns=corr.columns)

    return corr


def summarize_similarity(corr: pd.DataFrame) -> dict:
    values = corr.values
    tri = values[np.triu_indices_from(values, k=1)]  # upper triangle, no diagonal

    nan_count = int(np.isnan(tri).sum())
    valid = tri[~np.isnan(tri)]

    return {
        "num_genes": int(corr.shape[0]),
        "num_pairs_total": int(len(tri)),
        "num_pairs_nan": nan_count,
        "num_pairs_valid": int(len(valid)),
        "mean_correlation": float(np.mean(valid)) if len(valid) else float("nan"),
        "median_correlation": float(np.median(valid)) if len(valid) else float("nan"),
        "min_correlation": float(np.min(valid)) if len(valid) else float("nan"),
        "max_correlation": float(np.max(valid)) if len(valid) else float("nan"),
    }


def save_similarity_matrix(corr: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        corr.to_csv(output_path)
    elif suffix in {".tsv", ".txt"}:
        corr.to_csv(output_path, sep="\t")
    elif suffix == ".parquet":
        corr.to_parquet(output_path)
    else:
        output_path = output_path.with_suffix(".csv")
        logger.warning("Unrecognized output extension; saving as CSV: %s", output_path)
        corr.to_csv(output_path)

    logger.info("Saved Pearson similarity matrix to: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: compute Pearson gene-gene similarity matrix."
    )
    parser.add_argument("--input", required=True, help="Path to cleaned expression matrix (output of Phase 1).")
    parser.add_argument("--output", required=True, help="Path to save the Pearson similarity matrix.")
    parser.add_argument("--sep", default=None, help="Column separator for text files (default: inferred from extension).")

    orientation = parser.add_mutually_exclusive_group()
    orientation.add_argument("--genes-as-rows", action="store_true", default=True, help="Genes are rows, samples are columns (default).")
    orientation.add_argument("--samples-as-rows", action="store_true", help="Samples are rows, genes are columns.")

    parser.add_argument("--min-periods", type=int, default=3, help="Minimum overlapping non-missing values per gene pair (default: 3).")

    args = parser.parse_args()

    genes_as_rows = not args.samples_as_rows
    logger.info("Matrix orientation: %s", "genes as rows" if genes_as_rows else "samples as rows")

    expr = load_expression_matrix(input_path=args.input, sep=args.sep, genes_as_rows=genes_as_rows)
    expr = clean_expression_matrix(expr)

    corr = compute_pearson_similarity(expr, min_periods=args.min_periods)
    summary = summarize_similarity(corr)

    logger.info("Pearson similarity matrix computed successfully.")
    for key, value in summary.items():
        logger.info("  %s: %s", key, value)

    if summary["num_pairs_nan"] > 0:
        nan_pct = 100 * summary["num_pairs_nan"] / summary["num_pairs_total"]
        logger.warning(
            "%.1f%% of gene pairs have NaN correlation. "
            "Consider checking your min_periods setting or data quality.",
            nan_pct,
        )

    save_similarity_matrix(corr, args.output)


if __name__ == "__main__":
    main()
