import pandas as pd
import numpy as np
import warnings
import argparse

def calculate_proportionality(df):
    """
    Calculates the proportionality (rho) matrix for a gene expression dataframe.
    
    Implements the proportionality correlation coefficient rho_p from:
    Lovell et al. (2015) "Proportionality: A Valid Alternative to Correlation for Relative Data"
    PLoS Computational Biology 11(3): e1004075
    
    Formula: rho_p(log x, log y) = 2*cov(clr(x), clr(y)) / (var(clr(x)) + var(clr(y)))
    
    Expected format: 
    - Rows: Genes
    - Columns: Samples
    
    Note: Zero handling and CPM normalization should be applied BEFORE calling this function.
    """
    
    # 1. Centered Log-Ratio (CLR) Transformation
    # Take the natural log of the dataframe
    log_df = np.log(df)
    
    # Calculate the mean of the logs for each sample (column)
    # mean(log(x)) == log(geometric_mean(x)) -- this is the CLR denominator
    # axis=0 means we take the mean across genes for each sample
    log_geom_means = log_df.mean(axis=0, skipna=True)
    
    # Subtract the log geometric mean from the log values (CLR transform)
    # This ensures clr values sum to zero within each sample
    clr_df = log_df - log_geom_means
    
    # 2. Covariance Matrix
    # Transpose so genes are columns -> gives Gene x Gene covariance matrix
    # pandas .cov() handles NaNs by computing pairwise covariance
    cov_matrix = clr_df.T.cov()
    
    # 3. Variance of each gene
    # .var() matches the diagonal of the covariance matrix
    gene_variances = clr_df.T.var()
    
    # 4. Create matrix of variance sums: var(clr(X)) + var(clr(Y))
    # numpy broadcasting creates a Gene x Gene grid
    v_array = gene_variances.values
    var_sum_matrix = pd.DataFrame(
        v_array[:, None] + v_array[None, :],
        index=cov_matrix.index,
        columns=cov_matrix.columns
    )
    
    # 5. Calculate Rho_p
    # rho_p = 2 * cov(clr(X), clr(Y)) / (var(clr(X)) + var(clr(Y)))
    # Ranges from -1 (perfect reciprocality) to +1 (perfect proportionality)
    rho_matrix = (2 * cov_matrix) / var_sum_matrix
    
    return rho_matrix


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Calculate Proportionality (Rho_p) Similarity Matrix. "
                    "Implements Lovell et al. (2015) PLoS Comp Bio."
    )
    parser.add_argument("--input",  required=True, help="Path to counts CSV (raw or CTF-normalized)")
    parser.add_argument("--output", required=True, help="Path to save the similarity matrix CSV")
    args = parser.parse_args()
    
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print("Note: This may take several minutes for large matrices.")
    
    # 1. Read the CSV
    try:
        df_input = pd.read_csv(args.input, index_col=0)
    except FileNotFoundError:
        print(f"\nError: Could not find the file '{args.input}'. Please check the path.")
        exit(1)
    
    print(f"Loaded matrix: {df_input.shape[0]:,} genes x {df_input.shape[1]:,} samples")
    
    # 2. Add pseudo-count BEFORE normalization
    # Per Lovell et al. (2015): zeros are a known challenge in CoDA.
    # Adding pseudo-count before CPM preserves scale-invariance of the CLR transform.
    if (df_input == 0).any().any():
        n_zeros = (df_input == 0).sum().sum()
        warnings.warn(
            f"{n_zeros:,} zeros detected. Applying pseudo-count of 1 before CPM normalization. "
            "This preserves compositional ratios per CoDA principles.",
            UserWarning
        )
        df_input = df_input + 1
    
    # 3. CPM normalization to prevent numerical overflow in log transform
    # Dividing each sample by its library size preserves compositional ratios
    # (scale invariance principle from CoDA -- Lovell et al. 2015, p.5)
    print("Applying CPM normalization...")
    df_input = df_input.divide(df_input.sum(axis=0), axis=1) * 1e6
    
    print(f"  Value range after CPM: [{df_input.values.min():.4f}, {df_input.values.max():.4f}]")
    
    # 4. Run the proportionality calculation
    print("Computing CLR transform and rho_p matrix...")
    rho_matrix = calculate_proportionality(df_input)
    
    # 5. Sanity check
    n_nans = rho_matrix.isna().sum().sum()
    if n_nans > 0:
        warnings.warn(f"{n_nans:,} NaN values in output matrix. Check input data for issues.")
    else:
        print("  No NaN values detected in output matrix. ✓")
    
    print(f"  Rho range: [{rho_matrix.values[~np.isnan(rho_matrix.values)].min():.4f}, "
          f"{rho_matrix.values[~np.isnan(rho_matrix.values)].max():.4f}]")
    
    # 6. Save
    print(f"Saving to {args.output}...")
    rho_matrix.to_csv(args.output)
    
    rows, cols = rho_matrix.shape
    print(f"Success! Output is a {rows:,} x {cols:,} matrix.")