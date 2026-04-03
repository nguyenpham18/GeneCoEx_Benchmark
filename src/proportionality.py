import argparse
import pandas as pd
import numpy as np
import warnings

def calculate_proportionality(df):
    """
    Calculates the proportionality (rho) matrix for a gene expression dataframe.
    
    Expected format: 
    - Rows: Genes
    - Columns: Samples
    """
    
    # 1. Centered Log-Ratio (CLR) Transformation
    # We take the natural log of the dataframe.
    log_df = np.log(df)
    
    # Calculate the mean of the logs for each sample (column). 
    # skipna=True ensures we don't crash if a sample had a zero/NaN.
    # Note: The mean of the logs is mathematically equivalent to the log of the geometric mean.
    log_geom_means = log_df.mean(axis=0, skipna=True)
    
    # Subtract the log geometric mean from the log values (this is the clr transform)
    clr_df = log_df - log_geom_means
    
    # 2. Covariance and Variance
    # Transpose the clr_df so genes are columns, allowing us to get a Gene x Gene covariance matrix
    # pandas .cov() automatically handles NaNs by computing pairwise covariance
    cov_matrix = clr_df.T.cov()
    
    # 3a. Extract the variance of each gene.
    # .var() matches the diagonal of the covariance matrix
    gene_variances = clr_df.T.var()
    
    # 3b. Create a matrix of the sum of variances: var(clr(X)) + var(clr(Y))
    # We use numpy broadcasting to create a Gene x Gene grid of these sums
    v_array = gene_variances.values
    var_sum_matrix = pd.DataFrame(
        v_array[:, None] + v_array[None, :], 
        index=cov_matrix.index, 
        columns=cov_matrix.columns
    )
    
    # 4. Calculate Rho
    rho_matrix = (2 * cov_matrix) / var_sum_matrix
    
    return rho_matrix

# Execution Block
if __name__ == "__main__":
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Calculate Proportionality (Rho) Similarity Matrix")
    parser.add_argument("--input", required=True, help="Path to preprocessed counts CSV")
    parser.add_argument("--output", required=True, help="Path to save the similarity matrix CSV")
    args = parser.parse_args()
    
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print("Note: This may take several minutes.")
    
    # 1. Read the CSV
    try:
        df_input = pd.read_csv(args.input, index_col=0)
    except FileNotFoundError:
        print(f"\nError: Could not find the file '{args.input}'. Please check the path.")
        exit(1)
        
    # 2. Reverse log1p and handle zeros simultaneously
    # Since the input is ln(counts + 1), taking the exponent yields (counts + 1)
    # This returns the data to linear space and applies a baseline pseudo-count of 1.
    df_input = np.exp(df_input)
        
    # 3. Run the calculation
    rho_matrix = calculate_proportionality(df_input)
    
    # Get dimensions for the success message
    rows, cols = rho_matrix.shape
    
    # 4. Export the output to a new CSV
    rho_matrix.to_csv(args.output)
    
    print(f"Success! Output is a {rows:,} × {cols:,} matrix.")