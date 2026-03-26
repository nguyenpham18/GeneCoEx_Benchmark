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
    
    # 1. Zero Handling: Warn and skip (replace with NaN)
    if (df == 0).any().any():
        warnings.warn("Zeros detected in the expression matrix. They will be replaced with NaN and skipped in calculations.", UserWarning)
        df = df.replace(0, np.nan)
        
    # 2. Centered Log-Ratio (CLR) Transformation
    # We take the natural log of the dataframe.
    log_df = np.log(df)
    
    # Calculate the mean of the logs for each sample (column). 
    # skipna=True ensures we don't crash if a sample had a zero/NaN.
    # Note: The mean of the logs is mathematically equivalent to the log of the geometric mean.
    log_geom_means = log_df.mean(axis=0, skipna=True)
    
    # Subtract the log geometric mean from the log values (this is the clr transform)
    clr_df = log_df - log_geom_means
    
    # 3. Covariance and Variance
    # Transpose the clr_df so genes are columns, allowing us to get a Gene x Gene covariance matrix
    # pandas .cov() automatically handles NaNs by computing pairwise covariance
    cov_matrix = clr_df.T.cov()
    
    # Extract the variance of each gene.
    # .var() matches the diagonal of the covariance matrix
    gene_variances = clr_df.T.var()
    
    # Create a matrix of the sum of variances: var(clr(X)) + var(clr(Y))
    # We use numpy broadcasting to create a Gene x Gene grid of these sums
    v_array = gene_variances.values
    var_sum_matrix = pd.DataFrame(
        v_array[:, None] + v_array[None, :], 
        index=cov_matrix.index, 
        columns=cov_matrix.columns
    )
    
    # 4. Calculate Rho
    # rho = 2 * cov(X,Y) / (var(X) + var(Y))
    rho_matrix = (2 * cov_matrix) / var_sum_matrix
    
    return rho_matrix
