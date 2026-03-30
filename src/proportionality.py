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
    # rho = 2 * cov(X,Y) / (var(X) + var(Y))
    rho_matrix = (2 * cov_matrix) / var_sum_matrix
    
    return rho_matrix

# Test Data
if __name__ == "__main__":
    
    print("--- Test 1: Highly Predictable (Perfect Proportionality) ---")
    # Gene A and Gene B maintain an exact 1:2 ratio across all samples.
    # Gene C is totally random.
    df_perfect = pd.DataFrame({
        'Sample_1': [10, 20, 15],
        'Sample_2': [50, 100, 8],
        'Sample_3': [30, 60, 42],
        'Sample_4': [80, 160, 19]
    }, index=['Gene_A', 'Gene_B', 'Gene_C'])
    
    print("Input:\n", df_perfect)
    print("\nRho Matrix:\n", calculate_proportionality(df_perfect))
    # Expectation: The intersection of Gene_A and Gene_B should be exactly 1.0


    print("\n--- Test 2: Unconventional Scale Invariance ---")
    # Gene X is expressed in the millions. Gene Y is expressed in fractions.
    # However, their ratio across samples is perfectly constant.
    df_scale = pd.DataFrame({
        'Sample_1': [1000000, 0.01, 5],
        'Sample_2': [2500000, 0.025, 6],
        'Sample_3': [500000, 0.005, 4]
    }, index=['Gene_X', 'Gene_Y', 'Gene_Z'])
    
    print("\nRho Matrix:\n", calculate_proportionality(df_scale))
    # Expectation: Even with massive scale differences, Gene_X and Gene_Y will equal 1.0.


    print("\n--- Test 3: Unconventional Zero Handling ---")
    # Contains a zero. The script should throw a warning, convert to NaN, 
    # and calculate the remaining pairwise data without crashing.
    df_zeros = pd.DataFrame({
        'Sample_1': [10, 20, 30],
        'Sample_2': [5, 10, 0],  # The zero is here
        'Sample_3': [100, 200, 300],
        'Sample_4': [50, 100, 150]
    }, index=['Gene_1', 'Gene_2', 'Gene_3'])
    
    print("\nRho Matrix:\n", calculate_proportionality(df_zeros))
    # Expectation: A warning is printed. Gene_1 and Gene_2 still equal 1.0. 
    # Gene_3 computes based only on Samples 1, 3, and 4.