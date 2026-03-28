import pandas as pd
import numpy as np

print('Loading matrix...')
df = pd.read_csv('data/similarity/pearson_similarity.csv', index_col=0)

# Basic info
print('\n--- Basic Info ---')
print(f'Shape:          {df.shape}')
print(f'Genes:          {df.shape[0]}')

# Get upper triangle only (avoid diagonal and duplicates)
vals = df.values
tri = vals[np.triu_indices_from(vals, k=1)]

print('\n--- Correlation Stats ---')
print(f'Total pairs:    {len(tri):,}')
print(f'Mean:           {np.mean(tri):.4f}')
print(f'Median:         {np.median(tri):.4f}')
print(f'Std:            {np.std(tri):.4f}')
print(f'Min:            {np.min(tri):.4f}')
print(f'Max:            {np.max(tri):.4f}')

print('\n--- Distribution ---')
print(f'Pairs > 0.8:    {(tri > 0.8).sum():,}  ({100*(tri > 0.8).mean():.2f}%)')
print(f'Pairs > 0.5:    {(tri > 0.5).sum():,}  ({100*(tri > 0.5).mean():.2f}%)')
print(f'Pairs > 0.0:    {(tri > 0.0).sum():,}  ({100*(tri > 0.0).mean():.2f}%)')
print(f'Pairs < 0.0:    {(tri < 0.0).sum():,}  ({100*(tri < 0.0).mean():.2f}%)')
print(f'Pairs < -0.5:   {(tri < -0.5).sum():,}  ({100*(tri < -0.5).mean():.2f}%)')

print('\n--- NaN Check ---')
print(f'NaN values:     {np.isnan(tri).sum()}')