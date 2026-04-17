import pandas as pd
import numpy as np
import os


# path for raw count file, annotation file and output directory 
RAW_COUNTS = "data/raw/gene_reads_v11_brain_cortex.gct.gz"
ANNOTATION  = "data/annotation/gencode.v47.genes.gtf"
OUT_DIR     = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# step 1: load the raw count
# read raw gene count matrix from the GTEx file and remove the Ensembl version number from gene IDs
print("Loading raw counts...")
counts = pd.read_csv(RAW_COUNTS, sep="\t", skiprows=2, index_col=0)
counts = counts.drop(columns=["Description"]).astype(float)
counts.index = counts.index.str.split(".").str[0]  # strip Ensembl version
print(f"  Shape: {counts.shape}")

#step 2: sample SQ
# remove any sample where more than 50% of genes have 0 counts - which indicates a failed or low-quality library
# it's important because large amount of zero would contribute noise rather signal
print("Sample QC...")
zero_frac = (counts == 0).mean(axis=0)
counts = counts.loc[:, zero_frac < 0.5]
print(f"  Samples remaining: {counts.shape[1]}")

# step 3: gene filtering cpm-based
# we convert raw counts to CPM (count per million) and remove genes that don't reach at least 1 CPM in at least 20% samples
# genes with near zero expression across many sample would not be as informative for co-expression analysis, they can potentially hurt the network
print("Gene filtering (CPM >= 1 in >= 20% samples)...")
cpm = counts.divide(counts.sum(axis=0), axis=1) * 1e6
min_samples = int(0.2 * counts.shape[1])
counts = counts.loc[(cpm >= 1).sum(axis=1) >= min_samples]
print(f"  Genes remaining: {counts.shape[0]}")

# step 4: gene type filtering
# only keep protein_coding, lncRNA and atisense (long RNA gene types_ since they aree most common gene types , shorter read make mapping more difficult )
print("Gene type filtering...")
rows = []
with open(ANNOTATION) as f:
    for line in f:
        if line.startswith("#"):
            continue
        fields = line.strip().split("\t")
        if fields[2] != "gene":
            continue
        attrs = {}
        for attr in fields[8].split(";"):
            attr = attr.strip()
            if attr:
                key, _, val = attr.partition(" ")
                attrs[key] = val.strip('"')
        rows.append({
            "gene_id"  : attrs.get("gene_id", "").split(".")[0],
            "gene_type": attrs.get("gene_type", "")
        })

annot = pd.DataFrame(rows)
keep_types = ["protein_coding", "lncRNA", "antisense"]
keep_ids = annot[annot["gene_type"].isin(keep_types)]["gene_id"].values
counts = counts[counts.index.isin(keep_ids)]
print(f"  Genes remaining: {counts.shape[0]}")

# step 5: CTF normalization 
# calculate TMM (Trimmed mean of M-values) scaling factors for each sample and divides the raw counts by those faction 

print("CTF normalization...")

def calc_tmm_factors(counts):
    lib_sizes = counts.sum(axis=0)
    f75 = counts.divide(lib_sizes, axis=1).quantile(0.75)
    ref_col = (f75 - f75.mean()).abs().idxmin()
    ref = counts[ref_col]
    ref_size = lib_sizes[ref_col]
    factors = {}
    for col in counts.columns:
        samp = counts[col]
        samp_size = lib_sizes[col]
        with np.errstate(divide='ignore', invalid='ignore'):
            lfc = np.log2((samp / samp_size) / (ref / ref_size))
            mag = 0.5 * np.log2((samp / samp_size) * (ref / ref_size))
        mask = np.isfinite(lfc) & np.isfinite(mag) & (samp > 0) & (ref > 0)
        lfc_f = lfc[mask]
        mag_f = mag[mask]
        lfc_lo, lfc_hi = lfc_f.quantile(0.15), lfc_f.quantile(0.85)
        mag_lo, mag_hi = mag_f.quantile(0.025), mag_f.quantile(0.975)
        keep = ((lfc_f >= lfc_lo) & (lfc_f <= lfc_hi) &
                (mag_f >= mag_lo) & (mag_f <= mag_hi))
        w = 1 / samp[mask][keep] + 1 / ref[mask][keep]
        factors[col] = 2 ** (np.average(lfc_f[keep], weights=1/w))
    factors = pd.Series(factors)
    factors = factors / np.exp(np.mean(np.log(factors)))
    return factors

tmm_factors = calc_tmm_factors(counts)
ctf_matrix = counts.divide(tmm_factors, axis=1)
print("  CTF normalization done")

# ══════════════════════════════════════════════════════
# STEP 6: log1p TRANSFORMATION
# ══════════════════════════════════════════════════════
print("STEP 6: log1p transformation...")
transformed_matrix = np.log1p(ctf_matrix)
print("  log1p transformation done")


# ══════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════
print("Saving...")
# Save both versions in preprocessing
ctf_matrix.to_csv(f"{OUT_DIR}/ctf_counts.csv")           # for proportionality
transformed_matrix.to_csv(f"{OUT_DIR}/preprocessed_counts.csv")
print(f"  → data/processed/preprocessed_counts.csv")

print(f"\nFinal matrix: {transformed_matrix.shape[0]} genes × {transformed_matrix.shape[1]} samples")
print("Preprocessing complete!")

