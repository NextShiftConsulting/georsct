"""
S019D Interaction Diagnostic: arch_spread vs N_ceiling
Proof artifact for GeoRSCT §4.4 framing decision.

Source file: data/s019d/seed_42/s019d_results.json
S3 mirror:   s3://swarm-yrsn-datasets/rsct_curriculum/series_019/results/s019d/seed_42/s019d_results.json

Run:
    python reproduce_interaction_diagnostic.py data/s019d/seed_42/s019d_results.json

Expected output: Spearman r=0.0714, p=0.7233, n=27
"""
import json, sys, pandas as pd
from scipy import stats

path = sys.argv[1] if len(sys.argv) > 1 else 'data/s019d/seed_42/s019d_results.json'

with open(path) as f:
    results = json.load(f)

df = pd.DataFrame(results)
core3 = ['pca_v1', 'spatial_lag_v1', 'gnn_v2']

task_emb = df.groupby(['target','embedding','target_family'])['r2'].mean().reset_index()
core_df = task_emb[task_emb['embedding'].isin(core3)]
spread = core_df.groupby(['target','target_family'])['r2'].agg(lambda x: x.max()-x.min()).reset_index()
spread.columns = ['target','target_family','arch_spread']
nceil = df.groupby('target')['n_ceiling'].mean().reset_index()
merged = spread.merge(nceil, on='target').sort_values('n_ceiling')

print(f"{'target':32s} {'family':14s} {'arch_spread':>11s} {'n_ceiling':>9s}")
for _, row in merged.iterrows():
    print(f"{row['target']:32s} {row['target_family']:14s} {row['arch_spread']:11.4f} {row['n_ceiling']:9.4f}")

rs, ps = stats.spearmanr(merged['arch_spread'], merged['n_ceiling'])
rp, pp = stats.pearsonr(merged['arch_spread'], merged['n_ceiling'])
print(f"\nSpearman r={rs:.4f}, p={ps:.4f}, n={len(merged)}")
print(f"Pearson  r={rp:.4f}, p={pp:.4f}, n={len(merged)}")
