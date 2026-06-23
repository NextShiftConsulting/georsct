# DOE R3-TJEPA: Masked Tabular Representation Compatibility Probe

**Status**: DESIGNED
**Date**: 2026-06-22
**Experiment**: s035 model-ladder extension
**Representation tier**: R3 (sibling to R3-DGM, not a replacement)
**Pre-registration**: Hypothesized geometry dissociations declared before execution

---

## Purpose

R3-TJEPA is a **controlled probe** where representation-learning gain is
deliberately minimized in order to isolate the **geometry-compatibility effect**.

JEPA's architectural edge — latent prediction over high-dimensional, redundant
inputs where reconstruction wastes capacity on irrelevant detail — is largely
absent in the tabular ZCTA setting. The 31 R0 features are low-dimensional and
already semantic. This is intentional:

> If a representation-learning method shows a compatibility shift even when its
> structural advantages are absent, the shift is attributable to learned feature
> interactions and latent space geometry, not to the encoder's ability to
> compress redundant inputs.

The question is NOT "can JEPA win on tabular?" The question is:

> **Does learning feature interactions via masked latent prediction produce a
> representation that is compatible with all six crisis decision geometries,
> or does it create geometry-specific failure modes that hand-engineered
> features avoid?**

---

## Three Arms (Not Two)

A two-arm comparison (hand-engineered vs JEPA) cannot distinguish JEPA-specific
effects from generic learned-encoder effects. A reviewer will correctly ask:
"Is this result because of latent prediction, or because any neural encoder
finds interactions your feature list missed?"

| Arm | Encoder | Objective | Target Encoder | EMA | What It Tests |
|-----|---------|-----------|----------------|-----|---------------|
| **R0-baseline** | None (raw features) | N/A | N/A | N/A | Hand-engineered feature quality |
| **R3-MAE** | Masked Tabular AE | Reconstruct masked raw features | None (decoder) | No | Generic learned encoder effect |
| **R3-TJEPA** | Masked Tabular JEPA | Predict masked latent targets | Yes (EMA copy) | Yes | JEPA-specific latent prediction effect |

R3-MAE is the critical control. It uses the **same architecture** (same
tokenizer, same transformer encoder, same capacity, same mask ratio) but with a
reconstruction objective in feature space instead of latent prediction. This is
exactly the thing reviewers will confuse JEPA with — so we measure it.

The attribution logic:

| Comparison | What It Shows |
|---|---|
| R3-TJEPA vs R0 | Does any learned representation change compatibility? |
| R3-MAE vs R0 | Does a generic learned encoder change compatibility? |
| R3-TJEPA vs R3-MAE | Is the change JEPA-specific (latent prediction + EMA)? |

If R3-TJEPA and R3-MAE show the same compatibility pattern, JEPA's
architecture is irrelevant — any learned encoder produces the effect. If they
diverge, latent prediction specifically changes the geometry-compatibility
profile. Both outcomes are publishable.

---

## Spatial Leakage Guard: Per-Fold Pretraining

A learned encoder adds a fitting step the hand-engineered baseline lacks. Under
spatial autocorrelation, pretraining on ZCTAs adjacent to a held-out fold leaks
spatial structure into the embedding.

**Hard rule**: The encoder (TJEPA or MAE) is retrained inside each fold split.

```
For each spatial-blocked fold k:
    1. Split ZCTAs into train_k and test_k (frozen fold assignment)
    2. Pretrain TJEPA/MAE on train_k features ONLY
    3. Encode train_k and test_k using the fold-k encoder
    4. Train downstream solver (HistGBDT, Ridge) on encoded train_k
    5. Evaluate on encoded test_k
```

This means K encoders are trained (one per fold), not one global encoder. The
cost is ~5x pretraining time — acceptable for tabular (minutes, not hours).

**Why this matters**: Without per-fold pretraining, the clustering and transfer
geometry verdicts are contaminated. The encoder has seen spatial neighbors of
the test ZCTAs, so any residual autocorrelation finding is uninterpretable.

For `fold_random` splits, per-fold pretraining is still applied (consistency),
but the leakage risk is lower since random splits don't respect spatial
structure.

For `fold_leave_event_out`, pretraining uses all ZCTAs from non-held-out events
(same spatial coverage, different temporal slice). This tests whether the
encoder learns event-invariant representations.

---

## Pre-Registered Hypotheses

Declaring expected geometry dissociations BEFORE execution. The experiment is
designed and powered to detect these splits.

### H-TJEPA-1: Prediction geometry — marginal improvement expected

TJEPA may capture nonlinear feature interactions that HistGBDT already finds
but Ridge misses. Expected effect size: small (R2 delta < 0.02 for regression,
AUC delta < 0.01 for classification).

**Rationale**: R0 features are already semantic; JEPA's advantage on tabular is
small. If the improvement is large, it suggests the R0 feature set has
important missing interactions, not that JEPA is powerful.

### H-TJEPA-2: Ranking geometry — neutral to slight improvement

JEPA embeddings may improve Spearman/Kendall rank correlation for top-risk
ZCTAs. Expected: neutral (within CI of R0).

### H-TJEPA-3: Clustering geometry — EXPECTED DEGRADATION

**This is the key dissociation.** JEPA embeddings learned from spatially
autocorrelated features will encode spatial proximity into the latent space.
This should increase Moran's I of residuals compared to R0, even with per-fold
pretraining, because the encoder compresses spatial structure into the
embedding.

**Detection**: Compare Moran's I of prediction residuals between R0 and
R3-TJEPA on the same spatial-blocked folds.

### H-TJEPA-4: Transfer geometry — EXPECTED FAILURE

A JEPA encoder trained on Houston ZCTAs learns Houston-specific feature
interactions (coastal distance × elevation × impervious %). These interactions
may not transfer to New Orleans (different topography), NYC (different density
scale), or California AR (different climate regime).

**Detection**: Cross-scenario evaluation. Train encoder + solver on scenario A,
evaluate on scenario B. Compare R0 vs R3-TJEPA cross-scenario R2/AUC.

### H-TJEPA-5: Allocation geometry — EXPECTED NULL

Better prediction score does not automatically improve resource allocation
decisions. JEPA may improve RMSE but produce the same or worse allocation
utility (decision-theoretic loss).

**Detection**: Compute allocation utility metric alongside RMSE. If RMSE
improves but allocation utility does not, this is the headline result.

### H-TJEPA-6: MAE vs TJEPA dissociation — EXPECTED

If H-TJEPA-3 (clustering degradation) occurs for both MAE and TJEPA equally,
the effect is generic to learned encoders, not JEPA-specific. If TJEPA shows
worse clustering than MAE, latent prediction specifically amplifies spatial
structure encoding.

---

## Encoder Specifications

### Shared Architecture (Both Arms)

| Parameter | Value | Rationale |
|---|---|---|
| n_features | 31 (R0 feature count) | Match R0 baseline exactly |
| embed_dim | 128 | Match existing representation dimensions (PCA, GNN) |
| hidden_dim | 256 | 2x embed_dim, standard ratio |
| n_layers | 2 | Minimum for cross-feature attention |
| n_heads | 2 | embed_dim / 64, lightweight |
| mask_ratio | 0.3 | 30% features masked per sample |
| batch_size | 256 | Fits in CPU memory for all scenarios |
| n_epochs | 100 | Sufficient for convergence on <32K samples |
| learning_rate | 1e-3 | AdamW with weight_decay=1e-4 |
| seed | 42 | Match all other experiments |

### TJEPA-Specific

| Parameter | Value |
|---|---|
| ema_decay | 0.996 |
| loss | cosine_distance(predicted_latent, target_latent) + 0.1 * L2 |
| target_encoder | EMA copy of context encoder |
| prediction target | Target encoder output (latent), NOT raw features |

### MAE-Specific

| Parameter | Value |
|---|---|
| loss | MSE(reconstructed_features, original_features) |
| decoder | 2-layer MLP (embed_dim → hidden_dim → n_features) |
| prediction target | Raw masked feature values (feature space) |

### Capacity Match

Both arms have the same encoder architecture. MAE adds a lightweight decoder;
TJEPA adds a predictor + target encoder. Total parameter counts should be
reported and within 20% of each other to prevent capacity confounds.

---

## Folds and Solvers

Identical to R0/R1/R2/R3-DGM:

| Component | Value | Source |
|---|---|---|
| Targets | obs_nfip_event_claims (log1p), obs_has_311, obs_has_hwm | TARGETS in train_r0_baseline.py |
| Solvers | HistGBDT (max_iter=200, max_depth=6), Ridge (alpha=1.0) | Frozen hyperparameters |
| Splits | fold_random, fold_spatial_blocked, fold_leave_event_out | generate_folds.py |
| Scenarios | houston, new_orleans, nyc, southwest_florida, riverside_coachella | SCENARIOS |
| Seed | 42 | Global |

---

## Output Schema

### S3 Keys

```
results/s035/r3_tjepa_{scenario}.json          # TJEPA arm results
results/s035/r3_mae_{scenario}.json             # MAE control arm results
results/s035/r3_tjepa_{scenario}_predictions.parquet  # per-row predictions
results/s035/r3_mae_{scenario}_predictions.parquet
results/s035/r3_tjepa_{scenario}_encoder.pt     # per-fold encoder checkpoints
results/s035/r3_mae_{scenario}_encoder.pt
results/s035/r3_tjepa_{scenario}_stability.json # multi-mask stability metrics
```

### Results JSON Schema

```json
{
  "experiment": "s035-model-ladder",
  "phase": "r3_tjepa",
  "scenario": "houston",
  "representation": "R3-TJEPA",
  "encoder_config": { ... },
  "per_fold_pretraining": true,
  "leakage_guard": "spatial_blocked_retrain",
  "tool_contract": {
    "tool_id": "tjepa_spatial_encoder",
    "version": "0.1.0"
  },
  "runs": [ ... ],
  "stability": {
    "n_mask_seeds": 10,
    "mean_embedding_std": 0.042,
    "certificate_variance": { "R": 0.003, "S": 0.001 }
  }
}
```

---

## Stability Measurement Protocol

JEPA's masking provides a built-in perturbation mechanism for S (Stability):

1. After per-fold pretraining, encode test ZCTAs with 10 different mask seeds
2. Each seed produces a different embedding → different downstream prediction
3. Compute:
   - Embedding-level stability: mean std across mask seeds per ZCTA
   - Prediction-level stability: variance of y_pred across mask seeds
   - Certificate-level stability: variance of R, S values across mask seeds
4. Compare against R0 baseline stability (which has zero mask variance)

High mask sensitivity → low S → compatibility concern for production use.

---

## Compute Requirements

| Resource | Specification | Rationale |
|---|---|---|
| Instance | ml.m5.xlarge (4 vCPU, 16 GB) | CPU sufficient for tabular JEPA |
| Volume | 30 GB EBS | Data + checkpoints |
| Runtime estimate | ~30 min per scenario per arm | 5 folds × 100 epochs × 2 solvers × 3 targets |
| Total per scenario | ~60 min (both arms) | Parallel fold training |
| Total all scenarios | ~5 hours | Sequential scenarios |
| Image | pytorch 2.9 + sklearn | Standard SageMaker PyTorch image |
| pip extras | None beyond torch + sklearn | No new dependencies |

---

## Pre-Submission Checklist

- [ ] Per-fold pretraining implemented and verified (no train-test ZCTA overlap in encoder training)
- [ ] MAE control arm capacity-matched to TJEPA (parameter count within 20%)
- [ ] All 6 geometry verdicts computed for both arms
- [ ] Moran's I computed on residuals for clustering geometry
- [ ] Cross-scenario transfer matrix computed
- [ ] Allocation utility metric computed alongside RMSE/AUC
- [ ] Stability measurement (10 mask seeds) completed
- [ ] Tool contract registered (ADR-052)
- [ ] Results JSON includes encoder_config and per_fold_pretraining flag
- [ ] No reviewer can attribute any finding to spatial leakage, capacity mismatch, or missing control
