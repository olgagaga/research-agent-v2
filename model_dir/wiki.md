# Problem: rare-event medical detection (mammography)

You are optimizing a binary classifier that flags **breast-cancer
microcalcifications** from 6 numeric image-derived features.

## Data (fixed — do not touch data.py)
- Source: OpenML `mammography`, 11,183 samples, 6 numeric features.
- **Severe class imbalance: ~2.3% positive.** This is the whole challenge.
- Fixed, seeded, stratified 60/20/20 train/val/test split. Every experiment is
  scored on identical data, so score changes are caused by *your* edits.

## Objective
- **Maximize `val/auprc`** — area under the precision–recall curve (a.k.a.
  average precision) on the validation set. This is the correct metric for rare
  events; accuracy is useless here (predicting all-negative scores 97.7%).
- `test/auprc` is reported for reference but you optimize `val/auprc`.
- Naive baseline AUPRC ≈ prevalence ≈ 0.023. Strong pipelines reach ~0.6–0.75.

## Your levers (edit ONE group per experiment)
1. **model.py** — architecture. Baseline is a tiny 1-hidden-layer MLP (width 16).
   Try depth, width, dropout, bat/layer-norm, residuals.
2. **loss.py** — the loss. Baseline is plain BCE, which *ignores* the imbalance.
   High-value: `pos_weight` in BCE, focal loss, class-balanced loss.
3. **optimizer.py** — optimiser + LR schedule. Baseline is plain SGD lr=0.01.
   Try Adam/AdamW, weight decay, cosine/step schedules.
4. **transforms.py + config.yaml** — feature engineering + resampling +
   hyperparams. Baseline is identity + no resampling. Try standardisation,
   feature interactions/log transforms, oversampling the positive class, and
   tuning epochs/batch_size.

## Guidance
- The imbalance-aware levers (loss weighting, resampling, feature scaling)
  usually matter most early — a tiny MLP on raw features with unweighted BCE
  barely learns the minority class.
- Change ONE thing at a time so you can attribute the effect.
- Keep every function's name and signature exactly as run.py expects (see the
  contracts in each file's docstring). If you break a contract, the run crashes
  and is reverted.
