#!/usr/bin/env python -u
"""
Bootstrap 95% CIs for imputation, forecasting, and hypo-risk tables.
Run from /mnt/workspace/tvae inside Docker:
    python -u scripts/bootstrap_ci.py 2>&1 | tee results/stage2/bootstrap_ci.log

Outputs:
    results/stage2/bootstrap_ci.json   — all CIs and p-values
    results/stage2/forecasting/gn_run01/predictions.npz  — cached per-window preds
"""

import os, gc, json, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score

# ── Bootstrap primitives ───────────────────────────────────────────────────────

N_BOOT = 1000
_RNG   = np.random.default_rng(42)


def _resample_indices(n_obs):
    return _RNG.integers(0, n_obs, n_obs)


def boot_ci(fn, *arrays, n=N_BOOT):
    """Bootstrap 95% CI for a scalar fn(*arrays) — resamples first axis."""
    n_obs = len(arrays[0])
    vals  = []
    for _ in range(n):
        idx = _resample_indices(n_obs)
        vals.append(fn(*[a[idx] for a in arrays]))
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_boot_p(fn, y_true, score_a, score_b, n=N_BOOT):
    """Two-sided paired bootstrap p-value for H0: fn(y,a) == fn(y,b).
    Uses the shift method: centres bootstrap deltas at the observed delta."""
    n_obs    = len(y_true)
    obs_diff = fn(y_true, score_a) - fn(y_true, score_b)
    boot_diffs = []
    for _ in range(n):
        idx = _resample_indices(n_obs)
        d = fn(y_true[idx], score_a[idx]) - fn(y_true[idx], score_b[idx])
        boot_diffs.append(d)
    boot_diffs = np.array(boot_diffs)
    centred    = boot_diffs - boot_diffs.mean()   # shift to H0 (delta=0)
    p = float(np.mean(np.abs(centred) >= abs(obs_diff)))
    return p


results = {}

# ──────────────────────────────────────────────────────────────────────────────
# 1. HYPO RISK  — thesis table: raw_lstm, fm_ft_lstm, fm_decoder_ft_lstm
#                 predictions.npz already has exactly these (+ naive)
# ──────────────────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('1. HYPO RISK')
print('='*70)

HYPO_PREDS = 'results/stage2/hypo_risk/gn_run01/predictions.npz'
THESIS_HYPO = ['raw_lstm', 'fm_ft_lstm', 'fm_decoder_ft_lstm']

d      = np.load(HYPO_PREDS)
y_hypo = d['y_true'].astype(np.float64)
print(f'  N = {len(y_hypo)}, prevalence = {y_hypo.mean():.3f}')

hypo_scores = {tag: d[tag].astype(np.float64) for tag in THESIS_HYPO}

results['hypo_risk'] = {}
for tag, s in hypo_scores.items():
    auroc  = float(roc_auc_score(y_hypo, s))
    auprc  = float(average_precision_score(y_hypo, s))
    ci_roc = boot_ci(roc_auc_score, y_hypo, s)
    ci_prc = boot_ci(average_precision_score, y_hypo, s)
    results['hypo_risk'][tag] = {
        'auroc': auroc, 'auroc_ci': list(ci_roc),
        'auprc': auprc, 'auprc_ci': list(ci_prc),
    }
    print(f'  {tag:25s}  '
          f'AUROC={auroc:.4f} [{ci_roc[0]:.4f}, {ci_roc[1]:.4f}]  '
          f'AUPRC={auprc:.4f} [{ci_prc[0]:.4f}, {ci_prc[1]:.4f}]')

# Paired tests: dec FT vs raw, enc FT vs raw
for a_tag, b_tag in [('fm_decoder_ft_lstm', 'raw_lstm'), ('fm_ft_lstm', 'raw_lstm')]:
    p   = paired_boot_p(roc_auc_score, y_hypo, hypo_scores[a_tag], hypo_scores[b_tag])
    key = f'paired_p_{a_tag}_vs_{b_tag}_auroc'
    results['hypo_risk'][key] = p
    print(f'  Paired p-value ({a_tag} vs {b_tag}, AUROC): p = {p:.4f}')


# ──────────────────────────────────────────────────────────────────────────────
# 2. FORECASTING  — load model weights and run inference if not cached
# ──────────────────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('2. FORECASTING')
print('='*70)

FORE_CACHE = 'results/stage2/forecasting/gn_run01/predictions.npz'
FORE_DIR   = 'results/stage2/forecasting/gn_run01'

FORE_NEED = {'raw_lstm', 'fm_ft_lstm', 'fm_decoder_ft_lstm', 'naive'}

_cache_ok = False
if os.path.exists(FORE_CACHE):
    _fp = np.load(FORE_CACHE)
    _cache_ok = FORE_NEED <= set(_fp.files)
    _fp.close()

if _cache_ok:
    print('  Loading cached predictions...')
    fp          = np.load(FORE_CACHE)
    y_fore_true = fp['y_true'].astype(np.float32)
    fore_preds  = {k: fp[k].astype(np.float32) for k in fp.files if k != 'y_true'}
    print(f'  N = {len(y_fore_true)}, variants = {list(fore_preds)}')
else:
    print('  No cache (or incomplete) — running inference (needs GPU/Docker)...')
    import h5py
    import tensorflow as tf
    from tensorflow import keras
    from src.encoder import load_encoder, build_decoder
    from src.stage2.data import (load_all_patients, make_forecasting_dataset,
                                  make_eval_dataset, naive_forecast)
    from src.stage2.models import (build_raw_forecasting_lstm,
                                    build_forecasting_lstm, build_forecasting_lstm_decoder)

    def _load_dec_ft_weights_manual(model, wpath):
        """Load fm_decoder_ft weights by semantic position, bypassing the
        naming mismatch between old code (prefix_lm_block / norm1) and
        current code (causal_i / layer_normalization_N).

        Model variable order (89 vars):
          [0-1]   input_proj kernel/bias
          [2-81]  5 × causal block (16 vars each):
                    mhsa q/k/v/out  (8 vars)
                    norm1 γ/β       (2 vars)
                    ffn1 k/b        (2 vars)
                    ffn2 k/b        (2 vars)
                    norm2 γ/β       (2 vars)
          [82-84] lstm kernel/rec_kernel/bias
          [85-88] head_dense k/b, head_out k/b
        """
        with h5py.File(wpath, 'r') as f:
            def rd(*parts):
                return np.array(f['/'.join(parts)])

            L = 'layers'
            F = f'{L}/functional/layers'

            weights = []

            # ── Decoder input projection ──────────────────────────────────
            weights.append(rd(F, 'dense', 'vars', '0'))  # (10, 128)
            weights.append(rd(F, 'dense', 'vars', '1'))  # (128,)

            # ── 5 transformer blocks ───────────────────────────────────────
            for i in range(5):
                bn  = 'prefix_lm_block' if i == 0 else f'prefix_lm_block_{i}'
                blk = f'{F}/{bn}'
                # mhsa: query, key, value, output
                weights.append(rd(blk, 'mhsa', '_query_dense',  'vars', '0'))  # (128,4,32)
                weights.append(rd(blk, 'mhsa', '_query_dense',  'vars', '1'))  # (4,32)
                weights.append(rd(blk, 'mhsa', '_key_dense',    'vars', '0'))  # (128,4,32)
                weights.append(rd(blk, 'mhsa', '_key_dense',    'vars', '1'))  # (4,32)
                weights.append(rd(blk, 'mhsa', '_value_dense',  'vars', '0'))  # (128,4,32)
                weights.append(rd(blk, 'mhsa', '_value_dense',  'vars', '1'))  # (4,32)
                weights.append(rd(blk, 'mhsa', '_output_dense', 'vars', '0'))  # (4,32,128)
                weights.append(rd(blk, 'mhsa', '_output_dense', 'vars', '1'))  # (128,)
                # norm1, ffn, norm2
                weights.append(rd(blk, 'norm1', 'vars', '0'))   # (128,)
                weights.append(rd(blk, 'norm1', 'vars', '1'))   # (128,)
                weights.append(rd(blk, 'ffn1',  'vars', '0'))   # (128,256)
                weights.append(rd(blk, 'ffn1',  'vars', '1'))   # (256,)
                weights.append(rd(blk, 'ffn2',  'vars', '0'))   # (256,128)
                weights.append(rd(blk, 'ffn2',  'vars', '1'))   # (128,)
                weights.append(rd(blk, 'norm2', 'vars', '0'))   # (128,)
                weights.append(rd(blk, 'norm2', 'vars', '1'))   # (128,)

            # ── LSTM ───────────────────────────────────────────────────────
            weights.append(rd(L, 'lstm', 'cell', 'vars', '0'))  # (32,512)
            weights.append(rd(L, 'lstm', 'cell', 'vars', '1'))  # (128,512)
            weights.append(rd(L, 'lstm', 'cell', 'vars', '2'))  # (512,)

            # ── Head dense layers ──────────────────────────────────────────
            weights.append(rd(L, 'dense',   'vars', '0'))  # head_dense kernel (128,64)
            weights.append(rd(L, 'dense',   'vars', '1'))  # head_dense bias   (64,)
            weights.append(rd(L, 'dense_1', 'vars', '0'))  # head_out kernel   (64,1)
            weights.append(rd(L, 'dense_1', 'vars', '1'))  # head_out bias     (1,)

        assert len(weights) == len(model.variables), \
            f'Weight count mismatch: {len(weights)} in file vs {len(model.variables)} in model'
        for i, (w, v) in enumerate(zip(weights, model.variables)):
            assert w.shape == tuple(v.shape), \
                f'Shape mismatch at var {i}: file {w.shape} vs model {tuple(v.shape)} ({v.name})'
        model.set_weights(weights)
        print(f'    Loaded {len(weights)} vars via manual mapping (naming-mismatch workaround)')

    patients = load_all_patients('data/processed/adults_global_norm')
    splits   = make_forecasting_dataset(patients, batch_size=256)

    y_fore_true = None
    fore_preds  = {}

    def _build_dec_ft_model():
        dec = build_decoder()
        dec(tf.zeros((1, 288, 10)))
        dec.trainable = True
        return build_forecasting_lstm_decoder(dec)

    # Thesis table: naive, raw, enc FT, dec FT — no frozen variants.
    FORE_VARIANTS = [
        ('raw',           'raw_lstm',           lambda: build_raw_forecasting_lstm(), False),
        ('fm_ft',         'fm_ft_lstm',         lambda: build_forecasting_lstm(load_encoder(trainable=True)), False),
        ('fm_decoder_ft', 'fm_decoder_ft_lstm', _build_dec_ft_model, True),
    ]

    for _, tag, builder, manual_load in FORE_VARIANTS:
        wpath = os.path.join(FORE_DIR, f'weights_{tag}.weights.h5')
        if not os.path.exists(wpath):
            print(f'  {tag}: weights not found, skipping')
            continue
        keras.backend.clear_session()
        gc.collect()
        print(f'  {tag}...')
        model = builder()
        model({'window': tf.zeros((1, 288, 10)), 'last_cgm': tf.zeros((1, 1))})
        if manual_load:
            _load_dec_ft_weights_manual(model, wpath)
        else:
            model.load_weights(wpath)

        ys, ps = [], []
        for xb, yb in make_eval_dataset(splits['test_patients'], 256):
            ys.append(yb.numpy())
            ps.append(model.predict_on_batch(xb))
        y_test = np.concatenate(ys, axis=0).astype(np.float32)
        y_pred = np.concatenate(ps, axis=0).astype(np.float32)

        if y_fore_true is None:
            y_fore_true = y_test
        fore_preds[tag] = y_pred
        print(f'    N={len(y_test)}, RMSE_t5={np.sqrt(np.mean((y_pred[:,0]-y_test[:,0])**2)):.2f}')
        del model
        gc.collect()

    # Naive baseline
    naive_pred, naive_true = naive_forecast(splits['test_patients'])
    fore_preds['naive'] = naive_pred.astype(np.float32)
    if y_fore_true is None:
        y_fore_true = naive_true.astype(np.float32)

    np.savez(FORE_CACHE, y_true=y_fore_true, **fore_preds)
    print(f'  Saved cached predictions to {FORE_CACHE}')

print(f'  N = {len(y_fore_true)} windows')

# t+5 = h0, t+30 = h5, t+120 = h23
HORIZONS = [('t+5', 0), ('t+30', 5), ('t+120', 23)]

def rmse(yt, yp):
    return float(np.sqrt(np.mean((yp - yt) ** 2)))

results['forecasting'] = {}
for tag, yp in fore_preds.items():
    results['forecasting'][tag] = {}
    row_parts = []
    for hlabel, hidx in HORIZONS:
        fn   = lambda yt, yp_: rmse(yt[:, hidx], yp_[:, hidx])
        val  = rmse(y_fore_true[:, hidx], yp[:, hidx])
        ci   = boot_ci(fn, y_fore_true, yp)
        results['forecasting'][tag][hlabel] = {'rmse': val, 'rmse_ci': list(ci)}
        row_parts.append(f'{hlabel}={val:.2f} [{ci[0]:.2f},{ci[1]:.2f}]')
    print(f'  {tag:25s}  ' + '  '.join(row_parts))

# Paired bootstrap p-value at t+30: raw vs fm_ft (or dec_ft)
for a_tag, b_tag in [('raw_lstm', 'fm_ft_lstm'), ('raw_lstm', 'fm_decoder_ft_lstm')]:
    if a_tag in fore_preds and b_tag in fore_preds:
        fn30 = lambda yt, yp: rmse(yt[:, 5], yp[:, 5])
        p    = paired_boot_p(fn30, y_fore_true, fore_preds[a_tag], fore_preds[b_tag])
        key  = f'paired_p_{a_tag}_vs_{b_tag}_t30'
        results['forecasting'][key] = p
        print(f'  Paired p-value ({a_tag} vs {b_tag}, t+30): p = {p:.4f}')


# ──────────────────────────────────────────────────────────────────────────────
# 3. IMPUTATION  — re-run zero-shot + raw forward passes
# ──────────────────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('3. IMPUTATION')
print('='*70)

IMP_CACHE = 'results/stage2/imputation/gn_run01/predictions.npz'

if os.path.exists(IMP_CACHE):
    print('  Loading cached imputation predictions...')
    ip = np.load(IMP_CACHE)
    imp_data = {k: ip[k] for k in ip.files}
    print(f'  Keys: {list(imp_data)}')
else:
    print('  No cache — running forward passes (needs GPU/Docker)...')
    import tensorflow as tf
    from tensorflow import keras
    from src.stage2.data import (load_all_patients, make_imputation_dataset,
                                  make_eval_imputation_numpy,
                                  IMPUTATION_GAP_LENGTHS, IMPUTATION_GAP_LABELS)
    from src.stage2.models import build_mtsm_imputation_model, build_raw_imputation_model
    from src.stage2.evaluate import linear_interpolate

    patients = load_all_patients('data/processed/adults_global_norm')
    splits   = make_imputation_dataset(patients, batch_size=128)

    mtsm = build_mtsm_imputation_model()
    raw  = build_raw_imputation_model()
    raw.load_weights('results/stage2/imputation/gn_run01/raw_weights.weights.h5')
    print('  Models loaded')

    imp_data = {}
    for gap_len, gap_label in zip(IMPUTATION_GAP_LENGTHS, IMPUTATION_GAP_LABELS):
        print(f'  Gap {gap_label}...')
        ev = make_eval_imputation_numpy(splits['test_patients'], gap_len)
        W_masked = ev['windows_masked']
        masks    = ev['masks']
        true_z   = ev['windows_orig'][:, :, 0]   # CGM z-score (N, 288)

        mtsm_z = mtsm.predict(W_masked, batch_size=256, verbose=0)
        raw_z  = raw.predict(W_masked,  batch_size=256, verbose=0)
        lin_z  = linear_interpolate(W_masked, masks)

        imp_data[f'{gap_label}_true_z']   = true_z.astype(np.float32)
        imp_data[f'{gap_label}_masks']    = masks.astype(np.float32)
        imp_data[f'{gap_label}_s_mean']   = ev['scaler_means'].astype(np.float32)
        imp_data[f'{gap_label}_s_std']    = ev['scaler_stds'].astype(np.float32)
        imp_data[f'{gap_label}_fm_z']     = mtsm_z.astype(np.float32)
        imp_data[f'{gap_label}_raw_z']    = raw_z.astype(np.float32)
        imp_data[f'{gap_label}_linear_z'] = lin_z.astype(np.float32)
        print(f'    N={len(true_z)} windows')

    np.savez(IMP_CACHE, **imp_data)
    print(f'  Saved cached predictions to {IMP_CACHE}')


def imp_rmse_mg(true_z, pred_z, masks, s_mean, s_std):
    m    = masks.astype(bool)
    sm   = s_mean[:, np.newaxis]
    ss   = s_std[:, np.newaxis]
    t_mg = (true_z * ss + sm)[m]
    p_mg = (pred_z * ss + sm)[m]
    return float(np.sqrt(np.mean((p_mg - t_mg) ** 2)))


def imp_r2_z(true_z, pred_z, masks):
    m = masks.astype(bool)
    return float(r2_score(true_z[m], pred_z[m]))


# Each bootstrap resample is over windows (first axis)
def imp_boot_ci_rmse(true_z, pred_z, masks, s_mean, s_std, n=N_BOOT):
    n_obs = len(true_z)
    vals  = []
    for _ in range(n):
        idx = _resample_indices(n_obs)
        vals.append(imp_rmse_mg(true_z[idx], pred_z[idx],
                                masks[idx], s_mean[idx], s_std[idx]))
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def imp_boot_ci_r2(true_z, pred_z, masks, n=N_BOOT):
    n_obs = len(true_z)
    vals  = []
    for _ in range(n):
        idx = _resample_indices(n_obs)
        vals.append(imp_r2_z(true_z[idx], pred_z[idx], masks[idx]))
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


GAP_LABELS = ['4h', '5h', '6h', '8h']
METHODS    = [('fm', 'FM'), ('raw', 'Raw'), ('linear', 'Linear')]

results['imputation'] = {}
for gl in GAP_LABELS:
    true_z = imp_data[f'{gl}_true_z']
    masks  = imp_data[f'{gl}_masks']
    s_mean = imp_data[f'{gl}_s_mean']
    s_std  = imp_data[f'{gl}_s_std']
    print(f'\n  Gap {gl} (N={len(true_z)} windows):')
    results['imputation'][gl] = {}

    for tag, label in METHODS:
        pred_z = imp_data[f'{gl}_{tag}_z']
        rmse_v = imp_rmse_mg(true_z, pred_z, masks, s_mean, s_std)
        r2_v   = imp_r2_z(true_z, pred_z, masks)
        ci_rmse = imp_boot_ci_rmse(true_z, pred_z, masks, s_mean, s_std)
        ci_r2   = imp_boot_ci_r2(true_z, pred_z, masks)
        results['imputation'][gl][tag] = {
            'rmse_mg': rmse_v, 'rmse_mg_ci': list(ci_rmse),
            'r2_z':    r2_v,   'r2_z_ci':    list(ci_r2),
        }
        print(f'    {label:8s}  RMSE={rmse_v:.2f} [{ci_rmse[0]:.2f},{ci_rmse[1]:.2f}]'
              f'  R²={r2_v:.4f} [{ci_r2[0]:.4f},{ci_r2[1]:.4f}]')


# ── Save all results ───────────────────────────────────────────────────────────
OUT = 'results/stage2/bootstrap_ci.json'
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved → {OUT}')
