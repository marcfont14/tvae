"""
Variable justification analysis for the 10 input features.

For each feature group, the question is not just "does the global R² drop?"
but "does removing it hurt in the specific clinical contexts where it matters?"

Analyses:
  1. CGM & PI/RA — covered by feature_ablation.py (clear dominance).
  2. Binary flags (bolus 5, carbs 6):
       - Overall R² with/without (already known: Δ≈−0.02 at 4h).
       - Stratified: windows WITH a bolus/carb event inside the gap vs WITHOUT.
       - Driver direction accuracy (sign test): does the model correctly
         predict glucose direction after the event, with vs without flags?
  3. Therapy modality (AID 7, SAP 8, MDI 9):
       - Overall R² with/without.
       - Stratified: per-modality R² full vs no_therapy.
  4. Time features (hour_sin 3, hour_cos 4):
       - Overall R² with/without.
       - Stratified by time-of-day period (Dawn, Morning, Afternoon, Evening/Night).

Run from /mnt/workspace/tvae:
  python -u scripts/variable_justification.py 2>&1 | tee results/variable_justification.txt
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stage2.data import (
    load_all_patients, make_imputation_dataset, make_eval_imputation_numpy,
    IDX_CGM, IDX_PI, IDX_RA, IDX_HSIN, IDX_HCOS, IDX_BOLUS, IDX_CARBS,
)
from src.stage2.evaluate import driver_response_test
from src.stage2.models import build_mtsm_imputation_model

OUT_DIR = 'results/variable_justification'
os.makedirs(OUT_DIR, exist_ok=True)

GAP_LABELS  = ['4h', '6h']
GAP_LENGTHS = [48, 72]          # steps at 5-min resolution

IDX_AID, IDX_SAP, IDX_MDI = 7, 8, 9


# ── Helpers ───────────────────────────────────────────────────────────────────

def r2_subset(true_z, pred_z, masks, idx):
    """R² over masked timesteps for a subset of windows (idx = boolean array)."""
    if idx.sum() == 0:
        return np.nan
    m  = masks[idx].astype(bool)
    t  = true_z[idx][m]
    p  = pred_z[idx][m]
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))


def predict(mtsm, W_in, zero_idxs):
    """Run imputation with given feature indices zeroed out."""
    import tensorflow as tf
    W = W_in.copy()
    for idx in zero_idxs:
        W[:, :, idx] = 0.0
    return mtsm.predict(W, batch_size=256, verbose=0)


def decode_hour(W, gap_center):
    """Recover hour of day (0–24) from sin/cos at gap centre."""
    s = W[:, gap_center, IDX_HSIN]
    c = W[:, gap_center, IDX_HCOS]
    return (np.arctan2(s, c) / (2 * np.pi) * 24) % 24


def print_row(label, full_r2, ablated_r2, n):
    delta = ablated_r2 - full_r2
    print(f'  {label:<28}  full={full_r2:+.4f}  ablated={ablated_r2:+.4f}  Δ={delta:+.4f}  n={n}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('\n=== Variable Justification Analysis ===\n')

    patients = load_all_patients('data/processed/adults_global_norm')
    splits   = make_imputation_dataset(patients, batch_size=128)

    print('Loading FM imputation model...')
    mtsm = build_mtsm_imputation_model()
    print(f'  Params: {mtsm.count_params():,}  (frozen)\n')

    for gap_len, gap_label in zip(GAP_LENGTHS, GAP_LABELS):
        print(f'{"="*60}')
        print(f' Gap: {gap_label}')
        print(f'{"="*60}\n')

        ev       = make_eval_imputation_numpy(splits['test_patients'], gap_len,
                                              max_windows=3000)
        W_orig   = ev['windows_orig']     # (N, 288, 10)
        W_masked = ev['windows_masked']   # CGM zeroed in gap
        masks    = ev['masks']            # (N, 288) 1=gap
        gs, ge   = ev['gap_start'], ev['gap_end']
        N        = len(W_orig)
        true_z   = W_orig[:, :, IDX_CGM]  # (N, 288)

        print(f'  Test windows: {N}')

        # ── Pre-compute all predictions ───────────────────────────────────────
        print('  Running predictions...')
        pred_full      = predict(mtsm, W_masked, [])
        pred_noflags   = predict(mtsm, W_masked, [IDX_BOLUS, IDX_CARBS])
        pred_notherapy = predict(mtsm, W_masked, [IDX_AID, IDX_SAP, IDX_MDI])
        pred_notime    = predict(mtsm, W_masked, [IDX_HSIN, IDX_HCOS])
        print()

        # ── 1. Binary flags ───────────────────────────────────────────────────
        print('--- 1. Binary flags (bolus + carbs logged) ---\n')

        has_bolus = W_orig[:, gs:ge, IDX_BOLUS].sum(axis=1) > 0
        has_carbs = W_orig[:, gs:ge, IDX_CARBS].sum(axis=1) > 0
        has_event = has_bolus | has_carbs
        no_event  = ~has_event

        print('  Bolus events in gap:')
        print_row('  with bolus in gap (full)', r2_subset(true_z, pred_full, masks, has_bolus),
                  r2_subset(true_z, pred_noflags, masks, has_bolus), has_bolus.sum())
        print_row('  no event in gap (full)',   r2_subset(true_z, pred_full, masks, no_event),
                  r2_subset(true_z, pred_noflags, masks, no_event),   no_event.sum())

        print()
        print('  Carb events in gap:')
        print_row('  with carbs in gap (full)', r2_subset(true_z, pred_full, masks, has_carbs),
                  r2_subset(true_z, pred_noflags, masks, has_carbs), has_carbs.sum())
        print_row('  any event in gap',         r2_subset(true_z, pred_full, masks, has_event),
                  r2_subset(true_z, pred_noflags, masks, has_event),  has_event.sum())

        # Driver direction accuracy: full vs no_flags
        print()
        print('  Driver direction accuracy (sign test on gap CGM change):')
        dr = driver_response_test(
            W_orig,
            {'full': pred_full, 'no_flags': pred_noflags},
            masks
        )
        for method, res in dr.items():
            print(f'    {method:<12}  bolus_acc={res["bolus_direction_acc"]:.3f} (n={res["n_bolus_windows"]})'
                  f'  carb_acc={res["carb_direction_acc"]:.3f} (n={res["n_carb_windows"]})')
        print()

        # ── 2. Therapy modality ───────────────────────────────────────────────
        print('--- 2. Therapy modality (AID / SAP / MDI one-hot) ---\n')

        for mod_name, mod_idx in [('AID', IDX_AID), ('SAP', IDX_SAP), ('MDI', IDX_MDI)]:
            mask_mod = W_orig[:, 0, mod_idx].astype(bool)
            r2_f = r2_subset(true_z, pred_full,      masks, mask_mod)
            r2_a = r2_subset(true_z, pred_notherapy, masks, mask_mod)
            print_row(f'  {mod_name} patients', r2_f, r2_a, mask_mod.sum())
        print()

        # ── 3. Time-of-day ────────────────────────────────────────────────────
        print('--- 3. Time features (hour_sin / hour_cos) ---\n')

        hour    = decode_hour(W_orig, (gs + ge) // 2)
        periods = {
            'Dawn (4–8h)':       (4  <= hour) & (hour < 8),
            'Morning (8–14h)':   (8  <= hour) & (hour < 14),
            'Afternoon (14–20h)':(14 <= hour) & (hour < 20),
            'Night (20–4h)':     (hour >= 20) | (hour < 4),
        }
        for period, idx in periods.items():
            r2_f = r2_subset(true_z, pred_full,   masks, idx)
            r2_a = r2_subset(true_z, pred_notime, masks, idx)
            print_row(f'  {period}', r2_f, r2_a, idx.sum())
        print()

        # ── Summary bar chart ─────────────────────────────────────────────────
        _save_flag_chart(gap_label, has_bolus, has_carbs, no_event,
                         true_z, pred_full, pred_noflags, masks, dr)

    print('\nDone. Results in', OUT_DIR)


def _save_flag_chart(gap_label, has_bolus, has_carbs, no_event,
                     true_z, pred_full, pred_noflags, masks, dr):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: stratified R²
    ax = axes[0]
    labels   = ['Bolus\nin gap', 'Carbs\nin gap', 'No event\nin gap']
    indices  = [has_bolus, has_carbs, no_event]
    full_r2s = [r2_subset(true_z, pred_full,    masks, idx) for idx in indices]
    noflag_r2= [r2_subset(true_z, pred_noflags, masks, idx) for idx in indices]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, full_r2s,   w, label='Full model',    color='steelblue', alpha=0.85)
    ax.bar(x + w/2, noflag_r2,  w, label='No flags',      color='tomato',    alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('R²')
    ax.set_title(f'Flags (bolus/carbs) — {gap_label} gap\nR² by event context')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    # Right: driver direction accuracy
    ax2 = axes[1]
    methods = list(dr.keys())
    bolus_acc = [dr[m]['bolus_direction_acc'] for m in methods]
    carb_acc  = [dr[m]['carb_direction_acc']  for m in methods]
    x2 = np.arange(len(methods))
    ax2.bar(x2 - w/2, bolus_acc, w, label='Bolus direction', color='steelblue', alpha=0.85)
    ax2.bar(x2 + w/2, carb_acc,  w, label='Carb direction',  color='seagreen',  alpha=0.85)
    ax2.axhline(0.5, color='grey', ls='--', lw=0.8, label='Random baseline')
    ax2.set_xticks(x2)
    ax2.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=9)
    ax2.set_ylabel('Direction accuracy')
    ax2.set_title(f'Driver response accuracy — {gap_label} gap')
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT_DIR, f'flags_{gap_label}.{ext}'), dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
