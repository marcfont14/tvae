"""
analyse_H.py
============
H representation analysis for the Stage 1 MTSM encoder.

Produces (in results/mtsm/{run_id}/):
  attention_{post_bolus,meal,hypo,stable}.png   Signal + L5 attention heatmap per clinical scenario
  attention_event_aligned.png                   Average attention row at event timestep (±8h)
  pca_by_modality.png                           L5 H PCA coloured by AID/SAP/MDI
  feature_corr_per_layer.png                    Pearson r(||H_t||, feature) heatmap across layers
  H_norm_vs_drivers.png                         Event-triggered H norm at L5
  H_norm_circadian.png                          Mean H norm by hour of day at L5
  abstraction_trajectory.png                    PC1% + probe R² across all 5 layers
  H_enrichment_scores.json                      PC1_L5, Σ|r_L5|, abstraction depth, sign flip

Usage:
  python scripts/analyse_H.py --run_id encoder2 --no_age
  python scripts/analyse_H.py --run_id encoder3 --no_age --encoder3
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.encoder import load_encoder, load_encoder3

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ── Constants ─────────────────────────────────────────────────────────────────

WINDOW_LEN  = 288
PREFIX_LEN  = 144   # encoder3: steps 0–143 bidirectional, 144–287 causal
N_FEATURES  = 11
CGM_IDX     = 0
PI_IDX      = 1
RA_IDX      = 2
HOUR_SIN    = 3
HOUR_COS    = 4
BOLUS_IDX   = 5
CARBS_IDX   = 6
AID_IDX     = 7
SAP_IDX     = 8
MDI_IDX     = 9

D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 5
D_FF     = 256
DROPOUT  = 0.2
SEED     = 42
TEST_SPLIT = 0.1
VAL_SPLIT  = 0.1
RESULTS_BASE = 'results/mtsm'

FEATURE_NAMES = ['CGM', 'PI', 'RA', 'hour_sin', 'hour_cos', 'bolus', 'carbs']
EVENT_WINDOW  = 60   # steps either side of event for triggered analysis

np.random.seed(SEED)
tf.random.set_seed(SEED)

COLORS = {
    'cgm':   '#111827',
    'pi':    '#7C3AED',
    'ra':    '#059669',
    'bolus': '#DC2626',
    'carbs': '#D97706',
    'norm':  '#2563EB',
    'aid':   '#2563EB',
    'sap':   '#059669',
    'mdi':   '#DC2626',
}


# ── Architecture ──────────────────────────────────────────────────────────────

def get_positional_encoding(seq_len, d_model):
    positions = np.arange(seq_len)[:, np.newaxis]
    dims      = np.arange(d_model)[np.newaxis, :]
    angles    = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)


def build_transformer_encoder(window_len=WINDOW_LEN, n_features=10,
                               d_model=D_MODEL, n_heads=N_HEADS,
                               n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT):
    inp = keras.Input(shape=(window_len, n_features), name='input')
    x   = layers.Dense(d_model, name='input_proj')(inp)
    x   = x + get_positional_encoding(window_len, d_model)
    for i in range(n_layers):
        attn = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads,
            dropout=dropout, name=f'mhsa_{i}'
        )(x, x)
        attn = layers.Dropout(dropout)(attn)
        x    = layers.LayerNormalization(epsilon=1e-6, name=f'norm1_{i}')(x + attn)
        ffn  = layers.Dense(d_ff, activation='relu', name=f'ffn1_{i}')(x)
        ffn  = layers.Dropout(dropout)(ffn)
        ffn  = layers.Dense(d_model, name=f'ffn2_{i}')(ffn)
        ffn  = layers.Dropout(dropout)(ffn)
        x    = layers.LayerNormalization(epsilon=1e-6, name=f'norm2_{i}')(x + ffn)
    return keras.Model(inp, x, name='TransformerEncoder')


def _layer_output_tensor(encoder, layer_idx, encoder3):
    """Return the output tensor of block `layer_idx` for either architecture."""
    if encoder3:
        return encoder.get_layer(f'prefix_lm_{layer_idx}').output
    else:
        return encoder.get_layer(f'norm2_{layer_idx}').output


def _strip_cls(H, encoder3):
    """Drop CLS token (position 0) for encoder2; encoder3 has none."""
    return H if encoder3 else H[:, 1:, :]


def _get_attention_single(encoder, window_np, layer_idx, d_model=D_MODEL, encoder3=False):
    """Compute attention weights for one window at a given layer.

    Returns (n_heads, 288, 288). For encoder3 the prefix-LM mask is applied.
    """
    batch = tf.cast(window_np[np.newaxis], tf.float32)
    if encoder3:
        block   = encoder.get_layer(f'prefix_lm_{layer_idx}')
        q_model = keras.Model(encoder.input, block.input)
        q       = tf.cast(q_model(batch, training=False), tf.float32)  # (1, 288, d)
        _, attn = block.mhsa(q, q, attention_mask=block._mask,
                             return_attention_scores=True, training=False)
        return attn.numpy()[0]   # (n_heads, 288, 288) — no CLS to drop
    else:
        mha_layer = encoder.get_layer(f'mhsa_{layer_idx}')
        if layer_idx == 0:
            cls_model = keras.Model(encoder.input, encoder.get_layer('cls_token').output)
            q = tf.cast(cls_model(batch, training=False), tf.float32)  # (1, 289, d)
        else:
            prev_model = keras.Model(encoder.input,
                                     encoder.get_layer(f'norm2_{layer_idx-1}').output)
            q = tf.cast(prev_model(batch, training=False), tf.float32)  # (1, 289, d)
        _, attn = mha_layer(q, q, return_attention_scores=True, training=False)
        return attn.numpy()[0, :, 1:, 1:]   # (n_heads, 288, 288) — drop CLS row/col


# ── Data loading ──────────────────────────────────────────────────────────────

def index_test_patients(data_dir, max_patients=None):
    """Reconstruct the same test split used during training (SEED=42)."""
    npz_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.npz'))
    if max_patients:
        npz_files = npz_files[:max_patients]

    # Quality filter (same as experiment_mtsm)
    valid = []
    for fname in npz_files:
        fpath = os.path.join(data_dir, fname)
        try:
            d     = np.load(fpath, allow_pickle=True)
            wins  = d['windows'].astype(np.float32)
        except Exception:
            print(f"  [WARN] skipping corrupt file: {fname}")
            continue
        bolus = wins[:, :, BOLUS_IDX]
        carbs = wins[:, :, CARBS_IDX]
        cgm   = wins[:, :, CGM_IDX]
        keep  = ((bolus + carbs) > 0).any(axis=1) & (cgm.std(axis=1) > 0.3) & (cgm.std(axis=1) < 4.0)
        for i in np.where(keep)[0]:
            valid.append((fpath, int(i)))

    all_fpaths = sorted(set(fp for fp, _ in valid))
    rng        = np.random.default_rng(SEED)
    perm       = rng.permutation(len(all_fpaths))
    n_test     = int(len(all_fpaths) * TEST_SPLIT)
    test_set   = set(all_fpaths[i] for i in perm[:n_test])
    return [(fp, wi) for fp, wi in valid if fp in test_set]


def sample_windows(test_records, n_windows=1500, no_age=True):
    """Load up to n_windows from test records. Returns (windows, modalities, ages)."""
    if len(test_records) > n_windows:
        idx     = np.random.choice(len(test_records), n_windows, replace=False)
        records = [test_records[i] for i in idx]
    else:
        records = test_records

    windows, modalities, ages = [], [], []
    for fpath, win_idx in records:
        try:
            d   = np.load(fpath, allow_pickle=True)
            win = d['windows'][win_idx].astype(np.float32)
        except Exception:
            continue
        windows.append(win)
        # Modality from one-hot (features 7-9)
        onehot = win[0, AID_IDX:MDI_IDX + 1]
        mod_idx = int(np.argmax(onehot)) if onehot.sum() > 0 else 0
        modalities.append(['AID', 'SAP', 'MDI'][mod_idx])
        # Age from feature 10 if present
        ages.append(float(win[0, 10]) if win.shape[-1] > 10 else float('nan'))

    windows = np.stack(windows, axis=0)
    if no_age:
        windows = windows[:, :, :10]
    return windows, np.array(modalities), np.array(ages)


# ── H computation ─────────────────────────────────────────────────────────────

def compute_H_per_layer(encoder, windows, n_layers=N_LAYERS, batch_size=64, encoder3=False):
    """Returns list of H arrays, one per layer, shape (N, 288, d_model)."""
    H_layers = [[] for _ in range(n_layers)]
    for i in range(n_layers):
        layer_model = keras.Model(encoder.input, _layer_output_tensor(encoder, i, encoder3))
        for start in range(0, len(windows), batch_size):
            batch = tf.cast(windows[start:start + batch_size], tf.float32)
            H = layer_model(batch, training=False).numpy()
            H_layers[i].append(_strip_cls(H, encoder3))
    return [np.concatenate(h, axis=0) for h in H_layers]


# ── Plot: attention case studies ──────────────────────────────────────────────

def _find_event_windows(windows, n_per_type=4):
    """Select representative window indices per clinical category."""
    bolus_flag = windows[:, :, BOLUS_IDX] > 0
    carbs_flag = windows[:, :, CARBS_IDX] > 0
    cgm        = windows[:, :, CGM_IDX]
    n_bolus    = bolus_flag.sum(axis=1)
    n_carbs    = carbs_flag.sum(axis=1)

    mid_bolus = bolus_flag[:, 72:216].sum(axis=1)
    mid_carbs = carbs_flag[:, 72:216].sum(axis=1)

    # Post-bolus: 1-3 boluses in middle third, no carbs logged
    post_bolus = (mid_bolus >= 1) & (mid_bolus <= 3) & (n_carbs == 0)
    # Meal: carbs + bolus together in middle third
    meal       = (mid_carbs >= 1) & (mid_bolus >= 1)
    # Hypo: CGM dips below −1.5 z-score
    hypo       = cgm.min(axis=1) < -1.5
    # Stable: no events, flat CGM
    stable     = (n_bolus == 0) & (n_carbs == 0) & (cgm.std(axis=1) < 0.4)

    selected = {}
    labels   = {'post_bolus': post_bolus, 'meal': meal, 'hypo': hypo, 'stable': stable}
    for name, mask in labels.items():
        idxs = np.where(mask)[0]
        if len(idxs) > 0:
            selected[name] = idxs[:n_per_type]
    return selected


def _add_prefix_boundary(ax, step, alpha=0.55):
    """Draw the encoder3 prefix-LM boundary at step 144 on an attention heatmap."""
    bd = PREFIX_LEN // step
    ax.axvline(bd - 0.5, color='gold', lw=1.2, alpha=alpha, ls='-')
    ax.axhline(bd - 0.5, color='gold', lw=1.2, alpha=alpha, ls='-')


def plot_attention_case_studies(encoder, windows, results_dir, n_cases=3, encoder3=False):
    """
    For representative windows (post-bolus, meal, hypo, stable): plot the raw
    CGM/PI/RA signal alongside the L5 mean-head attention heatmap so that the
    model's contextual focus is interpretable in clinical terms.
    Outputs one file per category: attention_post_bolus.png, etc.
    """
    print("  [1/8] Attention case studies...")
    selected = _find_event_windows(windows, n_per_type=n_cases)
    if not selected:
        print("    [WARN] No event windows found — skipping")
        return

    step   = 6                                         # downsample: every 30 min → 48 ticks
    t_full = np.arange(WINDOW_LEN) * 5 / 60           # hours (full resolution)
    t_down = t_full[::step]
    n_down = len(t_down)
    tick_pos    = np.arange(0, n_down, max(1, n_down // 8))
    tick_labels = [f'{t_down[i]:.0f}h' for i in tick_pos]

    cat_titles = {
        'post_bolus': 'Post-Bolus',
        'meal':       'Meal (Bolus + Carbs)',
        'hypo':       'Hypoglycaemia',
        'stable':     'Stable — No Events',
    }

    for cat, idxs in selected.items():
        n   = len(idxs)
        fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n), squeeze=False)
        fig.suptitle(f"L5 Attention — {cat_titles.get(cat, cat)}",
                     fontsize=13, fontweight='bold')

        for row, idx in enumerate(idxs):
            win = windows[idx]   # (288, 10)

            # ── Signal ────────────────────────────────────────────────────────
            ax_s = axes[row, 0]
            ax_s.plot(t_full, win[:, CGM_IDX], color=COLORS['cgm'],  lw=1.5, label='CGM (z)')
            ax_s.plot(t_full, win[:, PI_IDX],  color=COLORS['pi'],   lw=1.2, label='PI (z)',  alpha=0.8)
            ax_s.plot(t_full, win[:, RA_IDX],  color=COLORS['ra'],   lw=1.2, label='RA (z)',  alpha=0.8)
            for bt in np.where(win[:, BOLUS_IDX] > 0)[0]:
                ax_s.axvline(t_full[bt], color=COLORS['bolus'], lw=0.9, alpha=0.7, ls='--')
            for ct in np.where(win[:, CARBS_IDX] > 0)[0]:
                ax_s.axvline(t_full[ct], color=COLORS['carbs'], lw=0.9, alpha=0.7, ls=':')
            if encoder3:
                ax_s.axvline(PREFIX_LEN * 5 / 60, color='gold', lw=1.0, alpha=0.6,
                             ls='-', label='prefix boundary')
            ax_s.set_xlabel('Time (h)', fontsize=9)
            ax_s.set_ylabel('z-score', fontsize=9)
            ax_s.legend(fontsize=8, loc='upper right')
            ax_s.grid(True, alpha=0.3)
            ax_s.spines[['top', 'right']].set_visible(False)

            # ── Attention heatmap ──────────────────────────────────────────────
            ax_a = axes[row, 1]
            try:
                attn      = _get_attention_single(encoder, win, layer_idx=N_LAYERS - 1,
                                                   encoder3=encoder3)
                mean_attn = attn.mean(axis=0)           # (T, T): avg over heads
                attn_down = mean_attn[::step, ::step]   # (n_down, n_down)

                im = ax_a.imshow(attn_down, aspect='auto', cmap='Blues',
                                 interpolation='nearest', origin='upper')
                plt.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)

                for bt in np.where(win[:, BOLUS_IDX] > 0)[0]:
                    bd = bt // step
                    ax_a.axvline(bd, color=COLORS['bolus'], lw=0.9, alpha=0.8, ls='--')
                    ax_a.axhline(bd, color=COLORS['bolus'], lw=0.9, alpha=0.8, ls='--')
                for ct in np.where(win[:, CARBS_IDX] > 0)[0]:
                    cd = ct // step
                    ax_a.axvline(cd, color=COLORS['carbs'], lw=0.9, alpha=0.6, ls=':')
                    ax_a.axhline(cd, color=COLORS['carbs'], lw=0.9, alpha=0.6, ls=':')

                if encoder3:
                    _add_prefix_boundary(ax_a, step)

                ax_a.set_xticks(tick_pos); ax_a.set_xticklabels(tick_labels, fontsize=8)
                ax_a.set_yticks(tick_pos); ax_a.set_yticklabels(tick_labels, fontsize=8)
                ax_a.set_xlabel('Key (time)', fontsize=9)
                ax_a.set_ylabel('Query (time)', fontsize=9)
                title = 'Mean-head attention (L5)'
                if encoder3:
                    title += '  [gold = prefix boundary @ 12h]'
                ax_a.set_title(title, fontsize=9)
            except Exception as e:
                ax_a.set_title(f'Error: {e}', fontsize=8)

        plt.tight_layout()
        path = os.path.join(results_dir, f'attention_{cat}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {path}")


def plot_attention_event_aligned(encoder, windows, results_dir, encoder3=False):
    """
    For windows with a bolus or meal event at a known timestep, extract the
    attention row at that timestep and express it relative to the event (±8h).
    Averaging across windows reveals what the model contextualises around events.
    """
    print("  [2/8] Event-aligned attention profiles...")
    EVENT_HALF = 96   # ±96 steps = ±8h
    step = 4          # every 20 min for display

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Event-Aligned Attention at L5  "
                 "(average attention row at event timestep, expressed relative to event)",
                 fontsize=12, fontweight='bold')

    t_rel = np.arange(-EVENT_HALF, EVENT_HALF) * 5 / 60   # hours
    t_rel_down = t_rel[::step]

    for ax, ev_idx, label, color in [
        (axes[0], BOLUS_IDX, 'Bolus',        COLORS['bolus']),
        (axes[1], CARBS_IDX, 'Meal (Carbs)', COLORS['carbs']),
    ]:
        profiles = []
        for i in range(len(windows)):
            ev_times = np.where(windows[i, :, ev_idx] > 0)[0]
            # Only use events in the safe middle range so we can always extract ±8h
            ev_times = ev_times[(ev_times >= EVENT_HALF) & (ev_times < WINDOW_LEN - EVENT_HALF)]
            if len(ev_times) == 0:
                continue
            t_ev = ev_times[len(ev_times) // 2]   # pick the middle event
            try:
                attn = _get_attention_single(encoder, windows[i], layer_idx=N_LAYERS - 1,
                                             encoder3=encoder3)
                row  = attn.mean(axis=0)[t_ev]                          # (T,)
                row  = row[t_ev - EVENT_HALF: t_ev + EVENT_HALF]        # (2*EVENT_HALF,)
                profiles.append(row[::step])
            except Exception:
                continue

        if profiles:
            profiles = np.array(profiles)
            mean_p   = profiles.mean(axis=0)
            mean_p  /= mean_p.sum() + 1e-8   # normalise so area = 1

            ax.plot(t_rel_down, mean_p, color=color, lw=2)
            ax.fill_between(t_rel_down, 0, mean_p, alpha=0.25, color=color)
            ax.axvline(0, color='black', ls='--', lw=1.2, alpha=0.7, label='Event (t=0)')
            ax.set_xlabel('Time relative to event (h)', fontsize=10)
            ax.set_ylabel('Normalised attention weight', fontsize=10)
            ax.set_title(f'{label}  (n={len(profiles)} windows)', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.spines[['top', 'right']].set_visible(False)
        else:
            ax.set_title(f'{label} — no valid events found')

    plt.tight_layout()
    path = os.path.join(results_dir, 'attention_event_aligned.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


# ── Plot: PCA by modality (L5) ────────────────────────────────────────────────

def plot_pca_by_modality(H_L5, modalities, results_dir):
    print("  [3/8] PCA by modality (L5)...")
    from sklearn.decomposition import PCA

    h_pool = H_L5.mean(axis=1)
    pca    = PCA(n_components=2)
    coords = pca.fit_transform(h_pool)
    pc1    = pca.explained_variance_ratio_[0] * 100
    pc2    = pca.explained_variance_ratio_[1] * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    for mod, color in [('AID', COLORS['aid']), ('SAP', COLORS['sap']), ('MDI', COLORS['mdi'])]:
        mask = modalities == mod
        if mask.sum() > 0:
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       alpha=0.4, s=8, color=color, label=f'{mod} (n={mask.sum()})')
    ax.set_xlabel(f'PC1 ({pc1:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pc2:.1f}%)', fontsize=11)
    ax.set_title('PCA of Mean-Pooled H (L5) by Therapy Modality', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(results_dir, 'pca_by_modality.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


# ── Plot: feature correlation per layer ───────────────────────────────────────

def plot_feature_corr_per_layer(H_layers, windows, n_layers, results_dir):
    print("  [5/8] Feature correlation heatmap per layer...")
    feat_indices = [CGM_IDX, PI_IDX, HOUR_SIN, HOUR_COS, BOLUS_IDX, CARBS_IDX]
    feat_labels  = ['CGM', 'PI', 'hour_sin', 'hour_cos', 'bolus', 'carbs']
    corr_matrix  = np.zeros((n_layers, len(feat_indices)))

    for li, H in enumerate(H_layers):
        norms = np.linalg.norm(H, axis=-1).reshape(-1)   # (N*288,)
        for fi, feat_idx in enumerate(feat_indices):
            feat = windows[:, :, feat_idx].reshape(-1)
            # Exclude flat features
            if feat.std() < 1e-6:
                corr_matrix[li, fi] = float('nan')
                continue
            r, _ = pearsonr(norms, feat)
            corr_matrix[li, fi] = r

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(corr_matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Pearson r')
    ax.set_xticks(range(len(feat_labels))); ax.set_xticklabels(feat_labels, fontsize=10)
    ax.set_yticks(range(n_layers)); ax.set_yticklabels([f'L{i+1}' for i in range(n_layers)], fontsize=10)
    ax.set_title('Pooled Pearson r(||H_t||₂, feature) per Layer', fontsize=12)
    for li in range(n_layers):
        for fi in range(len(feat_indices)):
            v = corr_matrix[li, fi]
            if not np.isnan(v):
                ax.text(fi, li, f'{v:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if abs(v) > 0.5 else 'black')
    plt.tight_layout()
    path = os.path.join(results_dir, 'feature_corr_per_layer.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    L5 CGM r = {corr_matrix[-1, 0]:.3f}")
    print(f"    Saved: {path}")
    return corr_matrix


# ── Plot: event-triggered H norm ─────────────────────────────────────────────

def _event_triggered_norm(norms, event_mask, win=EVENT_WINDOW):
    """Average H norm trace ±win steps around each event."""
    traces = []
    for i in range(len(norms)):
        for t in np.where(event_mask[i])[0]:
            if t - win >= 0 and t + win + 1 <= WINDOW_LEN:
                traces.append(norms[i, t - win:t + win + 1])
    if not traces:
        return None, 0
    arr = np.array(traces)
    return arr.mean(axis=0), len(traces)


def plot_H_norm_vs_drivers(H_L5, windows, results_dir):
    """H_L5: (N, 288, d_model) — precomputed from compute_H_per_layer (CLS already stripped)."""
    print("  [7/8] Event-triggered H norm at L5...")
    norms = np.linalg.norm(H_L5, axis=-1)

    bolus_mask = windows[:, :, BOLUS_IDX] > 0
    carbs_mask = windows[:, :, CARBS_IDX] > 0

    t_ax = np.arange(-EVENT_WINDOW, EVENT_WINDOW + 1) * 5

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Event-Triggered H Norm at L5 (Final Layer)', fontsize=13, fontweight='bold')

    for ax, mask, label, color in [
        (axes[0], bolus_mask, 'Bolus', COLORS['bolus']),
        (axes[1], carbs_mask, 'Carbs', COLORS['carbs']),
    ]:
        trace, n_events = _event_triggered_norm(norms, mask)
        if trace is not None:
            ax.plot(t_ax, trace, color=color, lw=2)
            ax.axvline(0, color='black', ls='--', lw=1, alpha=0.6, label='Event t=0')
            ax.set_title(f'{label}-triggered  (n={n_events:,} events)', fontsize=11)
        ax.set_xlabel('Time from event (min)', fontsize=10)
        ax.set_ylabel('Mean ||H_t||₂', fontsize=10)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(results_dir, 'H_norm_vs_drivers.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


# ── Plot: circadian H norm ─────────────────────────────────────────────────────

def plot_H_circadian(H_norms, windows, results_dir):
    """Mean H norm by hour of day. H_norms: (N, 288)."""
    print("  [8/8] Circadian H norm pattern...")
    hour_sin = windows[:, :, HOUR_SIN]   # (N, 288)
    hour_cos = windows[:, :, HOUR_COS]
    hours    = (np.arctan2(hour_sin, hour_cos) * 12 / np.pi) % 24

    norms_flat = H_norms.reshape(-1)
    hours_flat = hours.reshape(-1)

    mean_by_hour, std_by_hour = [], []
    hour_bins = np.arange(0, 24, 0.5)
    for h in hour_bins:
        mask = (hours_flat >= h) & (hours_flat < h + 0.5)
        if mask.sum() > 0:
            mean_by_hour.append(norms_flat[mask].mean())
            std_by_hour.append(norms_flat[mask].std())
        else:
            mean_by_hour.append(float('nan'))
            std_by_hour.append(0)
    mean_by_hour = np.array(mean_by_hour)
    std_by_hour  = np.array(std_by_hour)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hour_bins, mean_by_hour, color=COLORS['norm'], lw=2)
    ax.fill_between(hour_bins,
                    mean_by_hour - std_by_hour * 0.3,
                    mean_by_hour + std_by_hour * 0.3,
                    alpha=0.2, color=COLORS['norm'])
    ax.set_xlabel('Hour of day', fontsize=11)
    ax.set_ylabel('Mean ||H_t||₂ (L5)', fontsize=11)
    ax.set_title('Circadian Pattern of H Norm (L5)', fontsize=12)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 3)], fontsize=9)
    ax.grid(True, alpha=0.3); ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(results_dir, 'H_norm_circadian.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


# ── Plot: abstraction trajectory ──────────────────────────────────────────────

def plot_abstraction_trajectory(H_layers, windows, n_layers, results_dir):
    """H_layers: list of (N, 288, d_model) arrays — precomputed from compute_H_per_layer."""
    from sklearn.decomposition import PCA
    print("  Abstraction trajectory (PC1 + probe R² across layers)...")

    pc1s, r2s = [], []
    for i, H in enumerate(H_layers):
        h_pool = H.mean(axis=1)
        pc1s.append(PCA(n_components=1).fit(h_pool).explained_variance_ratio_[0] * 100)

        H_flat = H.reshape(-1, H.shape[-1])
        cgm    = windows[:, :, CGM_IDX].reshape(-1)
        idx    = np.random.choice(len(H_flat), min(30_000, len(H_flat)), replace=False)
        probe  = Ridge(alpha=1.0).fit(H_flat[idx], cgm[idx])
        pred   = probe.predict(H_flat[idx])
        ss_res = np.sum((cgm[idx] - pred) ** 2)
        ss_tot = np.sum((cgm[idx] - cgm[idx].mean()) ** 2)
        r2s.append(float(1 - ss_res / ss_tot))

    x = np.arange(1, n_layers + 1)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    ax1.plot(x, pc1s, 'o-', color=COLORS['norm'], lw=2, label='PC1 %')
    ax2.plot(x, r2s,  's--', color=COLORS['bolus'], lw=2, label='Probe R²')
    ax1.set_xlabel('Layer', fontsize=11)
    ax1.set_ylabel('PC1 (%)', fontsize=11, color=COLORS['norm'])
    ax2.set_ylabel('Probe R²', fontsize=11, color=COLORS['bolus'])
    ax1.set_title('Abstraction Trajectory Across Layers', fontsize=12)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=9)
    ax1.grid(True, alpha=0.3); ax1.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(results_dir, 'abstraction_trajectory.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    PC1 per layer: {[f'{v:.1f}%' for v in pc1s]}")
    print(f"    R²  per layer: {[f'{v:.3f}' for v in r2s]}")
    print(f"    Saved: {path}")
    return pc1s, r2s


# ── Save enrichment scores ────────────────────────────────────────────────────

def save_enrichment_scores(pc1_vals, r2s, corr_matrix, results_dir):
    pc1_L5   = pc1_vals[-1]
    r2_L5    = r2s[-1]
    cgm_r_L5 = float(corr_matrix[-1, 0]) if not np.isnan(corr_matrix[-1, 0]) else None
    pi_r_L5  = float(corr_matrix[-1, 1]) if not np.isnan(corr_matrix[-1, 1]) else None
    hs_r_L5  = float(corr_matrix[-1, 2]) if not np.isnan(corr_matrix[-1, 2]) else None

    # H richness score components (from H_analysis.md §13.1)
    distributed_variance = 100 - pc1_L5
    feature_coverage     = sum(abs(v) for v in [cgm_r_L5 or 0, pi_r_L5 or 0, hs_r_L5 or 0])
    abstraction_depth    = 1 - r2_L5
    sign_flip            = cgm_r_L5 is not None and cgm_r_L5 < 0

    scores = {
        'PC1_per_layer':          [round(v, 2) for v in pc1_vals],
        'probe_R2_per_layer':     [round(v, 4) for v in r2s],
        'L5': {
            'PC1':                round(pc1_L5, 2),
            'probe_R2':           round(r2_L5, 4),
            'CGM_r':              round(cgm_r_L5, 4) if cgm_r_L5 is not None else None,
            'PI_r':               round(pi_r_L5, 4)  if pi_r_L5  is not None else None,
            'hour_sin_r':         round(hs_r_L5, 4)  if hs_r_L5  is not None else None,
            'sign_flip':          sign_flip,
        },
        'H_richness': {
            'distributed_variance': round(distributed_variance, 2),
            'feature_coverage':     round(feature_coverage, 4),
            'abstraction_depth':    round(abstraction_depth, 4),
        },
    }

    path = os.path.join(results_dir, 'H_enrichment_scores.json')
    with open(path, 'w') as f:
        json.dump(scores, f, indent=2)

    print(f"\n{'='*52}")
    print(f"  H ENRICHMENT SCORES")
    print(f"{'='*52}")
    print(f"  PC1 per layer:   {[f'{v:.1f}%' for v in pc1_vals]}")
    print(f"  Probe R² L5:     {r2_L5:.4f}   (abstraction depth: {abstraction_depth:.4f})")
    print(f"  L5 CGM r:        {cgm_r_L5:.4f}  {'✓ sign flip' if sign_flip else '✗ no sign flip'}")
    print(f"  L5 PI r:         {pi_r_L5:.4f}")
    print(f"  Distributed var: {distributed_variance:.1f}%  (100 − PC1_L5)")
    print(f"  Feature coverage:{feature_coverage:.4f}  (Σ|r| CGM+PI+hour_sin)")
    print(f"  Saved: {path}")
    print(f"{'='*52}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    results_dir = os.path.join(RESULTS_BASE, args.run_id)
    enc_path    = os.path.join(results_dir, 'encoder_weights.weights.h5')
    assert os.path.exists(enc_path), f"Encoder weights not found: {enc_path}"

    n_features = 10 if args.no_age else N_FEATURES

    arch_tag = 'encoder3 (PrefixLM)' if args.encoder3 else 'encoder2 (CLS)'
    print(f"\n── analyse_H.py  run={args.run_id}  arch={arch_tag} ──────────────────")

    if args.encoder3:
        encoder = load_encoder3(weights_path=enc_path, trainable=False,
                                n_features=n_features)
    else:
        encoder = load_encoder(weights_path=enc_path, trainable=False,
                               n_features=n_features)
    print(f"  Encoder loaded: {enc_path}")

    # Load test windows
    print(f"  Indexing test split from: {args.data}")
    test_records = index_test_patients(args.data)
    print(f"  Test records: {len(test_records):,} windows")

    windows, modalities, ages = sample_windows(test_records, args.n_windows, no_age=args.no_age)
    print(f"  Sampled {len(windows)} windows  shape={windows.shape}")

    # Compute H at all layers
    print("\n  Computing H per layer...")
    H_layers = compute_H_per_layer(encoder, windows, encoder3=args.encoder3)

    # Plots
    plot_attention_case_studies(encoder, windows, results_dir,
                                encoder3=args.encoder3)              # attention_{cat}.png (×4)
    plot_attention_event_aligned(encoder, windows, results_dir,
                                 encoder3=args.encoder3)             # attention_event_aligned.png
    plot_pca_by_modality(H_layers[-1], modalities, results_dir)      # pca_by_modality.png
    corr_matrix = plot_feature_corr_per_layer(H_layers, windows, N_LAYERS, results_dir)
    plot_H_norm_vs_drivers(H_layers[-1], windows, results_dir)        # H_norm_vs_drivers.png
    H_norms_L5 = np.linalg.norm(H_layers[-1], axis=-1)
    plot_H_circadian(H_norms_L5, windows, results_dir)               # H_norm_circadian.png
    pc1_vals, r2s = plot_abstraction_trajectory(H_layers, windows, N_LAYERS, results_dir)

    save_enrichment_scores(pc1_vals, r2s, corr_matrix, results_dir)

    print(f"\n  Done. All plots in: {results_dir}/\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='H representation analysis for Stage 1 encoder')
    parser.add_argument('--run_id',    type=str, required=True)
    parser.add_argument('--data',      type=str, default='data/processed/adults')
    parser.add_argument('--no_age',    action='store_true', default=False)
    parser.add_argument('--n_windows', type=int, default=1500)
    parser.add_argument('--encoder3',  action='store_true', default=False,
                        help='Use encoder3 (PrefixLM, no CLS) instead of encoder2')
    args = parser.parse_args()
    main(args)
