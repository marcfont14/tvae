"""Generate Clarke Error Grid plots at t+30 min using saved gn_run01 weights."""
import gc, os, sys, numpy as np
sys.path.insert(0, '/mnt/workspace/tvae')
os.chdir('/mnt/workspace/tvae')

import tensorflow as tf
from src.encoder import build_encoder, build_decoder
from src.stage2.data import make_eval_dataset, load_all_patients, make_forecasting_dataset
from src.stage2.models import build_forecasting_lstm, build_forecasting_lstm_decoder, build_raw_forecasting_lstm
from src.stage2.evaluate import plot_clarke_grid

RUN     = 'results/stage2/forecasting/gn_run01'
DATA    = 'data/processed/adults_global_norm'
T30_IDX = 5   # index 5 → t+30 min

def _build_bare(tag_short):
    """Build model structure without loading any Stage 1 weights."""
    if tag_short == 'fm_decoder_ft':
        dec = build_decoder()
        dec(tf.zeros((1, 288, 10)))   # build variables
        dec.trainable = True
        return build_forecasting_lstm_decoder(dec)
    elif tag_short == 'fm_ft':
        enc = build_encoder()
        enc(tf.zeros((1, 288, 10)))
        enc.trainable = True
        return build_forecasting_lstm(enc)
    else:  # raw
        return build_raw_forecasting_lstm()

VARIANTS = ['raw']

patients = load_all_patients(DATA)
splits   = make_forecasting_dataset(patients, batch_size=128)

for tag_short in VARIANTS:
    tag     = f'{tag_short}_lstm'
    weights = os.path.join(RUN, f'weights_{tag}.weights.h5')
    out     = os.path.join(RUN, f'clarke_t30_{tag}.png')

    if not os.path.exists(weights):
        print(f'No weights for {tag}, skipping')
        continue

    gc.collect()
    model = _build_bare(tag_short)

    # warm-up to ensure all sublayer variables are created
    dummy_x    = np.zeros((2, 288, 10), dtype=np.float32)
    dummy_last = np.zeros((2, 1),       dtype=np.float32)
    try:
        model([dummy_x, dummy_last], training=False)
    except Exception:
        model(dummy_x, training=False)

    model.compile(optimizer='adam', loss='huber')
    model.load_weights(weights)
    print(f'\nLoaded {tag}')

    ys, ps = [], []
    for xb, yb in make_eval_dataset(splits['test_patients'], 128):
        ys.append(yb.numpy())
        ps.append(model.predict_on_batch(xb))
    y_test = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(ps, axis=0)

    plot_clarke_grid(y_test[:, T30_IDX], y_pred[:, T30_IDX], out, horizon_min=30)
    print(f'Saved {out}')

    del model; gc.collect()

print('\nDone.')
