"""Regenerate clean training curves for encoder_global_norm and decoder_global_norm."""
import re, os
os.chdir('/mnt/workspace/tvae')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Parse encoder log ────────────────────────────────────────────────────────
with open('results/mtsm/encoder_global_norm_log.txt', 'r') as f:
    enc_content = f.read()

pattern_enc = (r'4734/4734 \[=+\].*?'
               r'loss: ([\d.]+) - masked_acc: ([\d.]+) - '
               r'val_loss: ([\d.]+) - val_masked_acc: ([\d.]+)')
enc_rows = re.findall(pattern_enc, enc_content)
enc_train_loss = [float(r[0]) for r in enc_rows]
enc_val_loss   = [float(r[2]) for r in enc_rows]
enc_val_acc    = [float(r[3]) for r in enc_rows]
enc_epochs     = list(range(1, len(enc_rows) + 1))

# ── Parse decoder log ────────────────────────────────────────────────────────
with open('results/mtsm/decoder_global_norm_log.txt', 'r') as f:
    dec_lines = f.readlines()

dec_train_loss, dec_val_loss, dec_val_epochs = [], [], []
current_epoch = 0
for i, line in enumerate(dec_lines):
    if re.match(r'^Epoch \d+/70$', line.strip()):
        current_epoch += 1
    if '4734/4734' in line and 'val_loss:' in line and 'ntp_acc:' in line:
        m = re.search(r' loss: ([\d.]+).*val_loss: ([\d.]+)', line)
        if m:
            dec_train_loss_ep = float(m.group(1))
            dec_val = float(m.group(2))
            # Override current train_loss for this epoch
    if '4734/4734' in line and 'ntp_acc:' in line and 'val_loss' not in line:
        m = re.search(r' - loss: ([\d.]+) - ntp_acc:', line)
        if m:
            dec_train_loss.append(float(m.group(1)))

# Better approach: parse per-epoch systematically
dec_train_loss, dec_val_loss, dec_val_epochs = [], [], []
all_dec = dec_lines
ep_idx = 0
i = 0
while i < len(all_dec):
    line = all_dec[i].strip()
    if re.match(r'^Epoch \d+/70$', line):
        ep_idx += 1
        # Scan forward for completed epoch line
        for j in range(i+1, min(i+6000, len(all_dec))):
            l = all_dec[j]
            if '4734/4734' in l and 'loss:' in l:
                # Clean backspaces from line
                clean = re.sub(r'\x08+', '', l)
                m_train = re.search(r' - loss: ([\d.]+) - ntp_acc:', clean)
                m_val   = re.search(r'val_loss: ([\d.]+)', clean)
                if m_train:
                    tl = float(m_train.group(1))
                    # Only append if this is the final epoch line (has seconds/step)
                    if 's/step' in clean or 'ms/step' in clean:
                        dec_train_loss.append(tl)
                        if m_val:
                            dec_val_loss.append(float(m_val.group(1)))
                            dec_val_epochs.append(ep_idx)
                        break
    i += 1

dec_epochs = list(range(1, len(dec_train_loss) + 1))
print(f'Encoder: {len(enc_epochs)} epochs | Decoder: {len(dec_epochs)} epochs, {len(dec_val_epochs)} with val')

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

BLUE   = '#2166AC'
ORANGE = '#D6604D'
GRAY   = '#888888'

# ── Left: Encoder ────────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(enc_epochs, enc_train_loss, color=BLUE, lw=1.8)
ax.set_xlabel('Epoch', fontsize=10)
ax.set_ylabel('Cross-entropy loss', fontsize=10)
ax.set_title('Encoder (BERT-style, masked reconstruction)', fontsize=10, pad=6)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Right: Decoder ───────────────────────────────────────────────────────────
ax = axes[1]
ax.plot(dec_epochs, dec_train_loss, color=BLUE, lw=1.8)
ax.set_xlabel('Epoch', fontsize=10)
ax.set_ylabel('Cross-entropy loss', fontsize=10)
ax.set_title('Decoder (GPT-style, next-token prediction)', fontsize=10, pad=6)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout(pad=1.5)

for out in [
    'results/mtsm/training_curves_combined.png',
    'results/mtsm/encoder_global_norm/training_curves.png',
    'results/mtsm/decoder_global_norm/training_curves.png',
]:
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved {out}')
plt.close()
