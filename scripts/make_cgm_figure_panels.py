"""
Generate two-panel figure for thesis background section:
  Left  — 24 h CGM trace with TIR / TAR / TBR zones
  Right — table summarising all glycaemic metrics discussed in Section 2.1

Output: results/eda/cgm_metrics_panels.png
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATA_DIR   = "/mnt/workspace/tvae/data/processed/adults_global_norm"
OUT_PATH   = "/mnt/workspace/tvae/results/eda/cgm_metrics_panels.png"

CGM_MEAN = 144.40
CGM_STD  = 57.11

# ── 1. Find a representative 24 h window ─────────────────────────────────────

def score_window(cgm):
    """Higher = more representative (smooth, meals visible, balanced TIR, no hypo)."""
    if cgm.min() < 72 or cgm.max() > 285 or cgm.max() < 215:
        return -1
    mean_g = cgm.mean()
    if mean_g < 120 or mean_g > 165:
        return -1
    tir = np.mean((cgm >= 70) & (cgm <= 180))
    if tir < 0.45 or tir > 0.82:
        return -1
    # Penalise noisy/spiky traces
    steps = np.abs(np.diff(cgm))
    if steps.mean() > 4.0:
        return -1
    # Want at least 2 clear meal peaks above 200 mg/dL
    peaks = np.sum((cgm[1:-1] > cgm[:-2]) & (cgm[1:-1] > cgm[2:]) & (cgm[1:-1] > 200))
    if peaks < 2:
        return -1
    return float(peaks) * tir * (1 - tir) / steps.mean()

best_cgm, best_score = None, -1
for path in sorted(glob.glob(f"{DATA_DIR}/*.npz"))[:800]:
    d = np.load(path)
    cgm_z = d["windows"][:, :, 0]
    cgm   = cgm_z * CGM_STD + CGM_MEAN
    for row in cgm:
        s = score_window(row)
        if s > best_score:
            best_score, best_cgm = s, row

assert best_cgm is not None, "No suitable window found"
print(f"Selected window — min={best_cgm.min():.0f}  max={best_cgm.max():.0f}  "
      f"mean={best_cgm.mean():.0f}  std={best_cgm.std():.0f} mg/dL")

# ── 2. Layout ─────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 4.6))
gs  = gridspec.GridSpec(1, 2, width_ratios=[1.7, 1], wspace=0.05)

# ── 3. Left panel: CGM trace ─────────────────────────────────────────────────

ax = fig.add_subplot(gs[0])

time_h = np.linspace(0, 24, 288)

# Zone fills
TAR_COLOR = "#e6a817"
TIR_COLOR = "#3ab09e"
TBR_COLOR = "#d94f3d"

ax.axhspan(180, 310, alpha=0.13, color=TAR_COLOR, zorder=0)
ax.axhspan(70,  180, alpha=0.13, color=TIR_COLOR, zorder=0)
ax.axhspan(30,  70,  alpha=0.13, color=TBR_COLOR, zorder=0)

# Threshold dashes
ax.axhline(180, color=TAR_COLOR, linestyle="--", linewidth=1.1, alpha=0.85)
ax.axhline(70,  color=TBR_COLOR, linestyle="--", linewidth=1.1, alpha=0.85)

# Trace
ax.plot(time_h, best_cgm, color="#1a3a5c", linewidth=1.8, zorder=3)

# Zone annotations (inside axes, right-aligned)
ax.text(23.7, 248, "Hyperglycaemia\n(>180 mg/dL)",
        ha="right", va="center", fontsize=8.5, color=TAR_COLOR, linespacing=1.4)
ax.text(23.7, 125, "Target range\n(70–180 mg/dL)",
        ha="right", va="center", fontsize=8.5, color=TIR_COLOR, linespacing=1.4)
ax.text(23.7, 50,  "Hypoglycaemia\n(<70 mg/dL)",
        ha="right", va="center", fontsize=8.5, color=TBR_COLOR, linespacing=1.4)

ax.set_xlim(0, 24)
ax.set_ylim(30, 300)
ax.set_xticks(range(0, 25, 6))
ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 6)])
ax.set_xlabel("Time of day (hours)", fontsize=10)
ax.set_ylabel("CGM (mg/dL)", fontsize=10)
ax.set_title("Continuous glucose monitor trace, 24 h window", fontsize=10.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=9)

# ── 4. Right panel: metrics table ─────────────────────────────────────────────

ax_t = fig.add_subplot(gs[1])
ax_t.axis("off")

rows = [
    ["TIR",   "% time 70–180 mg/dL",          "> 70%"],
    ["TAR",   "% time > 180 mg/dL",            "< 25%"],
    ["TBR",   "% time < 70 mg/dL",             "< 4%"],
    ["HbA1c", "3-month mean glycaemia proxy",   "< 7%"],
    ["GRI",   "Weighted hypo/hyper burden",     "0 = best"],
    ["ISF",   "mg/dL lowered per unit insulin", "Patient-specific"],
    ["CR",    "g carbs covered per unit insulin","Patient-specific"],
]
col_hdrs = ["Metric", "Definition", "Target"]

HEADER_COLOR = "#1a3a5c"
ALT_ROW      = "#eef2f7"

tbl = ax_t.table(
    cellText=rows,
    colLabels=col_hdrs,
    loc="center",
    cellLoc="left",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.8)
tbl.auto_set_column_width([0, 1, 2])
tbl.scale(1, 1.52)

# Header row
for j in range(3):
    cell = tbl[0, j]
    cell.set_facecolor(HEADER_COLOR)
    cell.set_text_props(color="white", fontweight="bold")
    cell.set_edgecolor("#ffffff")

# Data rows
for i in range(1, len(rows) + 1):
    bg = ALT_ROW if i % 2 == 0 else "#ffffff"
    for j in range(3):
        cell = tbl[i, j]
        cell.set_facecolor(bg)
        cell.set_edgecolor("#cccccc")

ax_t.set_title("Glycaemic metrics at a glance", fontsize=10.5, pad=12)

# ── 5. Save ───────────────────────────────────────────────────────────────────

plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
