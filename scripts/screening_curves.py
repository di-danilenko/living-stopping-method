"""
Prioritised screening in a living systematic review:
  1. Baseline screening in batches of 100, three updates, red line = stopping point
  2. Reordered curves (re-ranked at each checkpoint) from update 2 onwards

Illustrative data only. To use real data, replace the BASELINE block with your
screening log (documents screened vs cumulative includes) and build the reordered
curves with `reordered_from_scores` (see below).

Requires: numpy, matplotlib
"""

#TODO: move the legend into top left corner

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASELINE = "#320e3b"
REORDERING = "#679436"
STOP = "#E24B4A"
SKIP = "#9A9A94"

POOL = 1000                       # documents in the sample of each update
N_UPDATES = 3
XMAX = POOL * N_UPDATES
PREVALENCE = 0.05                 # total includes = 5% of all documents
YMAX = PREVALENCE * XMAX
 
#  Cumulative includes after every 100 screened documents, per update
# (offset 0 = start of the update, value = includes found so far overall)
# Values are actual include counts and must end at YMAX (150 = 5% of 3000)
inc_by_offset = [
    [0, 6, 31, 47, 55, 59, 62, 63],             # baseline review: 700 screened
    [63, 93, 106, 111, 112, 113, 113],          # update 1: 600 screened
    [113, 141, 148, 149, 150, 150],             # update 2: 500 screened
]
assert inc_by_offset[-1][-1] == YMAX, "last value must equal total includes (PREVALENCE * XMAX)"
reordered_screened = [700, 500, 400]            
 
# Reordered curves: (update index, checkpoint offset, bend position on the x axis)
reordered_checkpoints = [
    (u, off, 0.6 * (pool_start_ + off))          # demo bend; real data: from the scores
    for u in (1, 2)                              # reordering starts at update 2
    for pool_start_ in [u * POOL]
    for off in range(100, reordered_screened[u] + 1, 100)   # every checkpoint up to the stop
]
 
def pool_start(u):
    return u * POOL
 

def baseline_xy():
    xs, ys = [], []
    for u, vals in enumerate(inc_by_offset):
        xs += [pool_start(u) + 100 * i for i in range(len(vals))]
        ys += vals
    return np.array(xs), np.array(ys)
 

def includes_at(u, offset):
    return np.interp(offset, 100 * np.arange(len(inc_by_offset[u])), inc_by_offset[u])
 

def reordered_demo(x_end, total, bend, k=0.4, steps=60):
    """Smooth curve from the origin with a single bend, then a plateau.
    REAL DATA: replace with the cumulative sum of labels sorted by prediction score."""
    t = np.linspace(0, 1, steps)
    x = 2 * (1 - t) * t * (k * bend) + t ** 2 * bend
    y = total * (2 * t - t ** 2)
    return np.r_[x, x_end], np.r_[y, total]
 

def reordered_from_scores(scores, labels):
    """REAL DATA: re-rank documents seen so far (highest score first)."""
    order = np.argsort(-np.asarray(scores))
    return np.arange(1, len(order) + 1), np.cumsum(np.asarray(labels)[order])
 
 
def style_axes(ax, stops):
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX+5)
    ax.set_yticks([0, YMAX])
    ax.set_yticklabels(["0", f"{YMAX:.0f}"])
    ax.set_xticks(np.arange(0, XMAX + 1, 500))
    ax.set_xticks(np.arange(0, XMAX + 1, 100), minor=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("Cumulative included documents")
    for s in stops:
        ax.axvline(s, color=STOP, lw=1.5, zorder=2)
        ax.text(s, -0.09, str(s), color=STOP, ha="center", va="top", fontsize=9,
                transform=ax.get_xaxis_transform())
    for u in range(N_UPDATES):
        ax.text(pool_start(u) + POOL / 2, -0.17, f"Update {u + 1}" + (" (random start)" if u == 0 else ""),
                ha="center", va="top", fontsize=9, transform=ax.get_xaxis_transform())
        if u > 0:
            ax.axvline(pool_start(u), color="grey", lw=0.8, ls=":", zorder=1)
 
 
def skipped_blocks(ax, stops):
    for u, s in enumerate(stops):
        end = pool_start(u) + POOL
        ax.axvspan(s, end, facecolor="none", edgecolor=SKIP, hatch="////", lw=0, alpha=0.5, zorder=0)
        ax.text((s + end) / 2, YMAX * 0.5, f"not screened\n{end - s} docs", ha="center",
                va="center", fontsize=8.5, color="#5f5e5a", zorder=3)
 
 
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9.5))
fig.subplots_adjust(hspace=0.55)
 
bx, by = baseline_xy()
baseline_stops = [pool_start(u) + 100 * (len(v) - 1) for u, v in enumerate(inc_by_offset)]
baseline_stop_y = [v[-1] for v in inc_by_offset]
 
# --- Panel 1 ---------------------------------------------------------------
ax1.set_title("1. Baseline: Calculate p-score using the existing screening curve ", loc="left", fontweight="bold")
skipped_blocks(ax1, baseline_stops)
for u, v in enumerate(inc_by_offset):                       # one line per update
    x = pool_start(u) + 100 * np.arange(len(v))
    ax1.plot(x, v, color=BASELINE, lw=2, zorder=4)
    ax1.plot(x, v, "o", mfc="none", mec=BASELINE, ms=5, zorder=4)
    if u < N_UPDATES - 1:                                   # carry includes across the blank
        ax1.plot([x[-1], pool_start(u + 1)], [v[-1], v[-1]], color=BASELINE, lw=1, ls=":", zorder=3)
ax1.plot(baseline_stops, baseline_stop_y, "o", color=BASELINE, ms=7, zorder=5)
style_axes(ax1, baseline_stops)
 
# --- Panel 2 ---------------------------------------------------------------
reordered_stops = [pool_start(u) + s for u, s in enumerate(reordered_screened)]
ax2.set_title("2. Reordering: Use model scores to reorder the documents at each checkpoint and calculate p-score", loc="left", fontweight="bold")
skipped_blocks(ax2, reordered_stops)
for u, v in enumerate(inc_by_offset):                       # baseline for reference
    x = pool_start(u) + 100 * np.arange(len(v))
    ax2.plot(x, v, color=BASELINE, lw=1.5, ls=(0, (4, 4)), zorder=3)
    if u < N_UPDATES - 1:
        ax2.plot([x[-1], pool_start(u + 1)], [v[-1], v[-1]], color=BASELINE, lw=1, ls=":", zorder=3)
ax2.plot(reordered_stops[0], inc_by_offset[0][-1], "o", color=BASELINE, ms=7, zorder=5)
 
for u, off, bend in reordered_checkpoints:
    x_end = pool_start(u) + off
    total = includes_at(u, off)
    x, y = reordered_demo(x_end, total, bend)
    ax2.plot(x, y, color=REORDERING, lw=2, alpha=0.5, zorder=4)
    ax2.plot(x_end, total, "o", color=REORDERING, alpha=0.6, ms=6, zorder=5)
 
style_axes(ax2, reordered_stops)
ax2.set_xlabel("Total records seen (retraining and stopping checks at every 100)", labelpad=30)
 
ax2.legend(handles=[
    Line2D([], [], color=BASELINE, lw=1.5, ls=(0, (4, 4)), label="Baseline"),
    Line2D([], [], color=REORDERING, lw=2, alpha=0.5, label="Reordering"),
    Line2D([], [], color=STOP, lw=1.5, label="Stop"),
    Patch(facecolor="none", edgecolor=SKIP, hatch="////", alpha=0.6, label="Not screened"),
], loc="lower right", frameon=False)
 
fig.savefig("screening_curves.png", dpi=300, bbox_inches="tight")
print("saved screening_curves.png")
