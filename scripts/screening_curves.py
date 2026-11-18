import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from buscarpy import retrospective_h0

BASELINE = "#320e3b"
REORDERING = "#679436"
STOP = "#E24B4A"
SKIP = "#9A9A94"

RECALL_TARGET = 0.95
ALPHA = 0.05  

POOL = 1000                       # documents in the sample of each update
N_UPDATES = 3
XMAX = POOL * N_UPDATES
PREVALENCE = 0.05                 # relevance rate in the search results
YMAX = PREVALENCE * XMAX
 
# Cumulative includes after every 100 screened documents, per update
# (offset 0 = start of the update, value = includes found so far overall)
# Values are actual include counts and must end at YMAX (150 = 5% of 3000)
inc_by_offset = [
    [0, 10, 38, 53, 59, 62, 63, 63, 63],        # baseline review: 800 screened, p = 0.039
    [63, 93, 106, 111, 112, 113, 113, 113, 113],  # update 1: 800 screened, p = 0.023
    [113, 141, 148, 149, 150, 150, 150, 150],   # update 2: 700 screened, p = 0.039
]

assert inc_by_offset[-1][-1] == YMAX, "last value must equal total includes (PREVALENCE * XMAX)"

baseline_screened = [100 * (len(v) - 1) for v in inc_by_offset]       

def pool_start(u):
    return u * POOL
 
def N_at(u):
    """Haystack at update u: every pool sampled so far (unscreened documents stay in it)."""
    return POOL * (u + 1)

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
 

# --- p-scores (buscarpy) -----------------------------------------------------
def baseline_labels():
    """Binary label sequence in screening order, rebuilt from the cumulative
    include counts (includes spread evenly inside each block of 100).
    REAL DATA: use the actual screening-order labels."""
    labels = []
    for vals in inc_by_offset:
        for d in np.diff(vals):
            block = np.zeros(100, dtype=int)
            block[np.floor((np.arange(d) + 0.5) * 100 / d).astype(int)] = 1
            labels.append(block)
    return np.concatenate(labels)
 
 
def reordered_labels_demo(n, total, frac=0.6):
    """Demo: n labels ranked by score, includes front-loaded, then a plateau.
    REAL DATA: use labels[np.argsort(-scores)] for the documents seen so far."""
    bend_n = max(int(round(frac * n)), 1)
    r = np.arange(n + 1)
    cum = np.where(r < bend_n, total * (1 - (1 - r / bend_n) ** 2), total)
    return np.diff(np.round(cum)).astype(int)
 
 
def pscore(labels, batch_size, N):
    """One call of buscarpy's retrospective_h0. confidence_level=1 so it never
    stops early and returns a p-score at every batch."""
    res = retrospective_h0(labels, N=N, recall_target=RECALL_TARGET,
                           batch_size=batch_size, confidence_level=1.0, plot=False)
    return res["p"]
 
 
def baseline_pscores():
    """p-score at every 100 screened documents, placed at its x-axis position."""
    
    labels = baseline_labels()
    xs, ps, n0 = [], [], 0
    for u, n in enumerate(baseline_screened):
        p = pscore(labels[:n0 + n], batch_size=100, N=N_at(u))[n0 // 100:]
        xs += [pool_start(u) + 100 * i for i in range(1, n // 100 + 1)]
        ps += list(p)
        n0 += n
    return np.array(xs), np.array(ps, dtype=float)

 
def reordered_p(u, off, n_prev):
    """p-score at each reordering checkpoint, from the re-ranked labels."""
    n_seen = n_prev + off
    total = int(round(includes_at(u, off)))
    labels = reordered_labels_demo(n_seen, total)
    return float(pscore(labels, batch_size=n_seen, N=N_at(u))[-1])      # single batch


def derive_reordered():
    """Reordering starts in update 1. The stop of each update is the first checkpoint
    (every 100) where p <= ALPHA. Returns the stops (screened per update), the
    checkpoints to draw and the p-scores (x, p)."""
    screened = [baseline_screened[0]]                  # update 0 is not reordered
    checkpoints, xs, ps = [], [], []
    for u in (1, 2):
        n_prev = sum(screened)
        for off in range(100, baseline_screened[u] + 1, 100):
            p = reordered_p(u, off, n_prev)
            checkpoints.append((u, off, 0.6 * (pool_start(u) + off)))   # demo bend
            xs.append(pool_start(u) + off)
            ps.append(p)
            if p <= ALPHA:
                screened.append(off)
                break
        else:
            raise AssertionError(f"update {u}: reordering never reaches p <= {ALPHA}")
    return screened, checkpoints, np.array(xs), np.array(ps)
 

 
 
def add_pscore_axis(ax, xs, ps, sim_type):
    ax_p = ax.twinx()
    ax_p.scatter(xs, ps, marker="o", s=22, color=sim_type, zorder=1)
    ax_p.axhline(ALPHA, color="black", lw=0.8, ls="dashed", alpha=0.8, zorder=1)
    ax_p.set_ylim(0, 1.05)
    ax_p.set_yticks([0, ALPHA, 0.5, 1])
    ax_p.set_yticklabels(["0", f"{ALPHA:g}", "0.5", "1"])
    ax_p.set_ylabel(f"p-score (H0: recall < {RECALL_TARGET:.0%})", color="black")
    ax_p.tick_params(axis="y", colors="black")
    ax_p.spines[["top", "left"]].set_visible(False)
    ax_p.spines["right"].set_color("black")
    return ax_p

 
def style_axes(ax, stops):
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX+5)
    ax.set_yticks([0, YMAX])
    ax.set_yticklabels(["0", f"{YMAX:.0f}"])
    ax.set_xticks(np.arange(0, XMAX + 1, 500))
    ax.set_xticks(np.arange(0, XMAX + 1, 100), minor=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("Cumulative included documents")
    for u, s in enumerate(stops):
        # grey line every 100 records within the screened part of this update
        for x in range(pool_start(u) + 100, int(s), 100):
            ax.axvline(x, color="black", lw=0.5, ls="dashed", alpha=0.5, zorder=2)
        ax.axvline(s, color=STOP, lw=1.5, zorder=2)
    for u in range(N_UPDATES):
        ax.text(pool_start(u) + POOL / 2, -0.12, f"Update {u}",
                ha="center", va="top", fontsize=9, transform=ax.get_xaxis_transform())
        if u > 0:
            ax.axvline(pool_start(u), color="grey", lw=0.5, ls=":", zorder=1)
 
 
def skipped_blocks(ax, stops):
    for u, s in enumerate(stops):
        end = pool_start(u) + POOL
        ax.axvspan(s, end, facecolor="none", edgecolor=SKIP, hatch="////", lw=0, alpha=0.5, zorder=0)
        ax.text((s + end) / 2, YMAX * 0.5, f"\n{end - s} docs", ha="center",
                va="center", fontsize=7.5, color="#5f5e5a", zorder=3)
 
 
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9.5))
fig.subplots_adjust(hspace=0.4)
 
bx, by = baseline_xy()
baseline_stops = [pool_start(u) + 100 * (len(v) - 1) for u, v in enumerate(inc_by_offset)]
baseline_stop_y = [v[-1] for v in inc_by_offset]
 
# --- Panel 1 ---------------------------------------------------------------
ax1.set_title("1. Baseline: Calculate p-score using the existing screening curve ", loc="left", 
              fontsize=10)
skipped_blocks(ax1, baseline_stops)
for u, v in enumerate(inc_by_offset):                       # one line per update
    x = pool_start(u) + 100 * np.arange(len(v))
    ax1.plot(x, v, color=BASELINE, lw=2, zorder=4)
    #ax1.plot(x, v, "o", mfc="none", mec=BASELINE, ms=5, zorder=4)
    #if u < N_UPDATES - 1:                                   # carry includes across the blank
    #    ax1.plot([x[-1], pool_start(u + 1)], [v[-1], v[-1]], color=BASELINE, lw=1, ls=":", zorder=3)
#ax1.plot(baseline_stops, baseline_stop_y, "o", color=BASELINE, ms=7, zorder=5)

style_axes(ax1, baseline_stops)
px1, pp1 = baseline_pscores()
add_pscore_axis(ax1, px1, pp1, BASELINE)
 
# --- Panel 2 ---------------------------------------------------------------
reordered_screened, reordered_checkpoints, px2, pp2 = derive_reordered()
reordered_stops = [pool_start(u) + s for u, s in enumerate(reordered_screened)]
ax2.set_title("2. Reordering: Use model scores to reorder the documents at each checkpoint and calculate p-score", loc="left", 
              fontsize=10)
skipped_blocks(ax2, reordered_stops)
for u, v in enumerate(inc_by_offset):                       # baseline for reference
    x = pool_start(u) + 100 * np.arange(len(v))
    ax2.plot(x, v, color=BASELINE, lw=1.5, ls=(0, (4, 4)), zorder=3)
    #if u < N_UPDATES - 1:
    #    ax2.plot([x[-1], pool_start(u + 1)], [v[-1], v[-1]], color=BASELINE, lw=1, ls=":", zorder=3)
#ax2.plot(reordered_stops[0], inc_by_offset[0][-1], "o", color=BASELINE, ms=7, zorder=5)
 
for u, off, bend in reordered_checkpoints:
    x_end = pool_start(u) + off
    total = includes_at(u, off)
    x, y = reordered_demo(x_end, total, bend)
    ax2.plot(x, y, color=REORDERING, lw=2, alpha=0.5, zorder=4)
    ax2.plot(x_end, total, "o", color=REORDERING, alpha=0.6, ms=6, zorder=5)
 
style_axes(ax2, reordered_stops)
ax2.set_xlabel("Total records seen (retraining and stopping checks at every 100)", labelpad=35)

n0 = baseline_screened[0]                                   # update 0 is identical to the baseline

ax_p2 = add_pscore_axis(ax2, np.r_[px1[:n0 // 100], px2], np.r_[pp1[:n0 // 100], pp2], REORDERING)
leg = ax_p2.legend(handles=[
    Line2D([], [], color=BASELINE, lw=1.5, ls=(0, (4, 4)), label="Baseline"),
    Line2D([], [], color=REORDERING, lw=2, alpha=0.5, label="Reordering"),
    Line2D([], [], color=STOP, lw=1.5, label="Stop"),
    Patch(facecolor="none", edgecolor=SKIP, hatch="////", alpha=0.6, label="Not screened"),
], loc="upper left", frameon=True, framealpha=1, facecolor="white", edgecolor="none", fontsize=7.5)
leg.set_zorder(10)

fig.savefig("screening_curves.png", dpi=300, bbox_inches="tight")
print("saved screening_curves.png")
