"""
Turn saved dreamer-cdp training logs into the figures used in the README.

Save a run's output while it trains:
    caffeinate -i python train.py | tee runs/my_run.txt

Then plot one or more runs, each as path or path:label:
    python plot_log.py runs/first.txt:"first version" runs/fixed.txt:"V3 fixes" --out docs

Writes docs/forecasting.png, docs/representation.png and docs/returns.png.
"""
import argparse
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEYS = {"upd", "env", "wm", "cos", "base", "skill", "rew", "kl",
        "L_a", "L_c", "ret", "erank", "klraw", "gnorm"}
PAIR = re.compile(r"([A-Za-z_]+)\s+([-+]?\d+(?:\.\d+)?)")
COLORS = ["#1F5FAD", "#E8590C", "#6741D9", "#495057"]


def parse(path):
    """Read the 'upd ... | env ... | ...' lines from a train.py log.

    Handles lines where a tqdm progress bar is printed in front of the log line,
    and lines the terminal wrapped onto a second row.
    """
    rows = []
    with open(path, errors="ignore") as f:
        for line in f:
            i = line.find("upd ")
            if i >= 0:
                rec = {k: float(v) for k, v in PAIR.findall(line[i:]) if k in KEYS}
                if "upd" in rec and "env" in rec:
                    rows.append(rec)
            elif rows:
                extra = {k: float(v) for k, v in PAIR.findall(line) if k in KEYS}
                for k, v in extra.items():
                    rows[-1].setdefault(k, v)
    if not rows:
        raise SystemExit(f"no 'upd ... | env ...' log lines found in {path}")
    keys = set().union(*rows)
    return {k: np.array([r.get(k, np.nan) for r in rows]) for k in keys}


def smooth(y, k):
    """Moving average that ignores NaNs; k=1 returns the input."""
    if k <= 1:
        return y
    out = np.full_like(y, np.nan, dtype=float)
    for i in range(len(y)):
        w = y[max(0, i - k + 1): i + 1]
        w = w[~np.isnan(w)]
        if len(w):
            out[i] = w.mean()
    return out


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="log file, optionally path:label")
    ap.add_argument("--out", default="docs", help="output folder")
    ap.add_argument("--smooth", type=int, default=10, help="moving-average window, in log lines")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = []
    for spec in args.logs:
        path, _, label = spec.partition(":")
        runs.append((label or os.path.splitext(os.path.basename(path))[0], parse(path)))

    # 1. forecasting: cos against the lazy baseline, and skill
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), dpi=160)
    for c, (label, d) in zip(COLORS, runs):
        a1.plot(d["upd"], smooth(d["cos"], args.smooth), color=c, lw=1.6, label=f"{label}: cos")
        a1.plot(d["upd"], smooth(d["base"], args.smooth), color=c, lw=1.2, ls="--", label=f"{label}: base")
        a2.plot(d["upd"], smooth(d["skill"], args.smooth), color=c, lw=1.6, label=label)
    a2.axhline(0, color="#868E96", lw=0.8)
    style(a1, "Prediction vs. the lazy baseline", "update", "cosine")
    style(a2, "Skill = cos − base", "update", "skill")
    a1.legend(fontsize=7, frameon=False)
    a2.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "forecasting.png"))
    plt.close(fig)

    # 2. representation: effective rank of embeddings, and information entering the state
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), dpi=160)
    for c, (label, d) in zip(COLORS, runs):
        if "erank" in d:
            a1.plot(d["upd"], smooth(d["erank"], args.smooth), color=c, lw=1.6, label=label)
        if "klraw" in d:
            a2.plot(d["upd"], smooth(d["klraw"], args.smooth), color=c, lw=1.6, label=label)
    style(a1, "Effective rank of embeddings", "update", "effective rank")
    style(a2, "KL(posterior ‖ prior) before free bits", "update", "nats")
    for ax in (a1, a2):
        if ax.lines:
            ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "representation.png"))
    plt.close(fig)

    # 3. returns: 'ret' is the return of the most recent episode at each log line
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=160)
    for c, (label, d) in zip(COLORS, runs):
        ax.plot(d["env"], smooth(d["ret"], max(args.smooth, 20)), color=c, lw=1.6, label=label)
    style(ax, "Episode return, smoothed", "environment steps", "return")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "returns.png"))
    plt.close(fig)

    for label, d in runs:
        print(f"{label}: {len(d['upd'])} log lines, last update {int(d['upd'][-1])}, "
              f"last env step {int(d['env'][-1])}")
    print(f"wrote forecasting.png, representation.png, returns.png to {args.out}/")


if __name__ == "__main__":
    main()
