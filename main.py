"""Quantile forecaster for Polymarket's "Next CEO of NVIDIA?" event.
CANDIDATE v3 (round 4, NOT shipped): v2 + trimmed-EWMA volatility.

Reads one case at a time (/input/<id>.in: every market's Yes price per minute
for the 72h up to a checkpoint) and writes /output/<id>.out with, for each
market and each of the 240 minutes after the checkpoint, five quantiles
(5%, 25%, 50%, 75%, 95%) of that minute's price.

Method, per market (stdlib only, deterministic, no network, no model calls):
  - level: the last price at the checkpoint.
  - median path: the level, pulled toward a cross-market sum of 1. All the
    event's markets are for one outcome (the next NVIDIA CEO), so their Yes
    prices should sum to ~1; when the sum at the checkpoint is off 1, the
    median of every market drifts proportionally back toward it, linearly
    over the 240 minutes: med(h) = level * (1 + (1-sum)/sum * h/240).
  - sigma1: EXPONENTIALLY-WEIGHTED trimmed RMS of the 1-minute price
    changes over the last 1440 minutes (half-life 120 min) -- the largest
    5% of squared changes are dropped first (a launch jump would otherwise
    dominate any RMS), then changes are winsorized at +-0.05 and weighted
    by recency. Floored at 0.0003, capped at 0.02, multiplied by 0.6.
    (v2 used a flat trimmed RMS over the same window; the EWMA version is
    the only round-4 idea that improved all three validation sets, by
    ~0.0015 each -- below the 0.003 bar, so this file was NOT shipped.)
  - spread(h) = sigma1 * sqrt(h).
  - quantiles are ASYMMETRIC: realised moves skew down (the book usually
    carries an overround, so prices drift down more often than up), so the
    downside bands are wider: z = -3.0 / -0.9 / 0 / +0.4 / +2.0.

Local validation (rebuilt round-4 scorer, answers re-fetched from
Polymarket's public price history):
  practice (6 cases):  mean 0.2698 vs v2's 0.2715 (mature p03-p05: 0.1807 vs 0.1821)
  pseudo-checkpoints (10): mean 0.2935 vs v2's 0.2950
  per practice case: p00 0.415, p01 0.299, p02 0.362, p03 0.237, p04 0.199, p05 0.106
"""

import json
import math
import os
from statistics import median

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def _f(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


# --- tuning parameters (defaults = this candidate) ---
WIN = int(os.environ.get("POLY_WIN", "1440"))          # lookback minutes for diffs
HALFLIFE = _f("POLY_HALFLIFE", 120.0)                # EWMA half-life in minutes
TRIM = _f("POLY_TRIM", 0.05)                         # fraction of largest squared diffs dropped
SIG_FLOOR = _f("POLY_FLOOR", 0.0003)                 # min per-minute sigma
SIG_CAP = _f("POLY_CAP", 0.02)                       # max per-minute sigma
SIG_MULT = _f("POLY_MULT", 0.6)                      # global sigma multiplier
Z95U = _f("POLY_Z95U", 2.0)                          # up-tail quantile
Z95D = _f("POLY_Z95D", 3.0)                          # down-tail quantile
Z75U = _f("POLY_Z75U", 0.4)                          # up inner quantile
Z75D = _f("POLY_Z75D", 0.9)                          # down inner quantile
ALPHA = _f("POLY_ALPHA", 0.5)                        # spread growth exponent
LEVEL = os.environ.get("POLY_LEVEL", "last")         # "med5" or "last"
WINSOR = _f("POLY_WINSOR", 0.05)                     # winsorize 1-min diffs at +-this
SUM_K = _f("POLY_SUM_K", 1.0)                        # cross-market sum drift strength

Z = [-Z95D, -Z75D, 0.0, Z75U, Z95U]


def price6(x):
    """Clamp to [0, 1] and round to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def forecast_market(prices, horizon, sum_drift=0.0):
    n = len(prices)
    if n == 0:
        raise ValueError("market has no prices")
    if LEVEL == "last" or n < 5:
        level = prices[-1]
    else:
        level = median(prices[-5:])
    # per-minute volatility: trimmed, exponentially-weighted RMS of recent diffs
    m = min(WIN, n - 1)
    if m < 10:
        sigma1 = 0.0025
    else:
        diffs = []
        for i in range(n - m, n):
            d = prices[i] - prices[i - 1]
            if d > WINSOR:
                d = WINSOR
            elif d < -WINSOR:
                d = -WINSOR
            diffs.append(d)
        sq = sorted(d * d for d in diffs)
        keep_n = max(10, int((1.0 - TRIM) * len(sq)))
        cut = sq[keep_n - 1]
        lam = 0.5 ** (1.0 / HALFLIFE)
        wsum = 0.0
        acc = 0.0
        w = 1.0
        for d in reversed(diffs):
            if d * d > cut:
                continue
            acc += w * d * d
            wsum += w
            w *= lam
        sigma1 = math.sqrt(acc / wsum) if wsum > 0 else 0.0025
        if sigma1 < SIG_FLOOR:
            sigma1 = SIG_FLOOR
        elif sigma1 > SIG_CAP:
            sigma1 = SIG_CAP
    sigma1 *= SIG_MULT
    rows = []
    for h in range(1, horizon + 1):
        # median drifts proportionally toward a cross-market sum of 1
        med = level * (1.0 + SUM_K * sum_drift * (h / horizon))
        spread = sigma1 * (h ** ALPHA)
        rows.append([price6(med + z * spread) for z in Z])
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fname in inputs:
        case_id = fname[: -len(".in")]
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        horizon = int(case["horizon_minutes"])
        levels = [mkt["prices"][-1] for mkt in case["markets"]]
        sum_x = sum(levels)
        # the sum signal only means something for a (near-)complete book
        sum_drift = (1.0 - sum_x) / sum_x if len(levels) >= 5 and sum_x > 0 else 0.0
        forecasts = []
        for mkt in case["markets"]:
            forecasts.append({"id": mkt["id"], "q": forecast_market(mkt["prices"], horizon, sum_drift)})
        out_path = os.path.join(OUTPUT_DIR, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
