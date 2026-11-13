# Polymarket NVIDIA CEO quantile forecaster

Entry for make.fun bounty `polymarket-nvidia-ceo-forecast-01`: forecast the
Yes price of every market of Polymarket's "Next CEO of NVIDIA?" event at
every minute of the 4 hours after a checkpoint, as five quantiles
(5%, 25%, 50%, 75%, 95%).

## Method

`main.py` (Python 3.13, stdlib only, deterministic, offline) forecasts each
market with a shared cross-market drift:

- **Level:** the last price at the checkpoint.
- **Median path:** the level, pulled toward a cross-market sum of 1. All
  the event's markets are for one outcome (the next NVIDIA CEO), so their
  Yes prices should sum to ~1; when the sum at the checkpoint is off 1,
  every market's median drifts proportionally back toward it, linearly
  over the 240 minutes. Mean reversion toward a market's own recent
  mean/median, slope reversal and logit-space modelling were all tested
  and lost.
- **Volatility:** trimmed RMS of the 1-minute price changes over the last
  1440 minutes — the largest 5% of squared changes are dropped, then
  changes are winsorized at ±0.05. Trimming matters: when a market is
  first listed its price sits near 0.48, then settles near 0.003, and that
  one launch jump would otherwise dominate any RMS. Floored at 0.0003,
  capped at 0.02, multiplied by 0.6.
- **Spread:** `sigma * sqrt(h)` for horizon `h` minutes.
- **Quantiles:** asymmetric — realised moves skew down (the book usually
  carries an overround), so the downside bands are wider:
  z = −3.0 / −0.9 / 0 / +0.4 / +2.0.

Ideas tested and rejected on the practice cases: median of the last 5
minutes as the level (lags genuine moves); pulling the median toward the
recent mean (the recent mean is contaminated by stale/launch prices);
logit-space modelling (worse at every setting); volatility clustering
(scaling sigma by recent vol — worse); empirical 240-minute move
quantiles from each market's own history (far too wide — history
overcounts one-off jumps); relative volatility floors (sigma ∝ level —
worse).

## Local validation

No local grader exists for this competition (program/v1, server-graded),
so the practice cases were scored locally against answers re-fetched from
Polymarket's public price history (`GET /clob.polymarket.com/prices-history`,
fidelity=1), gridded exactly as the rules describe. On that scorer:

| case | this entry (v2) | v1 | house baseline |
|------|-----------|-----------|----------------|
| p00 | 0.4246 | 0.5072 | 0.5361 |
| p01 | 0.2948 | 0.2972 | 0.3804 |
| p02 | 0.3636 | 0.3655 | 0.4084 |
| p03 | 0.2365 | 0.2405 | 0.3686 |
| p04 | 0.2008 | 0.2112 | 0.2988 |
| p05 | 0.1090 | 0.1071 | 0.2382 |
| **mean** | **0.2715** | **0.2881** | **0.3718** |

Mature cases (p03–p05, like the final's 12 cases) average 0.182, vs
0.186 for v1. The live leaderboard (checked 22:12 WITA, 1,170 entries)
has rank 1 at 0.240, rank 20 at 0.250 and median 0.276 — so v2 at
0.272 sits around rank ~500 on practice, still outside the paid top 20.
The gains came from the cross-market sum drift (−0.009), the spread
re-tune and asymmetric quantiles (−0.008 combined); every other
structural idea tested made it worse. Honest read: the median is
essentially unpredictable from price history alone (perfect-median
oracle: 0.062; true-quantile oracle: 0.259), and this entry is close
to the ceiling of what a price-history-only forecaster can do here.

Two runs on the same cases produce byte-identical output; 6 cases run in
~0.45 s total, far under the 8 s per-case cap.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
