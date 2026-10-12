"""Experiments on top of train_model.py: which moving base rate works best, and does the LightGBM correction help?

Run from the repo root (next to train_model.py):

    python experiments.py                      # uses datasets/volunteer_data_en.csv
    python experiments.py some_other_file.csv  # optional custom data file

Tested separately, all on 30-day test blocks (about 4x more rows per fold than 7-day-noisy 14-day blocks):
  1. window length x shrinkage grid for the moving rate         (const rate, then LGBM on the best)
  2. blend of a short (7d) and a long (56d) rate                (logit average)
  3. exponentially weighted rate (half-life 7 / 14 / 28 days)
  5. 30-day blocks everywhere (fold 0 = last 30 days, which contains the official last 14 days)
  6. shrinking the LGBM correction: logit(p) = logit(rate) + s * f(x), s in {1, 0.5, 0.25}

Model selection hygiene: variants are CHOSEN using folds 1..5 only (older blocks). Fold 0 (the last 30 days) is a
holdout that is only reported, never used to pick anything. "last14" is the official last-14-day slice of fold 0.

Leakage rules (all enforced / tested in code):
  * features at t: reused from train_model.build_table + train_model.leakage_checks
  * every rate at t uses only labels of rows with time <= t-1h (their outcome window ended by t)
  * rate_leakage_test: recompute each rate from data truncated at t with the label at t flipped -> must be identical
  * train < validation < test with a 1h gap; the fold's test block is never used for fitting or early stopping
"""
import functools
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import train_model as tm
from train_model import FEATS, H, add_moving_rate, build_table, leakage_checks, logit, ns, sigmoid

DATA = sys.argv[1] if len(sys.argv) > 1 else tm.DATA
TEST_DAYS, VAL_DAYS, N_FOLDS, WARMUP_DAYS, HALF_LIFE_DAYS = 30, 56, 6, 30, 180
SCALES = (1.0, 0.5, 0.25)  # experiment 6
BASELINE = "win28_k20"  # what train_model.py uses now
D = lambda n: pd.Timedelta(days=n)


# ----------------------------------------------------------------------------- moving rates (all causal)
def rate_window(idx, y, window_days, k):
    """shrunk onset rate over (t-window, t-1h]; only rows <= t-1h (label already observed at t)."""
    cs = np.concatenate([[0.0], np.cumsum(y)])
    hi = np.searchsorted(idx, idx - H, side="right")
    lo = np.searchsorted(idx, idx - np.timedelta64(window_days, "D"), side="right")
    n, pos = hi - lo, cs[hi] - cs[lo]
    prior = np.where(hi > 0, cs[hi] / np.maximum(hi, 1), 0.06)  # long-run rate known at t
    return (pos + k * prior) / (n + k)


def rate_ewm(idx, y, half_life_days, k=20):
    """exponentially weighted rate: a label that is a half-life old counts half as much. Only rows <= t-1h."""
    td = (idx - idx[0]) / np.timedelta64(1, "D")
    avail = np.searchsorted(idx, idx - H, side="right")  # rows whose label is known at t
    cs = np.concatenate([[0.0], np.cumsum(y)])
    out, num, den, last, j = np.empty(len(idx)), 0.0, 0.0, 0.0, 0
    for i in range(len(idx)):
        while j < avail[i]:
            d = 0.5 ** ((td[j] - last) / half_life_days)
            num, den, last = num * d + y[j], den * d + 1.0, td[j]
            j += 1
        d = 0.5 ** ((td[i] - last) / half_life_days)
        prior = cs[avail[i]] / avail[i] if avail[i] > 0 else 0.06
        out[i] = (num * d + k * prior) / (den * d + k)
    return out


def rate_blend(idx, y, short=7, long=56, k=20):
    z = 0.5 * (logit(rate_window(idx, y, short, k)) + logit(rate_window(idx, y, long, k)))
    return sigmoid(z)


def rate_registry():
    fns = {f"win{w}_k{k}": functools.partial(rate_window, window_days=w, k=k) for w in (7, 14, 28, 56) for k in (5, 20, 50)}
    fns["blend7+56"] = rate_blend
    fns.update({f"ewm{h}d": functools.partial(rate_ewm, half_life_days=h) for h in (7, 14, 28)})
    return fns


def rate_leakage_test(fns, rates, full, end, n_checks=5):
    """For random test-period hours t: rebuild each rate from data that ENDS at t and whose label at t is flipped.
    If the rate at t changed, it would mean the rate peeks at t's own label or at anything later."""
    idx, y = ns(full.index), full.y.to_numpy(float)
    rng = np.random.default_rng(0)
    cand = np.where(full.index >= end - D(TEST_DAYS))[0]
    for k in rng.choice(cand, n_checks, replace=False):
        y_cut = y[: k + 1].copy()
        y_cut[k] = 1 - y_cut[k]  # the label at t looks into (t, t+1h]: it must not matter
        for name, f in fns.items():
            assert np.isclose(f(idx[: k + 1], y_cut)[-1], rates[name][k]), f"rate leakage: {name} at {full.index[k]}"
    print(f"rate leakage test passed for {len(fns)} rate variants at {n_checks} random test hours")


# ----------------------------------------------------------------------------- folds
def fold_bounds(end):
    """fold 0 = last 30 days (holdout), folds 1..N-1 = earlier consecutive 30-day blocks (used for selection)."""
    return [(end - D(TEST_DAYS * (i + 1)), end - D(TEST_DAYS * i)) for i in range(N_FOLDS)]


def with_rate(full, rate):
    t = full.assign(rate_recent=rate, logit_rate=logit(rate))
    return t[t.index >= t.index.min() + D(WARMUP_DAYS)]  # rates are computed on the full table first, then trimmed


def split(tab, te_start, te_end):
    gap, val_start = pd.Timedelta(hours=1), te_start - D(VAL_DAYS)
    te = tab[(tab.index >= te_start) & (tab.index < te_end)]
    va = tab[(tab.index >= val_start) & (tab.index < te_start - gap)]
    tr = tab[tab.index < val_start - gap]
    assert tr.index.max() + gap < va.index.min() and va.index.max() + gap <= te.index.min(), "split overlap"
    assert tr.index.max() < te_start - D(VAL_DAYS), "training touches validation/test period"
    return tr, va, te


def fit_fold(tab, te_start, te_end):
    tr, va, te = split(tab, te_start, te_end)
    w = np.asarray(0.5 ** ((tr.index.max() - tr.index).total_seconds() / 86400 / HALF_LIFE_DAYS))
    model = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=4, min_child_samples=80,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=20.0, verbose=-1)
    model.fit(tr[FEATS], tr.y, sample_weight=w, init_score=tr.logit_rate.to_numpy(),
              eval_set=[(va[FEATS], va.y)], eval_init_score=[va.logit_rate.to_numpy()],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    corr = model.predict(te[FEATS], raw_score=True)  # the learned deviation f(x), WITHOUT the offset
    preds = {"const rate": te.rate_recent.to_numpy()}
    for s in SCALES:  # experiment 6: same trained model, correction shrunk by s
        preds[f"LGBM x{s}"] = sigmoid(s * corr + te.logit_rate.to_numpy())
    return te, preds


def record(store, variant, i, te, preds, end):
    y = te.y.to_numpy().astype(int)
    last14 = (te.index >= end - D(14))
    for model, p in preds.items():
        d = store.setdefault((variant, model), dict(ll={}, last14=np.nan))
        d["ll"][i] = log_loss(y, p, labels=[0, 1])
        if i == 0:
            d["last14"] = log_loss(y[last14], np.asarray(p)[last14], labels=[0, 1])


def run_const(store, variant, full, rate, end):
    tab = with_rate(full, rate)
    for i, (s, e) in enumerate(fold_bounds(end)):
        te = tab[(tab.index >= s) & (tab.index < e)]
        record(store, variant, i, te, {"const rate": te.rate_recent.to_numpy()}, end)


def run_lgbm(store, variant, full, rate, end):
    tab = with_rate(full, rate)
    for i, (s, e) in enumerate(fold_bounds(end)):
        te, preds = fit_fold(tab, s, e)
        record(store, variant, i, te, preds, end)


def summarize(store, kind, ref_key):
    ref = np.array([store[ref_key]["ll"][i] for i in range(N_FOLDS)])
    rows = []
    for (variant, model), d in store.items():
        if not model.startswith(kind):
            continue
        ll = np.array([d["ll"][i] for i in range(N_FOLDS)])
        rows.append(dict(variant=variant, model=model, sel_ll=ll[1:].mean(), holdout30=ll[0], last14=d["last14"],
                         wins_vs_ref=int((ll[1:] < ref[1:]).sum()), d_vs_ref=ll[1:].mean() - ref[1:].mean()))
    return pd.DataFrame(rows).sort_values("sel_ll").round(4).to_string(index=False)


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    t0 = time.time()
    raw = pd.read_csv(DATA, parse_dates=["started_at", "finished_at"])
    full = build_table(raw)
    idx, y = ns(full.index), full.y.to_numpy(float)
    end = full.index.max() + pd.Timedelta(hours=1)
    print(f"data through {end} | holdout fold 0 = last {TEST_DAYS} days [{end - D(TEST_DAYS)}, {end}) | "
          f"selection folds 1..{N_FOLDS - 1} are the {N_FOLDS - 1} blocks before it")

    leakage_checks(raw, add_moving_rate(full), end - D(TEST_DAYS))  # feature leakage tests from train_model.py
    fns = rate_registry()
    rates = {name: f(idx, y) for name, f in fns.items()}
    rate_leakage_test(fns, rates, full, end)

    store = {}
    for name, r in rates.items():  # cheap: constant-rate score of every variant on all 6 blocks
        run_const(store, name, full, r, end)

    def best(names):
        return min(names, key=lambda n: np.mean([store[(n, "const rate")]["ll"][i] for i in range(1, N_FOLDS)]))

    best_grid = best([n for n in rates if n.startswith("win")])
    best_ewm = best([n for n in rates if n.startswith("ewm")])
    print(f"\nEXP 1 grid winner by const-rate sel_ll: {best_grid} | EXP 3 EWM winner: {best_ewm}")
    for name in dict.fromkeys([BASELINE, best_grid, "blend7+56", best_ewm]):  # LGBM only for the finalists
        print(f"  training LGBM folds for {name} ... ({time.time() - t0:.0f}s elapsed)")
        run_lgbm(store, name, full, rates[name], end)

    print(f"\n=== A. constant moving rate only (no model); ref = {BASELINE} ===")
    print("    sel_ll = mean log loss on folds 1..5 (used to choose) | holdout30 = last 30 days | last14 = official slice")
    print(summarize(store, "const", (BASELINE, "const rate")))
    print(f"\n=== B. LGBM correction on top of the rate; ref = {BASELINE} + LGBM x1.0 (current train_model.py) ===")
    print("    x1.0 = full correction, x0.5 / x0.25 = shrunk correction (exp 6); const rate = correction switched off")
    print(summarize(store, "LGBM", (BASELINE, "LGBM x1.0")))
    print(f"\ndone in {time.time() - t0:.0f}s. Pick the variant with the lowest sel_ll; trust holdout30/last14 only as a "
          "sanity check (they are small and were not used to choose).")
