"""Kyiv City air-raid ONSET forecast (final): two-timescale moving base rate + small LightGBM correction.

    pip install pandas numpy scikit-learn lightgbm
    python train_model.py

Question: at each full hour t (UTC grid, Kyiv-local calendar features), and only if no Kyiv City alert is active at t,
what is P(a NEW Kyiv City alert starts in (t, t+1h])?

Model:  logit(p) = logit(rate(t)) + CORRECTION_SCALE * f(features(t))
  rate(t) = average (in logit space) of a SHORT (7d) and a LONG (56d) shrunk onset rate, each computed only from labels
            already completed by t (rows <= t-1h). Short reacts to the current regime, long anchors it.
  f       = tiny LightGBM trained on top of that offset (init_score), so it only learns the deviation from the rate.

Why these choices (experiments.py, 5 selection blocks of 30 days, mean log loss; lower is better):
  * LGBM correction on top of the rate : ~ -0.008   (0.3322 -> 0.3240 on the 28d rate)  <- the one clearly useful step
  * 7d+56d blend instead of 28d rate   : ~ -0.0007  (with LGBM on; essentially a tie, kept because it won 4/5 blocks)
  * shrinkage K (5/20/50)               : <= 0.0002  (irrelevant, kept at 20)
  * shrinking the correction (x0.5)     : +0.0025 on the selection blocks -> kept at 1.0 (see CORRECTION_SCALE note)

Leakage rules enforced in code (see leakage_checks):
  * every feature at t uses data with timestamp <= t; ongoing alerts expose only "still active" and elapsed time
  * train < validation < test in time, with a 1h gap (a label at t looks up to t+1h)
  * the last 30 days (which contain the official last 14) are NEVER used for fitting, early stopping or choosing anything
  * every rate at t reads only labels with row time <= t-1h (past outcomes, like any other lag feature)
"""
import json
import pathlib
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).parent.resolve()
DATA = ROOT / "datasets" / "volunteer_data_en.csv"
OUT = ROOT / "artifacts"
KYIV = "Kyiv City"
NEIGHBOURS = ["Kyivska oblast", "Chernihivska oblast", "Zhytomyrska oblast", "Poltavska oblast", "Cherkaska oblast",
              "Sumska oblast"]
TEST_DAYS, VAL_DAYS, N_FOLDS = 30, 56, 6  # fold 0 = holdout (last 30 days); folds 1.. = earlier blocks used for selection
RATE_WINDOW_DAYS, SHRINK_K, HALF_LIFE_DAYS, WARMUP_DAYS = 28, 20, 180, 30  # 28d rate = the first version, kept as baseline
SHORT_DAYS, LONG_DAYS = 7, 56  # final rate = logit-average of the 7d and 56d shrunk rates
# 1.0 is what the selection blocks prefer. On the last 30 days the correction HURTS (x0.5 or x0.25 would be better there),
# but the holdout must not be used to choose; if you lower this after looking at the holdout, say so in the reflection.
CORRECTION_SCALE = 1.0
H, MIN = np.timedelta64(1, "h"), np.timedelta64(1, "m")
CALENDAR = ["hour_sin", "hour_cos"]
FEATS = CALENDAR + ["log_since_end", "kyiv_starts_1h", "kyiv_starts_3h", "kyiv_starts_24h", "n_oblasts_active"] + \
        [f"act_{r.split()[0]}" for r in NEIGHBOURS]
FINAL, REF = "FINAL: 7d+56d rate + LGBM", "const: 7d+56d rate"


def ns(s) -> np.ndarray:
    """tz-aware UTC timestamps -> plain datetime64[ns] numpy array."""
    s = pd.Series(s)
    return s.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").to_numpy()


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ----------------------------------------------------------------------------- features
def build_table(df: pd.DataFrame, until=None) -> pd.DataFrame:
    """One row per full hour t with features known at t and label y = Kyiv alert STARTS in (t, t+1h].
    Rows where Kyiv is already under alert at t are dropped (onset question)."""
    df = df[df.region != "Luhanska oblast"]  # 3 stray rows (permanent siren artifact)
    grid_end = until if until is not None else df.started_at.max().floor("h") - pd.Timedelta(hours=1)
    grid = pd.date_range(df.started_at.min().ceil("h"), grid_end, freq="h")
    t = ns(grid)
    X = pd.DataFrame(index=grid)
    local = grid.tz_convert("Europe/Kyiv")
    X["hour"] = local.hour
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * local.hour / 24), np.cos(2 * np.pi * local.hour / 24)

    n_active = np.zeros(len(t))
    for region, g in df.groupby("region"):
        s, e = np.sort(ns(g.started_at)), np.sort(ns(g.finished_at))
        n_started, n_ended = np.searchsorted(s, t, side="right"), np.searchsorted(e, t, side="right")
        active = (n_started - n_ended) > 0  # ongoing at t; the end itself is never read, only compared with t
        n_active += active
        if region in NEIGHBOURS:
            X[f"act_{region.split()[0]}"] = active.astype(int)
        if region == KYIV:
            ks, kactive = s, active
            X["kyiv_since_end"] = np.where(n_ended > 0, (t - e[np.maximum(n_ended - 1, 0)]) / MIN, 1e5).clip(max=4320)
            for h in (1, 3, 24):
                X[f"kyiv_starts_{h}h"] = n_started - np.searchsorted(s, t - np.timedelta64(h, "h"), side="right")
    X["n_oblasts_active"] = n_active
    X["log_since_end"] = np.log1p(X.kyiv_since_end)
    X["y"] = (np.searchsorted(ks, t + H, side="right") - np.searchsorted(ks, t, side="right")) > 0
    X["kyiv_active"] = kactive
    return X[~X.kyiv_active].drop(columns="kyiv_active")


def window_rate(idx, y, window_days, k):
    """shrunk onset rate over (t-window, t-1h], using only rows <= t-1h (label already observed at t)."""
    cs = np.concatenate([[0.0], np.cumsum(y)])
    hi = np.searchsorted(idx, idx - H, side="right")
    lo = np.searchsorted(idx, idx - np.timedelta64(window_days, "D"), side="right")
    n, pos = hi - lo, cs[hi] - cs[lo]
    prior = np.where(hi > 0, cs[hi] / np.maximum(hi, 1), 0.06)  # long-run rate known at t
    return (pos + k * prior) / (n + k)


def add_moving_rate(tab: pd.DataFrame) -> pd.DataFrame:
    """First-version rate (28d window): kept as a baseline and for the leakage check. Also adds rate_all."""
    idx, y = ns(tab.index), tab.y.to_numpy(float)
    cs = np.concatenate([[0.0], np.cumsum(y)])
    hi = np.searchsorted(idx, idx - H, side="right")
    tab = tab.copy()
    tab["rate_all"] = np.where(hi > 0, cs[hi] / np.maximum(hi, 1), 0.06)
    tab["rate_recent"] = window_rate(idx, y, RATE_WINDOW_DAYS, SHRINK_K)
    tab["logit_rate"] = logit(tab.rate_recent)
    return tab


def add_final_rate(tab: pd.DataFrame) -> pd.DataFrame:
    """Final rate: logit-average of the 7d and 56d shrunk rates (short = reacts, long = anchors)."""
    idx, y = ns(tab.index), tab.y.to_numpy(float)
    z = 0.5 * (logit(window_rate(idx, y, SHORT_DAYS, SHRINK_K)) + logit(window_rate(idx, y, LONG_DAYS, SHRINK_K)))
    tab = tab.copy()
    tab["rate_final"] = sigmoid(z)
    tab["logit_rate_final"] = logit(tab.rate_final)
    return tab


# ----------------------------------------------------------------------------- leakage tests
def leakage_checks(raw: pd.DataFrame, tab: pd.DataFrame, test_start: pd.Timestamp) -> None:
    """Automated proof that features/rates at t do not depend on anything after t."""
    rng = np.random.default_rng(0)
    test_rows = tab[tab.index >= test_start].index
    for t0 in rng.choice(test_rows, 3, replace=False):
        t0 = pd.Timestamp(t0).tz_localize("UTC") if pd.Timestamp(t0).tzinfo is None else pd.Timestamp(t0)
        # 1) rebuild features from a world that ENDS at t0: later alerts dropped, unfinished alerts get an unknown end
        r = raw[raw.started_at <= t0].copy()
        r["finished_at"] = r.finished_at.where(r.finished_at <= t0, t0 + pd.Timedelta(days=36500))
        cols = [c for c in build_table(r, until=t0).columns if c not in ("y", "hour")]
        a, b = build_table(r, until=t0).loc[t0, cols], tab.loc[t0, cols]
        assert np.allclose(a.astype(float), b.astype(float)), f"feature leakage at {t0}:\n{pd.concat([a, b], axis=1)}"
        # 2) brute-force moving base rates from past labels only
        past = tab[tab.index <= t0 - pd.Timedelta(hours=1)]

        def brute(days):
            win = past[past.index > t0 - pd.Timedelta(days=days)]
            return (win.y.sum() + SHRINK_K * past.y.mean()) / (len(win) + SHRINK_K)

        assert np.isclose(brute(RATE_WINDOW_DAYS), tab.loc[t0, "rate_recent"]), f"base-rate leakage at {t0}"
        if "rate_final" in tab.columns:
            expected = sigmoid(0.5 * (logit(brute(SHORT_DAYS)) + logit(brute(LONG_DAYS))))
            assert np.isclose(expected, tab.loc[t0, "rate_final"]), f"final-rate leakage at {t0}"
    print("leakage checks passed: features rebuilt from truncated data are identical; rates use only past labels")


# ----------------------------------------------------------------------------- one train/val/test fold
def fit_lgbm(tr, va, w, offset_col):
    model = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=4, min_child_samples=80,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=20.0, verbose=-1)
    model.fit(tr[FEATS], tr.y, sample_weight=w, init_score=tr[offset_col].to_numpy(),
              eval_set=[(va[FEATS], va.y)], eval_init_score=[va[offset_col].to_numpy()],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def run_fold(tab: pd.DataFrame, te_start: pd.Timestamp, te_end: pd.Timestamp) -> dict:
    gap, val_start = pd.Timedelta(hours=1), te_start - pd.Timedelta(days=VAL_DAYS)
    te = tab[(tab.index >= te_start) & (tab.index < te_end)]
    va = tab[(tab.index >= val_start) & (tab.index < te_start - gap)]
    tr = tab[tab.index < val_start - gap]
    assert tr.index.max() + gap < va.index.min() and va.index.max() + gap <= te.index.min(), "split overlap"
    assert tr.index.max() < te_start - pd.Timedelta(days=VAL_DAYS), "training touches validation/test period"

    w = np.asarray(0.5 ** ((tr.index.max() - tr.index).total_seconds() / 86400 / HALF_LIFE_DAYS))
    final = fit_lgbm(tr, va, w, "logit_rate_final")
    first = fit_lgbm(tr, va, w, "logit_rate")  # first version: 28d rate + LGBM, for the before/after comparison
    preds = {
        "const: all-history rate": te.rate_all.to_numpy(),
        "const: 28d rate (first version)": te.rate_recent.to_numpy(),
        REF: te.rate_final.to_numpy(),
        "LGBM on 28d rate (first version)": sigmoid(first.predict(te[FEATS], raw_score=True) + te.logit_rate.to_numpy()),
        FINAL: sigmoid(CORRECTION_SCALE * final.predict(te[FEATS], raw_score=True) + te.logit_rate_final.to_numpy()),
    }
    return dict(te=te, preds=preds, model=final, n_trees=final.best_iteration_, n=(len(tr), len(va), len(te)))


def metrics(y, p) -> dict:
    return dict(logloss=log_loss(y, p, labels=[0, 1]), brier=brier_score_loss(y, p),
                pr_auc=average_precision_score(y, p), roc_auc=roc_auc_score(y, p) if np.ptp(p) > 0 else np.nan)


def row_loss(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


if __name__ == "__main__":
    raw = pd.read_csv(DATA, parse_dates=["started_at", "finished_at"])
    full = add_final_rate(add_moving_rate(build_table(raw)))
    tab = full[full.index >= full.index.min() + pd.Timedelta(days=WARMUP_DAYS)]  # drop the noisy first weeks
    end = tab.index.max() + pd.Timedelta(hours=1)
    test_start = end - pd.Timedelta(days=TEST_DAYS)
    print(f"data through {end}  |  HOLDOUT = [{test_start}, {end})  never used for fitting / early stopping / selection")
    leakage_checks(raw, full, test_start)

    # ---- all folds: fold 0 = holdout (last 30 days), folds 1.. = earlier 30-day blocks ---------------
    folds = []
    for i in range(N_FOLDS):
        e_i = end - pd.Timedelta(days=TEST_DAYS * i)
        folds.append(run_fold(tab, e_i - pd.Timedelta(days=TEST_DAYS), e_i))
    res = folds[0]
    te, y = res["te"], res["te"].y.to_numpy()
    print(f"\nrows train/val/holdout = {res['n']}   holdout positive rate = {y.mean():.3f}   LGBM trees = {res['n_trees']}")

    table = pd.DataFrame({k: metrics(y, p) for k, p in res["preds"].items()}).T
    ref = row_loss(y, res["preds"][REF])
    rng = np.random.default_rng(1)
    for name, p in res["preds"].items():
        d = row_loss(y, p) - ref
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        table.loc[name, f"dLL vs '{REF}' [95% CI]"] = f"{d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]"
    last14 = te.index >= end - pd.Timedelta(days=14)
    table["logloss_last14d"] = [log_loss(y[last14], np.asarray(p)[last14], labels=[0, 1]) for p in res["preds"].values()]
    print("\nHOLDOUT (last 30 days; the official last 14 days are the 'logloss_last14d' column)\n",
          table.round(4).to_string())

    p = res["preds"][FINAL]
    cal = pd.DataFrame({"pred": p, "obs": y, "bin": pd.qcut(p, 4, duplicates="drop")})
    print("\ncalibration of FINAL on the holdout (4 quantile bins):\n", cal.groupby("bin", observed=True).agg(
        mean_pred=("pred", "mean"), observed=("obs", "mean"), n=("obs", "size")).round(3).to_string())

    # ---- selection blocks vs holdout --------------------------------------------------------------
    rows = []
    for i, r_i in enumerate(folds):
        for name, pr in r_i["preds"].items():
            rows.append(dict(fold=i, model=name, ll=log_loss(r_i["te"].y, pr, labels=[0, 1])))
    r = pd.DataFrame(rows)
    wide = r.pivot(index="fold", columns="model", values="ll")
    summ = pd.DataFrame({"sel_ll (folds 1-5)": wide.loc[1:].mean(), "holdout30 (fold 0)": wide.loc[0]})
    summ["sel_blocks_better_than_ref"] = [(wide.loc[1:, m] < wide.loc[1:, REF]).sum() for m in summ.index]
    print(f"\nROLLING: {N_FOLDS - 1} selection blocks of {TEST_DAYS} days + the holdout (ref = '{REF}')\n",
          summ.sort_values("sel_ll (folds 1-5)").round(4).to_string())
    print("\nNote: constant-predictor log loss is set by the base rate (about 0.33 at ~10%, about 0.53 at ~22%), so the "
          "holdout column is higher mostly because the last 30 days have many more alerts, not because models got worse.")

    # ---- save the evaluated model (trained without the last 30 days) ------------------------------
    OUT.mkdir(exist_ok=True)
    res["model"].booster_.save_model(str(OUT / "kyiv_onset_lgbm.txt"))
    (OUT / "meta.json").write_text(json.dumps(dict(
        features=FEATS, short_days=SHORT_DAYS, long_days=LONG_DAYS, shrink_k=SHRINK_K,
        correction_scale=CORRECTION_SCALE, trained_before=str(test_start),
        note="rate = sigmoid(0.5*(logit(rate_7d)+logit(rate_56d))), each shrunk rate from labels of past no-alert hours "
             "(rows <= t-1h); prediction = sigmoid(correction_scale*booster.predict(raw_score=True) + logit(rate))"),
        indent=2))
    print(f"\nsaved model to {OUT}")
