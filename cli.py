"""Tiny CLI: probability that a NEW air-raid alert starts in Kyiv City within the hour after a given time.

    python cli.py --date 2026-10-10 --time 18:00     # Kyiv local time
    python cli.py                                    # now; needs a dataset that is up to date

Needs artifacts/ (run train_model.py first) and the dataset csv. The features are rebuilt from alert history that ends
at the requested time (later alerts are dropped, unfinished ones get an unknown end), the same way the leakage test
does it, so asking about a past hour never peeks at what happened afterwards.

If the dataset ends up to --max-gap-days (default 7) before the requested time, the CLI still answers, but only with the
recent base rate and a warning (alerts in the gap are unknown, so the live-state features and the LightGBM correction
cannot be used). Further out it refuses.

HOW TO USE THE NUMBER: this is a rough base-rate style estimate, not a precise forecast. On the last 30 days the model
did not beat a plain recent-alert-rate (ROC-AUC about 0.5). It tells you roughly how busy Kyiv has been lately.
It is NOT a safety tool: rely on official air-raid alerts and sirens.
"""
import argparse
import json
import os
import sys

import lightgbm as lgb
import pandas as pd

import train_model as tm

TZ = "Europe/Kyiv"
if os.name == "nt":
    os.system("")  # makes the Windows console understand ANSI colour codes
USE_COLOR = (sys.stdout.isatty() or "FORCE_COLOR" in os.environ) and "NO_COLOR" not in os.environ


def hl(text) -> str:
    """highlight a number in bold purple (plain text when output is piped, NO_COLOR is set or --no-color is used)"""
    return f"\033[1;38;5;135m{text}\033[0m" if USE_COLOR else str(text)

NOTE = ("Use with care: estimates are only slightly better than a plain recent-alert-rate guess (ranking skill is near\n"
        "chance on the latest data). Read it as 'how busy has Kyiv been lately', not as a prediction of one hour.\n"
        "It only covers the START of a NEW alert. Do NOT use it for safety decisions; follow official alerts/sirens.")


def parse_args():
    now = pd.Timestamp.now(tz=TZ)
    ap = argparse.ArgumentParser(description="P(new air-raid alert starts in Kyiv City in the next hour).")
    ap.add_argument("--date", default=now.strftime("%Y-%m-%d"), help="Kyiv local date, YYYY-MM-DD (default: today)")
    ap.add_argument("--time", default=now.strftime("%H:%M"), help="Kyiv local time, HH:MM (default: now); "
                                                                   "rounded down to the full hour")
    ap.add_argument("--data", default=str(tm.DATA), help="alerts csv (default: datasets/volunteer_data_en.csv)")
    ap.add_argument("--artifacts", default=str(tm.OUT), help="folder with the saved model (default: artifacts/)")
    ap.add_argument("--max-gap-days", type=float, default=7.0,
                    help="how far past the end of the dataset a time may be (default 7); inside that gap only the "
                         "recent base rate is shown, beyond it the CLI refuses")
    ap.add_argument("--no-color", action="store_true", help="plain output without the purple highlighting")
    return ap.parse_args()


def stale_forecast(raw, t, data_end) -> int:
    """The dataset ends BEFORE t, so alerts in the gap are unknown and the live-state features (is an alert active,
    alerts in the last hours, time since the last one) cannot be built. We do NOT fill the gap with fake 'no alert'
    hours (that would drag the rate towards 0). Instead: recent base rate from the labels that really exist, and the
    LightGBM correction is skipped because its inputs are unknown."""
    tab = tm.build_table(raw)  # its last row is the last hour whose label is fully known
    dummy = tab.iloc[[-1]].copy()
    dummy.index = pd.DatetimeIndex([t])
    dummy["y"] = False  # never read: rates at t only use rows <= t-1h
    rate = float(tm.add_final_rate(pd.concat([tab, dummy])).loc[t, "rate_final"])
    local, end_local = t.tz_convert(TZ), data_end.tz_convert(TZ)
    print(f"Kyiv City: chance that a NEW air-raid alert starts between {local:%H:%M} and "
          f"{(local + pd.Timedelta(hours=1)):%H:%M} ({local:%a %Y-%m-%d})")
    print(f"  probability:        {hl(f'{rate:.1%}')}   (recent base rate only, see warning)")
    print(f"  WARNING: the dataset ends {end_local:%Y-%m-%d %H:%M} Kyiv time, "
          f"{hl(f'{(t - data_end) / pd.Timedelta(hours=1):.1f} h')} "
          "before this time.\n  Alerts since then are unknown, so the current alert state and recent-alert counts cannot "
          "be computed.\n  This is the recent base rate from the data that exists (the LightGBM correction is skipped), "
          "so it says\n  nothing specific about this hour, and it does not apply if an alert is active right now.")
    print("\n" + NOTE)
    return 0


def main() -> int:
    global USE_COLOR
    args = parse_args()
    if args.no_color:
        USE_COLOR = False
    try:
        t = pd.Timestamp(f"{args.date} {args.time}").tz_localize(TZ).tz_convert("UTC").floor("h")
    except Exception as e:  # bad format, or a time that does not exist / is ambiguous around the DST change
        print(f"error: could not read '{args.date} {args.time}' as Kyiv time ({e})", file=sys.stderr)
        return 2
    meta_path, model_path = (f"{args.artifacts}/meta.json", f"{args.artifacts}/kyiv_onset_lgbm.txt")
    try:
        meta = json.load(open(meta_path))
        booster = lgb.Booster(model_file=model_path)
        raw = pd.read_csv(args.data, parse_dates=["started_at", "finished_at"])
    except FileNotFoundError as e:
        print(f"error: {e.filename} not found. Run train_model.py first and check --data/--artifacts.", file=sys.stderr)
        return 2
    assert meta["features"] == tm.FEATS, "saved model and train_model.py use different features; retrain"

    data_end = max(raw.started_at.max(), raw.finished_at.max()).floor("h")
    if t - data_end > pd.Timedelta(days=args.max_gap_days):
        print(f"error: the dataset ends at {data_end.tz_convert(TZ):%Y-%m-%d %H:%M} (Kyiv), more than "
              f"{args.max_gap_days:g} days before the requested time. Update the dataset or pick an earlier time.",
              file=sys.stderr)
        return 2
    if t > data_end:
        return stale_forecast(raw, t, data_end)

    # world that ENDS at t: later alerts dropped, unfinished alerts get an unknown end (no peeking at the future)
    r = raw[raw.started_at <= t].copy()
    r["finished_at"] = r.finished_at.where(r.finished_at <= t, t + pd.Timedelta(days=36500))
    tab = tm.add_final_rate(tm.build_table(r, until=t))

    local = t.tz_convert(TZ)
    when = f"{local:%a %Y-%m-%d %H:%M} Kyiv time"
    if args.time != f"{local:%H:%M}":
        print(f"(requested {args.time}; the model works on full hours, so using {local:%H:%M})")
    if t not in tab.index:
        print(f"At {when} an alert in Kyiv City is already active (or the history is too short), so there is nothing "
              "to forecast: this model only predicts the start of a NEW alert.")
        return 0

    row = tab.loc[[t]]
    rate = float(row.rate_final.iloc[0])
    correction = float(booster.predict(row[tm.FEATS], raw_score=True)[0])
    p = float(tm.sigmoid(meta["correction_scale"] * correction + tm.logit(rate)))

    print(f"Kyiv City: chance that a NEW air-raid alert starts between {local:%H:%M} and "
          f"{(local + pd.Timedelta(hours=1)):%H:%M} ({local:%a %Y-%m-%d})")
    print(f"  probability:        {hl(f'{p:.1%}')}")
    print(f"  recent base rate:   {hl(f'{rate:.1%}')}   (7d+56d blend of past hours; the model nudged it by "
          f"{hl(f'{100 * (p - rate):+.1f} pp')})")
    if t < pd.Timestamp(meta["trained_before"]):
        print(f"  warning: {when} is inside the model's training period (before "
              f"{pd.Timestamp(meta['trained_before']):%Y-%m-%d}), so this is not a fair out-of-sample prediction.")
    print("\n" + NOTE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
