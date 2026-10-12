# Kyiv air-raid alert onset forecast

Command line tool that estimates the probability that **a new air raid alert starts in Kyiv City within the next hour**
from the history of alerts (moving base rate + a small LightGBM correction). Built for Agentic AI School 2026, Stage 2.

> The estimate is only slightly better than a plain "recent alert rate" guess. It is **not** a safety tool: rely on
> official air raid alerts and sirens.

Method, leakage rules, results and the path to the solution are in [PROJECT_NOTES.md](PROJECT_NOTES.md).

## 1. Setup

You need **Python 3.9 or newer** and **Git**.

**Get the code**

```
git clone https://github.com/AnnaProkopchenko20/TimeSeriesAnalysisAirRaidAlertsUkraine
cd TimeSeriesAnalysisAirRaidAlertsUkraine
```

**Create and activate a virtual environment**

Windows (PowerShell):

```
py -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell refuses to run the activation script, allow it for this window only and activate again:

```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```
python3 -m venv .venv
source .venv/bin/activate
```

When it worked, your prompt starts with `(.venv)`.

**Install the dependencies**

```
python -m pip install pandas numpy scikit-learn lightgbm
```

These four packages are everything the project needs. Their own requirements, such as the time zone database, are
installed automatically.

**Data**

The alert history is already in the repo: `datasets/volunteer_data_en.csv`, from
[Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset). To use fresher
alerts, replace that file with the latest `volunteer_data_en.csv` from the upstream repo, then re-run step 2.

## 2. Train the model

```
python train_model.py
```

Takes about a minute. It runs the automatic data-leakage checks, evaluates on the last 30 days of data (never used for
training), prints the results tables and saves the model to `artifacts/`.

## 3. Get a forecast

```
python cli.py --date 2026-10-10 --time 18:00
python cli.py
python cli.py --help
```

- `--date` is `YYYY-MM-DD` and `--time` is `HH:MM`, both in Kyiv time. Without arguments it uses the current time.
- Minutes are rounded down to the full hour: 18:40 is treated as 18:00.
- The result is the chance that a new alert starts between that hour and the next one. Numbers are shown in purple; add
  `--no-color` for plain text.
- If an alert in Kyiv is already active at that hour, it says so and gives no number. The model only predicts the start
  of a new alert.
- If the dataset ends up to 7 days before the time you ask about, you get the recent base rate only, with a warning.
  Further out it stops with an error. Update the dataset first for real use.

Example output:

```
Kyiv City: chance that a NEW air-raid alert starts between 18:00 and 19:00 (Sat 2026-10-10)
  probability:        xx.x%
  recent base rate:   xx.x%   (7d+56d blend of past hours; the model nudged it by +x.x pp)
```

## 4. Optional: compare model variants

```
python experiments.py
```

Compares moving-rate windows, blends and correction strengths on 30-day test blocks (takes under a minute).

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'pandas'` (or `lightgbm`) | The virtual environment is not active. Activate it (step 1) and run the install line again. |
| `py` is not recognised | Use `python` instead, or install Python from python.org and tick "Add to PATH". |
| `Activate.ps1 cannot be loaded` | Run the `Set-ExecutionPolicy -Scope Process ...` line from step 1, then activate again. |
| `cli.py` says the dataset ends too long before the requested time | Pick an earlier `--date`, or update `datasets/volunteer_data_en.csv`. |
| `cli.py` cannot find the model | Run `python train_model.py` first. |

## Files

| File | Purpose |
|---|---|
| `train_model.py` | features, moving rate, LightGBM, leakage tests, evaluation; saves `artifacts/` |
| `cli.py` | command line forecast on top of the saved model |
| `experiments.py` | comparison of the variants used to pick the final settings |
| `datasets/` | alert history csv |
| `artifacts/` | saved model (`kyiv_onset_lgbm.txt`) and its settings (`meta.json`) |
