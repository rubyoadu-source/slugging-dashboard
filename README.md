# SlugGuard: flow-assurance decision-support dashboard (research prototype)

Explainable AI for slugging risk, built on nine OLGA simulations.
It **identifies the flow state of the current record** (proxy label), **explains** the flag, and lists
**checks and options for an engineer to review**. It does not forecast, and it never sends commands.

## Files

| File | Purpose |
|---|---|
| `app.py` | the Streamlit interface (nothing else) |
| `slug_core.py` | features, models, quality checks, exact Shapley explanations, rules, leave-one-scenario-out check |
| `ingest.py`, `consolidate.py` | read raw OLGA workbooks (values are hidden inside Excel formulas) and reject corrupted or duplicate sheets |
| `data/training_data.csv` | the nine scenarios with the proxy label (2,601 records) |
| `artifacts/` | optional `models_full.joblib` from Colab (the app trains the same models itself if absent) |
| `selfcheck.py` | confirms the logic reproduces the report (5 of 9 and 7 of 9) |
| `colab_export_cell.py` | the cell to paste into the Colab notebook when the data changes |
| `.streamlit/config.toml` | dark theme (optional) |

## Deploy on Streamlit Community Cloud

1. GitHub: create a repository `slugging-dashboard`, choose Add file > Upload files, and drag in everything in this folder.
   If the hidden `.streamlit` folder is skipped, use Add file > Create new file, type `.streamlit/config.toml` and paste its text (the theme is optional).
2. Open https://share.streamlit.io, sign in with GitHub, choose Create app, pick the repository, branch `main`, main file `app.py`.
3. Under Advanced settings pick the same Python minor version as your Colab (`!python --version` in Colab), then Deploy.
4. The first load takes 10 to 20 seconds while the models are trained once and cached.

## Run locally (optional)

    pip install -r requirements.txt
    python selfcheck.py
    streamlit run app.py

## When you have more scenarios or a real OLGA flow-regime label

1. Re-run the Colab notebook with the new workbooks.
2. Paste `colab_export_cell.py` as a new cell at the end of the notebook and run it. It downloads `training_data.csv` and `models_full.joblib`.
3. Replace `data/training_data.csv` (and `artifacts/models_full.joblib`) in the repository and commit. The app restarts by itself.
4. `selfcheck.py` expects the current nine scenarios and will fail once the data grows. That is intended; update its expected numbers after you have read the new results.

## Plugging in a live feed (next stage, not built)

The app already runs on a replayed record, so a live source only has to supply a table with these columns:
`TIME_h` (h), `QLT` (m3/s), `UL` (m/s), `UG` (m/s), `VALVDP` (bar, optional), `gas_lift_MMSCFD`,
`topside_choke_TCV_pct_open` (blank when unrestricted). Gas lift and choke are model inputs, so they must be streamed too.
Still needed before any real use: a read-only gateway (historian API, OPC UA or MQTT), authentication,
unit and timestamp checks on the live data (the per-record checks in `slug_core.quality_flags` are the start),
hosting suited to continuous monitoring, and validation against OLGA flow-regime output and field data.

## Limits

* The classes are a proxy label, not OLGA flow regimes.
* Nine scenarios, each almost one class: on unseen scenarios logistic regression named the dominant class in 7 of 9, gradient boosting in 5 of 9.
* Scores are not calibrated probabilities. Explanations describe the model, not physical causes.
* Research prototype on simulated data. Not for operational use.
