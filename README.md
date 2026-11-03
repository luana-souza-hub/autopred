# AutoPred

Code and results of a study that compares ten forecasting models on **accuracy**, **robustness to adverse data** and **explainability**, under a single protocol, on three monthly datasets with different characteristics (M4 Monthly, Tourism Monthly and Car Parts).

The study is described in the article *Regime-dependent performance, robustness and explainability of regression models for time series forecasting* (Gomes de Souza and Sipoli Sanches, submitted) and in the master's dissertation from which it derives (Graduate Program in Informatics, UTFPR, Cornélio Procópio, Brazil).

## What is here

| Folder | Content |
|---|---|
| `src/` | Pipeline (`pipeline_v2.py`), the single implementation of models, features, windows, scenarios, metrics and explainability used by all results |
| `scripts/` | Command-line runners: `run_v2.py` (forecasting and scenarios) and `run_xai_v2.py` (SHAP and LIME) |
| `tests/` | Unit and regression tests of the pipeline |
| `notebooks/` | Notebooks that read the results and produce the tables and figures (files with `_v2` use the official pipeline) |
| `tools/` | Auxiliary scripts: figures, hyperparameter report, consistency checks |
| `results/v2/` | Official results: per-window outputs, tables, explainability outputs and `hiperparametros_v2.md` (the effective hyperparameters of every model) |

Only the files with `v2` in the name and the folder `results/v2/` correspond to the official protocol. Older notebooks are kept for history.

## The protocol in brief

- **Models (10, four families):** persistence, seasonal naive, historical mean; SES, Holt-Winters; SARIMA with a fixed order and SARIMA with the order selected by AIC; random forest, gradient boosting and SVR. No hyperparameter search: all configurations are fixed in advance (see `results/v2/hiperparametros_v2.md`).
- **Windows:** three forecast origins per series with an expanding training window (minimum 24 observations). Horizon 18 (M4, Car Parts) and 24 (Tourism).
- **Scenarios (9):** Gaussian noise (10% and 30%), missing values (10%, 20%, 30%), outliers (2%, 5%) and truncation of the history (50%, 25%), applied to the training data only. Car Parts is evaluated on clean data only.
- **Metrics:** MAPE, MAE and MASE. Per-window values are aggregated by the median within each series and then across series. Confidence intervals come from a paired bootstrap over series (2,000 resamples, seed 42).
- **Explainability:** SHAP and LIME on random forest, gradient boosting and SVR, for 30 series per dataset, with stability measured under 30% noise.
- **Seeds:** global seed 42; the 30 series for explainability are drawn with seed 43.

## Data

The raw data are **not** included (the folder `data/raw/` is ignored by git). Download them from the original sources and place them as follows:

| File in `data/raw/` | Source |
|---|---|
| `Monthly-train.csv`, `Monthly-test.csv` | M4 Competition, monthly series. Repository: <https://github.com/Mcompetitions/M4-methods> (folder `Dataset`). Makridakis, Spiliotis and Assimakopoulos (2020), doi:10.1016/j.ijforecast.2019.04.014 |
| `tourism_monthly_dataset.tsf` | Tourism Monthly, Monash Time Series Forecasting Archive, doi:10.5281/zenodo.4656096. Athanasopoulos et al. (2011), doi:10.1016/j.ijforecast.2010.04.009 |
| `car_parts/car_parts_dataset_without_missing_values.tsf` | Car Parts (without missing values), Monash Time Series Forecasting Archive, doi:10.5281/zenodo.4656021 (CC BY 4.0) |

The Monash archive is described in Godahewa et al. (2021), *Monash Time Series Forecasting Archive*, NeurIPS Datasets and Benchmarks Track. Each result file has a `.meta.json` with the SHA-256 of the data it used.

## Setup

Python 3.11 was used.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the results

Run from the root of the project (on Windows, use `.venv\Scripts\python.exe` instead of `python`).

```bash
# 1. Tests of the pipeline (some need the data in data/raw/)
python -m unittest -v tests.test_pipeline_v2

# 2. Clean data (fast models, then the automatic SARIMA, which is slow)
python scripts/run_v2.py --base m4 --etapa limpo --modelos rapidos --n-jobs 6
python scripts/run_v2.py --base m4 --etapa limpo --modelos sarima_auto --n-jobs 6

# 3. One adverse scenario (the automatic SARIMA is not run on scenarios)
python scripts/run_v2.py --base m4 --etapa C_outliers_2pct --modelos rapidos

# 4. Explainability, clean and with 30% noise
python scripts/run_xai_v2.py --base m4 --condicao limpo
python scripts/run_xai_v2.py --base m4 --condicao ruido30
```

`--base` is `m4`, `cp` (Car Parts) or `tm` (Tourism). The scenario names are `A_ruido_10pct`, `A_ruido_30pct`, `B_ausencias_10pct`, `B_ausencias_20pct`, `B_ausencias_30pct`, `C_outliers_2pct`, `C_outliers_5pct`, `D_trunca_50pct` and `D_trunca_25pct`. Interrupted runs resume from the partial file. Outputs are written to `results/v2/<base>/<stage>/`.

Running time depends on the machine. On the machine used by the authors, the clean baseline of the ten models took about 44 min (M4), 166 min (Car Parts) and 34 min (Tourism), most of it in the automatic SARIMA.

## Pre-registration of the Tourism predictions

The predictions for Tourism (hypotheses H1 to H4) were committed before any model was run on that dataset, and the history of the repository is published so that the order can be verified:

| Commit | Content |
|---|---|
| `1a08fa1` | Predictions registered before the execution on Tourism (`results/v2/PREVISOES_tourism.md`, not edited afterwards) |
| `1e3d9f0` | Interpretation of the evaluation criteria, registered before the explainability runs on Tourism |
| `6b49e3d` | Evaluation code for H1 to H3, before the results |

## Citation

If you use this code, please cite the repository (see `CITATION.cff`) and the article.

## License

The code is released under the MIT License (see `LICENSE`). The datasets keep the licenses of their original sources.
