"""Pipeline Car Parts — compartilhado pelo notebook 10 e scripts/run_car_parts_ml.py."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# ── Protocolo (igual notebooks 02–07 / nb 10) ─────────────────────────────────
SEED = 42
H = 18
MIN_TRAIN = 24
N_ORIGINS = 3
SEASON = 12
N_LAGS = 12
N_ESTIMATORS = 30
MIN_SAMPLES_LEAF = 2
LEARNING_RATE = 0.1
GBT_MAX_DEPTH = 3
GBT_SUBSAMPLE = 0.8
SVR_C, SVR_EPS, SVR_KERNEL, SVR_GAMMA = 1.0, 0.1, "rbf", "scale"

ML_MODEL_NAMES = ("random_forest", "gbt", "svr")


def find_project_root() -> Path:
    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "results").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    return start


def result_dir(root: Path | None = None) -> Path:
    d = (root or find_project_root()) / "results" / "car_parts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tsf_path(root: Path | None = None) -> Path:
    root = root or find_project_root()
    p = root / "data" / "raw" / "car_parts" / "car_parts_dataset_without_missing_values.tsf"
    if not p.is_file():
        alt = root / "data" / "raw" / "car_parts_dataset_without_missing_values.tsf"
        if alt.is_file():
            return alt
        raise FileNotFoundError(f"TSF não encontrado: {p}")
    return p


def load_car_parts_tsf(path: Path) -> pd.DataFrame:
    records: list[dict] = []
    in_data = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower() == "@data":
                in_data = True
                continue
            if line.startswith("@") or not in_data:
                continue
            parts = line.split(":")
            if len(parts) < 3:
                continue
            name = parts[0].strip()
            try:
                values = [float(v) for v in parts[-1].split(",") if v.strip()]
            except ValueError:
                continue
            for step, val in enumerate(values, start=1):
                records.append({"unique_id": name, "step": step, "y": val})
    return pd.DataFrame(records).sort_values(["unique_id", "step"]).reset_index(drop=True)


def load_train_df(root: Path | None = None) -> pd.DataFrame:
    df = load_car_parts_tsf(tsf_path(root))
    lengths = df.groupby("unique_id")["y"].count()
    valid = lengths[lengths >= MIN_TRAIN + H].index
    return df[df["unique_id"].isin(valid)].copy()


def make_rolling_windows(series, horizon, min_train, n_origins):
    n = len(series)
    if n < min_train + horizon:
        return []
    last_origin = n - horizon
    first_origin = min_train
    if last_origin <= first_origin:
        return []
    origins = sorted(set(np.linspace(first_origin, last_origin, n_origins, dtype=int).tolist()))
    return [
        {"origin": o, "train": series[:o], "test": series[o : o + horizon]}
        for o in origins
        if len(series[o : o + horizon]) == horizon
    ]


def calc_mape(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def calc_metrics(y_true, y_pred):
    return {
        "mape": calc_mape(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_features(series, n_lags=N_LAGS, season=SEASON):
    n = len(series)
    include_lag2 = n > 2 * season
    start = max(n_lags, season, 2 * season if include_lag2 else 0)
    n_feat = n_lags + 3 + int(include_lag2)
    if n <= start:
        return np.empty((0, n_feat)), np.empty(0), []

    lags = np.column_stack([series[start - i : n - i] for i in range(1, n_lags + 1)])
    roll_mean = lags.mean(axis=1)
    roll_std = np.std(lags, axis=1, ddof=1)
    lag_s = series[start - season : n - season]
    cols = [f"lag_{i}" for i in range(1, n_lags + 1)] + ["roll_mean", "roll_std", "lag_season"]
    parts = [lags, roll_mean.reshape(-1, 1), roll_std.reshape(-1, 1), lag_s.reshape(-1, 1)]
    if include_lag2:
        lag_s2 = series[start - 2 * season : n - 2 * season]
        parts.append(lag_s2.reshape(-1, 1))
        cols.append("lag_season2")
    X = np.column_stack(parts)
    y = series[start:]
    return X, y, cols


def predict_recursive(model, train, horizon, n_lags=N_LAGS, season=SEASON, feature_cols=None):
    if feature_cols is None:
        _, _, feature_cols = build_features(train, n_lags, season)
    use_lag2 = "lag_season2" in feature_cols
    history = list(train.copy())
    preds = []
    for _ in range(horizon):
        n = len(history)
        lags = [history[n - i] for i in range(1, n_lags + 1)]
        roll_mean = float(np.mean(lags))
        roll_std = float(np.std(lags, ddof=1)) if len(lags) > 1 else 0.0
        lag_s = history[n - season] if n >= season else history[0]
        feats = lags + [roll_mean, roll_std, lag_s]
        if use_lag2:
            feats.append(history[n - 2 * season] if n >= 2 * season else history[0])
        x = np.nan_to_num(np.array(feats, dtype=float).reshape(1, -1), nan=0.0)
        pred = model.predict(x)[0]
        preds.append(pred)
        history.append(pred)
    return np.array(preds)


def model_random_forest(train, horizon, n_lags=N_LAGS, season=SEASON):
    min_needed = max(n_lags, season) + 1
    if len(train) < min_needed:
        return np.full(horizon, np.mean(train))
    X, y, feat_cols = build_features(train, n_lags, season)
    if len(X) < 2:
        return np.full(horizon, np.mean(train))
    rf = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=SEED,
        n_jobs=1,
    )
    rf.fit(X, y)
    return predict_recursive(rf, train, horizon, n_lags, season, feature_cols=feat_cols)


def model_gbt(train, horizon, n_lags=N_LAGS, season=SEASON):
    min_needed = max(n_lags, season) + 1
    if len(train) < min_needed:
        return np.full(horizon, np.mean(train))
    X, y, feat_cols = build_features(train, n_lags, season)
    if len(X) < 2:
        return np.full(horizon, np.mean(train))
    gbt = GradientBoostingRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=GBT_MAX_DEPTH,
        subsample=GBT_SUBSAMPLE,
        random_state=SEED,
    )
    gbt.fit(X, y)
    return predict_recursive(gbt, train, horizon, n_lags, season, feature_cols=feat_cols)


def _predict_recursive_svr(pipeline, scaler_y, train, horizon, n_lags, season, feature_cols):
    use_lag2 = "lag_season2" in feature_cols
    history = list(train.copy())
    preds = []
    for _ in range(horizon):
        n = len(history)
        lags = [history[n - i] for i in range(1, n_lags + 1)]
        roll_mean = float(np.mean(lags))
        roll_std = float(np.std(lags, ddof=1)) if len(lags) > 1 else 0.0
        lag_s = history[n - season] if n >= season else history[0]
        feats = lags + [roll_mean, roll_std, lag_s]
        if use_lag2:
            feats.append(history[n - 2 * season] if n >= 2 * season else history[0])
        x = np.array(feats, dtype=float).reshape(1, -1)
        pred = scaler_y.inverse_transform([[pipeline.predict(x)[0]]])[0, 0]
        preds.append(pred)
        history.append(pred)
    return np.array(preds)


def model_svr(train, horizon, n_lags=N_LAGS, season=SEASON):
    min_needed = max(n_lags, season) + 1
    if len(train) < min_needed:
        return np.full(horizon, np.mean(train))
    X, y, feat_cols = build_features(train, n_lags, season)
    if len(X) < 2 or np.std(y) == 0:
        return np.full(horizon, np.mean(train))
    scaler_y = StandardScaler()
    y_norm = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(C=SVR_C, epsilon=SVR_EPS, kernel=SVR_KERNEL, gamma=SVR_GAMMA)),
    ])
    pipe.fit(X, y_norm)
    return _predict_recursive_svr(pipe, scaler_y, train, horizon, n_lags, season, feat_cols)


ML_MODELS = {
    "random_forest": model_random_forest,
    "gbt": model_gbt,
    "svr": model_svr,
}


def run_pipeline(
    train_df: pd.DataFrame,
    model_name: str,
    predict_fn,
    *,
    horizon=H,
    min_train=MIN_TRAIN,
    n_origins=N_ORIGINS,
    cache_file: Path | None = None,
    force: bool = False,
    checkpoint_every: int = 50,
    print_every: int = 50,
) -> pd.DataFrame:
    series_dict = {
        uid: grp["y"].values.astype(float)
        for uid, grp in train_df.groupby("unique_id", sort=False)
    }
    ids = list(series_dict.keys())
    total = len(ids)
    records: list[dict] = []
    done: set[str] = set()

    if cache_file and cache_file.is_file() and not force:
        partial = pd.read_csv(cache_file)
        records = partial.to_dict("records")
        done = set(partial["unique_id"].dropna().unique())
        print(f"Retomando {model_name}: {len(done)}/{total} séries no cache parcial", flush=True)

    t0 = time.time()
    processed = 0
    for i, uid in enumerate(ids):
        if uid in done:
            continue
        if processed == 0 or (processed + 1) % print_every == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 and processed else 0
            eta = (total - len(done) - processed) / rate / 60 if rate else 0
            print(f"  [{len(done)+processed+1}/{total}] {uid}  (ETA ~{eta:.0f} min)", flush=True)

        values = series_dict[uid]
        windows = make_rolling_windows(values, horizon, min_train, n_origins)
        if not windows:
            records.append({
                "unique_id": uid, "model": model_name,
                "origin": None, "n_train": None,
                "mape": None, "mae": None, "r2": None,
                "exec_time_s": None, "status": "SKIP_SHORT",
            })
        else:
            for win in windows:
                t1 = time.perf_counter()
                try:
                    pred = predict_fn(win["train"], horizon)
                    metrics = calc_metrics(win["test"], pred)
                    status = "OK"
                except Exception as e:
                    metrics = {"mape": None, "mae": None, "r2": None}
                    status = f"ERROR: {str(e)[:80]}"
                records.append({
                    "unique_id": uid,
                    "model": model_name,
                    "origin": win["origin"],
                    "n_train": len(win["train"]),
                    **metrics,
                    "exec_time_s": time.perf_counter() - t1,
                    "status": status,
                })

        processed += 1
        if cache_file and processed % checkpoint_every == 0:
            pd.DataFrame(records).to_csv(cache_file, index=False)
            print(f"  checkpoint → {cache_file.name} ({len(records):,} linhas)", flush=True)

    df = pd.DataFrame(records)
    if cache_file:
        df.to_csv(cache_file, index=False)
    ok = (df["status"] == "OK").sum()
    print(f"Concluído {model_name}: {ok:,} OK | {len(df):,} linhas | {(time.time()-t0)/60:.1f} min", flush=True)
    return df


def cache_path(model_name: str, root: Path | None = None) -> Path:
    return result_dir(root) / f"results_cp_{model_name}_by_window.csv"
