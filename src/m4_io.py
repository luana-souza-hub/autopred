"""Leitura eficiente dos CSVs M4 mensais (wide → long)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def list_series_ids(data_dir: Path) -> np.ndarray:
    """Lista IDs das séries lendo só a primeira coluna do train."""
    train_path = data_dir / 'Monthly-train.csv'
    id_col = pd.read_csv(train_path, nrows=0).columns[0]
    return pd.read_csv(train_path, usecols=[id_col])[id_col].to_numpy()


def sample_pilot_ids(data_dir: Path, n: int, seed: int) -> np.ndarray:
    ids = list_series_ids(data_dir)
    rng = np.random.default_rng(seed)
    return rng.choice(ids, size=n, replace=False)


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    id_col = df.columns[0]
    melted = df.melt(
        id_vars=id_col,
        value_vars=df.columns[1:],
        var_name='step_raw',
        value_name='y',
    )
    melted = melted.rename(columns={id_col: 'unique_id'})
    melted['step'] = melted['step_raw'].str.extract(r'(\d+)').astype(int)
    melted = melted.drop(columns='step_raw').dropna(subset=['y'])
    return melted.sort_values(['unique_id', 'step']).reset_index(drop=True)


def _read_wide_filtered(path: Path, series_ids: set, chunksize: int = 3000) -> pd.DataFrame:
    id_col = pd.read_csv(path, nrows=0).columns[0]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        sub = chunk[chunk[id_col].isin(series_ids)]
        if not sub.empty:
            parts.append(sub)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def load_m4_monthly(
    data_dir: Path,
    series_ids: Iterable | None = None,
    chunksize: int = 3000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega Monthly-train/test.csv em formato long.

    Se series_ids for informado, lê só essas séries (recomendado para piloto).
    """
    data_dir = Path(data_dir)
    train_path = data_dir / 'Monthly-train.csv'
    test_path = data_dir / 'Monthly-test.csv'

    if series_ids is None:
        train_raw = pd.read_csv(train_path)
        test_raw = pd.read_csv(test_path)
    else:
        ids_set = set(series_ids)
        train_raw = _read_wide_filtered(train_path, ids_set, chunksize)
        test_raw = _read_wide_filtered(test_path, ids_set, chunksize)

    return wide_to_long(train_raw), wide_to_long(test_raw)
