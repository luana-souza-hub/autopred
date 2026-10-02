#!/usr/bin/env python
"""
Roda modelos ML pesados do Car Parts FORA do Jupyter (não trava o Cursor).

Uso (na raiz do projeto autopred):
  .venv\\Scripts\\python.exe scripts\\run_car_parts_ml.py random_forest
  .venv\\Scripts\\python.exe scripts\\run_car_parts_ml.py gbt
  .venv\\Scripts\\python.exe scripts\\run_car_parts_ml.py svr
  .venv\\Scripts\\python.exe scripts\\run_car_parts_ml.py all

Se interromper (Ctrl+C), rode de novo — retoma do checkpoint parcial.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.car_parts_pipeline import (  # noqa: E402
    ML_MODELS,
    ML_MODEL_NAMES,
    cache_path,
    load_train_df,
    run_pipeline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Car Parts — modelos ML (RF, GBT, SVR)")
    parser.add_argument(
        "model",
        choices=[*ML_MODEL_NAMES, "all"],
        help="Modelo a executar ou 'all' para os três ML",
    )
    parser.add_argument("--force", action="store_true", help="Ignora cache e recomeça do zero")
    args = parser.parse_args()

    names = list(ML_MODEL_NAMES) if args.model == "all" else [args.model]
    print(f"Carregando train_df...", flush=True)
    train_df = load_train_df(ROOT)
    print(f"  {train_df['unique_id'].nunique():,} séries | {len(train_df):,} linhas\n", flush=True)

    for name in names:
        out = cache_path(name, ROOT)
        if out.is_file() and not args.force:
            import pandas as pd
            prev = pd.read_csv(out)
            n_series = prev["unique_id"].nunique()
            if n_series >= train_df["unique_id"].nunique():
                print(f"{name}: cache completo ({len(prev):,} linhas) — use --force para refazer\n", flush=True)
                continue
        print(f"=== {name} ===", flush=True)
        run_pipeline(
            train_df,
            name,
            ML_MODELS[name],
            cache_file=out,
            force=args.force,
            checkpoint_every=50,
            print_every=50,
        )
        print(f"Salvo: {out}\n", flush=True)

    print("Pronto. Volte ao notebook 10 e rode a célula ## 8 (só carrega cache).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
