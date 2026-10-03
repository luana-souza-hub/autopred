#!/usr/bin/env python
"""Execução do pipeline v2 fora do Jupyter, com checkpoint e retomada.

Uso (na raiz do projeto):
  .venv\\Scripts\\python.exe scripts\\run_v2.py --base m4 --etapa limpo --modelos rapidos --n-jobs 6
  .venv\\Scripts\\python.exe scripts\\run_v2.py --base m4 --etapa limpo --modelos sarima_auto --n-jobs 6
  .venv\\Scripts\\python.exe scripts\\run_v2.py --base m4 --etapa C_outliers_2pct --modelos rapidos
  .venv\\Scripts\\python.exe scripts\\run_v2.py --base m4 --etapa C_outliers_2pct --alvo-ruidoso --modelos rapidos

Bases: m4 (200 séries do piloto, H=18), cp (Car Parts completo, H=18), tm (Tourism, H=24; --n-series).
Saída: results/v2/<sub>/<etapa>/<sub>_<etapa>_<modelo>_by_window.csv (+ .meta.json), com <sub> = <base> no
tamanho padrão da amostra e <base>_<n> em outro tamanho (ex.: tm_366).
Se interromper, rode de novo: retoma do arquivo .parcial.csv.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src import pipeline_v2 as pv  # noqa: E402

DATA = RAIZ / 'data' / 'raw'
ARQ = {
    'm4': DATA / 'Monthly-train.csv',
    'cp': DATA / 'car_parts' / 'car_parts_dataset_without_missing_values.tsf',
    'tm': DATA / 'tourism_monthly_dataset.tsf',
}
HORIZONTE = {'m4': 18, 'cp': 18, 'tm': 24}


def carregar_base(base, n_series=None):
    """Retorna (ids, séries, H). Amostras pela regra única (SEED=42, ordem do arquivo)."""
    if base == 'm4':
        ids, series = pv.load_m4_piloto(DATA, n=n_series or 200)
    elif base == 'cp':
        ids, series = pv.amostra_tsf(pv.load_tsf(ARQ['cp'], 'utf-8'), n_series)
    elif base == 'tm':
        ids, series = pv.amostra_tsf(pv.load_tsf(ARQ['tm'], 'latin-1'), n_series if n_series else 200)
    else:
        raise ValueError(base)
    return ids, series, HORIZONTE[base]


PADRAO_N = {'m4': 200, 'cp': None, 'tm': 200}


def subpasta(base, n_series):
    """Pasta (e prefixo) dos resultados: '<base>' no tamanho padrão da amostra; '<base>_<n>' em qualquer
    outro tamanho (ex.: tm_366), para nunca sobrescrever os resultados oficiais."""
    return base if n_series in (None, PADRAO_N[base]) else f'{base}_{n_series}'


def pasta_saida(base, etapa, alvo_ruidoso, min_train, n_series):
    nome = etapa + ('_alvo_ruidoso' if alvo_ruidoso else '')
    if min_train != pv.MIN_TRAIN:
        nome += f'_min_train{min_train}'
    return RAIZ / 'results' / 'v2' / subpasta(base, n_series) / nome, nome


def rodar_modelo(base, ids, series, h, modelo, etapa, alvo_ruidoso, n_jobs, min_train, n_series,
                 lote, force):
    pasta, nome_etapa = pasta_saida(base, etapa, alvo_ruidoso, min_train, n_series)
    pasta.mkdir(parents=True, exist_ok=True)
    final = pasta / f'{subpasta(base, n_series)}_{nome_etapa}_{modelo}_by_window.csv'
    parcial = final.with_name(final.name + '.parcial.csv')
    if final.exists() and not force:
        print(f'[{modelo}] cache completo: {final.relative_to(RAIZ)}', flush=True)
        return
    feitos = set()
    if parcial.exists() and not force:
        feitos = set(pd.read_csv(parcial, usecols=['serie_idx'])['serie_idx'].unique())
        print(f'[{modelo}] retomando: {len(feitos)}/{len(ids)} séries no parcial', flush=True)
    elif parcial.exists():
        parcial.unlink()
    cenario = None if etapa == 'limpo' else etapa
    pend = [i for i in range(len(ids)) if i not in feitos]
    t0 = time.time()
    from joblib import Parallel, delayed
    with Parallel(n_jobs=n_jobs) as par:
        for k in range(0, len(pend), lote):
            bloco = pend[k:k + lote]
            res = par(delayed(pv.avaliar_serie)(series[i], ids[i], i, [modelo], h, cenario, alvo_ruidoso,
                                                min_train) for i in bloco)
            df = pd.DataFrame([r for regs in res for r in regs])
            df.to_csv(parcial, mode='a', header=not parcial.exists(), index=False)
            feito = len(ids) - len(pend) + k + len(bloco)
            el = time.time() - t0
            eta = el / (k + len(bloco)) * (len(pend) - k - len(bloco)) / 60
            print(f'[{modelo}] {feito}/{len(ids)} séries | {el/60:.1f} min | ETA {eta:.1f} min', flush=True)
    df = pd.read_csv(parcial).sort_values(['serie_idx', 'janela']).reset_index(drop=True)
    df.to_csv(final, index=False)
    parcial.unlink()
    pv.write_meta(final, {'base': base, 'etapa': nome_etapa, 'modelo': modelo, 'H': h,
                          'min_train': min_train, 'n_series': len(ids), 'alvo_ruidoso': alvo_ruidoso,
                          'linhas': len(df), 'duracao_s': round(time.time() - t0, 1),
                          'ids_sha256': pv.hashlib.sha256('|'.join(map(str, ids)).encode()).hexdigest()},
                  arquivos_dados=[ARQ[base]])
    st = df['status'].value_counts().to_dict()
    print(f'[{modelo}] concluído em {(time.time()-t0)/60:.1f} min: {final.relative_to(RAIZ)} | {st}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, choices=['m4', 'cp', 'tm'])
    ap.add_argument('--etapa', default='limpo', help="'limpo' ou nome de cenário (ex.: A_ruido_30pct)")
    ap.add_argument('--modelos', default='rapidos', help="'rapidos', 'todos' ou lista separada por vírgula")
    ap.add_argument('--alvo-ruidoso', action='store_true', help='análise suplementar: perturba treino E teste')
    ap.add_argument('--n-series', type=int, default=None)
    ap.add_argument('--min-train', type=int, default=pv.MIN_TRAIN)
    ap.add_argument('--n-jobs', type=int, default=6)
    ap.add_argument('--lote', type=int, default=24, help='séries por checkpoint')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    if a.etapa != 'limpo' and a.etapa not in pv.CENARIOS:
        ap.error(f'etapa desconhecida: {a.etapa}')
    modelos = {'rapidos': pv.MODELOS_RAPIDOS, 'todos': list(pv.MODELOS)}.get(a.modelos, a.modelos.split(','))
    ids, series, h = carregar_base(a.base, a.n_series)
    print(f'base={a.base} etapa={a.etapa} alvo_ruidoso={a.alvo_ruidoso} séries={len(ids)} H={h} '
          f'min_train={a.min_train} modelos={modelos} n_jobs={a.n_jobs}', flush=True)
    for m in modelos:
        rodar_modelo(a.base, ids, series, h, m, a.etapa, a.alvo_ruidoso, a.n_jobs, a.min_train,
                     a.n_series, a.lote, a.force)


if __name__ == '__main__':
    main()
