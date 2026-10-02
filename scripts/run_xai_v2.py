#!/usr/bin/env python
"""Explicabilidade v2 (SHAP e LIME nos MESMOS pontos), fora do Jupyter.

Uso (na raiz do projeto):
  .venv\\Scripts\\python.exe scripts\\run_xai_v2.py --base m4 --condicao limpo
  .venv\\Scripts\\python.exe scripts\\run_xai_v2.py --base m4 --condicao ruido30
  .venv\\Scripts\\python.exe scripts\\run_xai_v2.py --base m4 --condicao limpo --metodos lime --lime-samples 1000
  .venv\\Scripts\\python.exe scripts\\run_xai_v2.py --base m4 --condicao limpo --lag24      (suplementar)

Subconjunto: 30 séries sorteadas com SEED+1 sobre as posições da amostra da base (regra do NB09).
Modelo: RF, GB e SVR treinados na última janela (série[:-H]); ruído σ=30% só no treino.
Pontos explicados: as 29 linhas de treino mais recentes + o ponto de previsão (iguais para SHAP e LIME).
Saída: results/v2/<base>/xai/xai_<base>_<condicao>[_lime<N>][_lag24]_pontos.csv (+ .meta.json).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / 'scripts'))
from src import pipeline_v2 as pv  # noqa: E402
from run_v2 import ARQ, carregar_base  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, choices=['m4', 'cp', 'tm'])
    ap.add_argument('--condicao', required=True, choices=['limpo', 'ruido30'])
    ap.add_argument('--metodos', default='shap,lime')
    ap.add_argument('--lime-samples', type=int, default=pv.LIME_NUM_SAMPLES)
    ap.add_argument('--lag24', action='store_true', help='análise suplementar com lag_24')
    ap.add_argument('--n-series', type=int, default=None, help='tamanho da amostra-base antes do sorteio das 30')
    ap.add_argument('--n-expl', type=int, default=pv.N_EXPL)
    ap.add_argument('--n-jobs', type=int, default=10)
    ap.add_argument('--elegiveis', action='store_true',
                    help='sorteia só entre séries com dados suficientes (>= 10 linhas e alvo não constante)')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    ids, series, h = carregar_base(a.base, a.n_series)
    linhas_treino = [len(pv.build_features(s[:-h])[0]) for s in series]
    alvo_var = [np.std(pv.build_features(s[:-h])[1]) > 0 if n else False for s, n in zip(series, linhas_treino)]
    if a.elegiveis:
        candidatos = [i for i in range(len(ids)) if linhas_treino[i] >= pv.MIN_LINHAS_XAI and alvo_var[i]]
    else:
        candidatos = list(range(len(ids)))
    pos = np.array(candidatos)[np.random.default_rng(pv.SEED + 1).choice(len(candidatos), size=a.n_expl, replace=False)]
    elig = {'filtro_elegiveis': a.elegiveis, 'n_series_base': len(ids), 'n_elegiveis': len(candidatos),
            'n_descartadas': len(ids) - len(candidatos),
            'descartadas_poucas_linhas': int(sum(n < pv.MIN_LINHAS_XAI for n in linhas_treino)),
            'descartadas_alvo_constante': int(sum((n >= pv.MIN_LINHAS_XAI) and not v for n, v in zip(linhas_treino, alvo_var))),
            'linhas_treino_sorteadas': sorted({int(linhas_treino[i]) for i in pos})}
    print(elig, flush=True)
    metodos = tuple(a.metodos.split(','))
    nome = f'xai_{a.base}_{a.condicao}'
    if a.lime_samples != pv.LIME_NUM_SAMPLES:
        nome += f'_lime{a.lime_samples}'
    if a.lag24:
        nome += '_lag24'
    pasta = RAIZ / 'results' / 'v2' / a.base / 'xai'
    pasta.mkdir(parents=True, exist_ok=True)
    arq = pasta / f'{nome}_pontos.csv'
    if arq.exists() and not a.force:
        print(f'cache: {arq.relative_to(RAIZ)}')
        return
    print(f'{nome}: {len(pos)} séries | métodos={metodos} | lime_samples={a.lime_samples} | lag24={a.lag24}', flush=True)
    t0 = time.time()
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=a.n_jobs)(
        delayed(pv.explicar_serie)(series[i], ids[i], int(i), h, a.condicao, a.lag24, metodos, a.lime_samples)
        for i in pos)
    df = pd.DataFrame([r for regs in res for r in regs])
    df.to_csv(arq, index=False)
    pv.write_meta(arq, {'base': a.base, 'condicao': a.condicao, 'metodos': list(metodos),
                        'lime_samples': a.lime_samples, 'com_lag24': a.lag24, 'H': h,
                        'n_series_amostra': len(ids), 'posicoes_expl': [int(i) for i in pos], **elig,
                        'pontos': f'{pv.N_PONTOS_XAI - 1} linhas de treino mais recentes + ponto de previsão',
                        'shap_svr_nsamples': '2*p + 2048', 'linhas': len(df),
                        'duracao_s': round(time.time() - t0, 1)},
                  arquivos_dados=[ARQ[a.base]])
    print(f'{nome}: {df.serie.nunique()} séries, {len(df)} linhas em {(time.time()-t0)/60:.1f} min -> {arq.relative_to(RAIZ)}',
          flush=True)


if __name__ == '__main__':
    main()
