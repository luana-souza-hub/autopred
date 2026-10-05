#!/usr/bin/env python
"""Gera as tabelas das lacunas L01, L03 e L04 (M4) a partir dos caches v2, sem rodar modelos.

L01: SARIMA heurístico nos 9 cenários do M4: Δ% e Δp.p. com fallback, sem fallback e sem explosivas,
     e taxa de fallback por cenário (mesmas regras de tm_adversos_sarima_variantes.csv, nb11b_v2.py).
L03: MAPE, MAE, MASE e posto (MAPE e MASE) de cada modelo por janela de origem (mediana entre séries),
     como desempenho_por_janela_tm.csv (nb11a_v2.py), com os 10 modelos.
L04: distribuição por série (mediana das janelas de cada série) de MAPE, MASE, MAE e R² por modelo:
     mínimo, quartis, máximo e unique_id dos extremos; versões oficial (com fallback), sem fallback e
     sem explosivas; número de séries com MAPE > 100%.

Saída: results/v2/pacote_escrita/00_controle/tabelas/L01_*.csv, L03_*.csv, L04_*.csv
Uso (na raiz do projeto):  .venv\\Scripts\\python.exe tools\\gerar_lacunas_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src import pipeline_v2 as pv  # noqa: E402

V2 = RAIZ / 'results' / 'v2'
SAIDA = V2 / 'pacote_escrita' / '00_controle' / 'tabelas'
NOMES = {'persistence': 'Persistência', 'mean': 'Média histórica', 'seasonal_naive': 'Sazonal ingênuo', 'ses': 'SES',
         'holt_winters': 'Holt-Winters', 'sarima_heuristic': 'SARIMA heurístico', 'sarima_auto': 'SARIMA automático',
         'random_forest': 'Random Forest', 'gbt': 'Gradient Boosting', 'svr': 'SVR'}
DESC = {c: d for c, (_, _, d) in pv.CENARIOS.items()}


def carregar(cond, modelos=None):
    arqs = [a for a in sorted((V2 / 'm4' / cond).glob(f'm4_{cond}_*_by_window.csv')) if '.parcial' not in a.name]
    d = pd.concat([pd.read_csv(a) for a in arqs], ignore_index=True)
    return d[d.model.isin(modelos)] if modelos else d


def l01():
    base = carregar('limpo', ['sarima_heuristic'])
    linhas = []
    for c in pv.CENARIOS:
        adv = carregar(c, ['sarima_heuristic'])
        lin = {'cenário': DESC[c]}
        for nome, inc_fb, exc_exp in [('com fallback', True, False), ('sem fallback', False, False), ('sem explosivas', True, True)]:
            t = pv.tabela_delta(pv.agregar_por_serie(base, inc_fb, exc_exp), pv.agregar_por_serie(adv, inc_fb, exc_exp))
            r = t.loc['sarima_heuristic']
            lin[f'Δ% {nome}'] = r['delta_pct']
            lin[f'Δp.p. {nome}'] = r['delta_pp']
            lin[f'n séries ({nome})'] = int(r['n_series'])
        lin['fallback no cenário (% jan.)'] = 100 * adv.status.isin([pv.FB_SNAIVE, pv.FB_MEAN]).mean()
        lin['fallback na linha de base (% jan.)'] = 100 * base.status.isin([pv.FB_SNAIVE, pv.FB_MEAN]).mean()
        lin['explosivas no cenário (n jan.)'] = int(pv.flag_explosiva(adv).sum())
        linhas.append(lin)
    return pd.DataFrame(linhas).set_index('cenário')


def m4_com_mase():
    ids, series = pv.load_m4_piloto(RAIZ / 'data' / 'raw', 200)
    return pv.adicionar_mase(carregar('limpo'), dict(zip(ids, series)))


def l03(d):
    pj = d.groupby(['model', 'janela']).agg(n_train=('n_train', 'median'), MAPE=('mape', 'median'),
                                            MAE=('mae', 'median'), MASE=('mase', 'median')).reset_index()
    pj['linhas ML'] = (pj['n_train'] - pv.historico_minimo()).where(pj.model.isin(['random_forest', 'gbt', 'svr']))
    pj['posto MAPE'] = pj.groupby('janela')['MAPE'].rank(method='min').astype(int)
    pj['posto MASE'] = pj.groupby('janela')['MASE'].rank(method='min').astype(int)
    pj['janela'] = pj['janela'].astype(int)
    pj['modelo'] = pj['model'].map(NOMES)
    return pj[['janela', 'n_train', 'MAPE', 'MAE', 'MASE', 'linhas ML', 'modelo', 'posto MAPE', 'posto MASE']] \
        .sort_values(['janela', 'posto MAPE']).reset_index(drop=True)


def l04(d):
    linhas = []
    for versao, inc_fb, exc_exp in [('oficial (com fallback)', True, False), ('sem fallback', False, False), ('sem explosivas', True, True)]:
        bs = pv.agregar_por_serie(d, inc_fb, exc_exp)
        for m, g in bs.groupby('model'):
            for met in ['mape', 'mase', 'mae', 'r2']:
                s = g.set_index('unique_id')[met].dropna()
                linhas.append({'versão': versao, 'modelo': NOMES[m], 'métrica': met.upper().replace('R2', 'R²'),
                               'n séries': len(s), 'mínimo': s.min(), 'unique_id do mínimo': s.idxmin(),
                               'Q1': s.quantile(0.25), 'mediana': s.median(), 'Q3': s.quantile(0.75),
                               'máximo': s.max(), 'unique_id do máximo': s.idxmax(),
                               'séries com valor > 100': int((s > 100).sum()) if met == 'mape' else None})
    return pd.DataFrame(linhas)


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    t1 = l01()
    t1.to_csv(SAIDA / 'L01_m4_adversos_sarima_variantes.csv')
    d = m4_com_mase()
    t3 = l03(d)
    t3.to_csv(SAIDA / 'L03_m4_desempenho_por_janela.csv', index=False)
    t4 = l04(d)
    t4.to_csv(SAIDA / 'L04_m4_distribuicao_por_serie.csv', index=False)
    pd.set_option('display.width', 250)
    print(t1.round(2).to_string()); print()
    print(t3.pivot(index='modelo', columns='janela', values=['posto MAPE', 'posto MASE']).to_string()); print()
    print(f'L04: {len(t4)} linhas -> {SAIDA.relative_to(RAIZ)}')


if __name__ == '__main__':
    main()
