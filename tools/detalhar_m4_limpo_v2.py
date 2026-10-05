#!/usr/bin/env python
"""Detalhes da linha de base limpa do M4 (200 séries) a partir dos caches v2 já existentes (sem rodar modelos).

1. Distribuição entre séries da mediana por série (janelas agregadas por pv.agregar_por_serie): quartis,
   mínimo e máximo de MAPE, MASE, MAE e R², com o unique_id da série no mínimo e no máximo do MAPE e do R².
   Versão oficial com fallback para os 10 modelos; SARIMA heurístico e SVR também sem fallback; SARIMA
   (automático e heurístico) também sem as janelas explosivas.
2. MAPE máximo do SARIMA automático e nº de séries com MAPE (mediana por série) acima de 100%, por modelo.
3. Tabela por série (unique_id x modelo) e Figura 9 refeita (MAPE truncado em 100%, MASE truncado em 6),
   com a legenda das famílias feita por handles, sem a barra fantasma em x = 0 que deixava espaço vazio.
4. MAPE, MAE, MASE e posto de cada modelo por janela de origem, no formato de desempenho_por_janela_tm.csv.
5. Proporção de janelas explosivas com os dois denominadores (todas as janelas e só as janelas OK).

Saídas em results/v2/pacote_escrita/03_m4_limpo/ (arquivos novos; nada existente é alterado).
Uso (na raiz do projeto):  .venv\\Scripts\\python.exe tools\\detalhar_m4_limpo_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src import pipeline_v2 as pv  # noqa: E402

V2 = RAIZ / 'results' / 'v2'
SAIDA = V2 / 'pacote_escrita' / '03_m4_limpo'
NOMES = {'holt_winters': 'Holt-Winters', 'persistence': 'Persistência', 'ses': 'SES', 'sarima_auto': 'SARIMA automático',
         'gbt': 'Gradient Boosting', 'sarima_heuristic': 'SARIMA heurístico', 'random_forest': 'Random Forest',
         'seasonal_naive': 'Sazonal ingênuo', 'svr': 'SVR', 'mean': 'Média histórica'}
CORES_FAM = {'Baseline': '#607D8B', 'Suavização exponencial': '#2196F3', 'ARIMA': '#FF9800', 'ML': '#9C27B0'}
METRICAS = ['mape', 'mase', 'mae', 'r2']
ROT = {'mape': 'MAPE', 'mase': 'MASE', 'mae': 'MAE', 'r2': 'R2'}

ids, series = pv.load_m4_piloto(RAIZ / 'data' / 'raw', 200)
jan = pd.concat([pd.read_csv(f) for f in sorted((V2 / 'm4' / 'limpo').glob('m4_limpo_*_by_window.csv'))], ignore_index=True)
jan = pv.adicionar_mase(jan, dict(zip(ids, series)))
assert jan.unique_id.nunique() == 200 and jan.model.nunique() == 10 and len(jan) == 6000
MODS = list(pv.MODELOS)
jan['explosiva'] = pv.flag_explosiva(jan)
jan['fallback'] = jan['status'].isin([pv.FB_SNAIVE, pv.FB_MEAN])

# ---- 1. distribuição entre séries
VERSOES = [('com fallback (oficial)', MODS, dict()),
           ('sem fallback', ['sarima_heuristic', 'svr'], dict(incluir_fallback=False)),
           ('sem explosivas', ['sarima_auto', 'sarima_heuristic'], dict(excluir_explosivas=True))]
linhas, por_serie_oficial = [], None
for versao, modelos, kw in VERSOES:
    bs = pv.agregar_por_serie(jan, **kw)
    if versao.startswith('com'):
        por_serie_oficial = bs
    for m in modelos:
        b = bs[bs.model == m].set_index('unique_id')
        lin = {'modelo': NOMES[m], 'família': pv.FAMILIA[m], 'versão': versao, 'n_series': len(b)}
        for met in METRICAS:
            v = b[met].dropna()
            q = v.quantile([0.25, 0.5, 0.75])
            lin.update({f'{ROT[met]} mín': v.min(), f'{ROT[met]} Q1': q[0.25], f'{ROT[met]} mediana': q[0.5],
                        f'{ROT[met]} Q3': q[0.75], f'{ROT[met]} máx': v.max()})
            if met in ('mape', 'r2'):
                lin[f'{ROT[met]} mín série'] = v.idxmin()
                lin[f'{ROT[met]} máx série'] = v.idxmax()
        lin['séries com MAPE > 100%'] = int((b['mape'] > 100).sum())
        linhas.append(lin)
dist = pd.DataFrame(linhas)
# conferência: a mediana oficial é a da Tabela 3 v2
t3 = pd.read_csv(V2 / 'm4' / 'limpo' / 'tabela3_v2_m4.csv', index_col=0)
of = dist[dist['versão'] == 'com fallback (oficial)'].set_index('modelo')
assert np.allclose(of['MAPE mediana'].round(2), t3.loc[of.index, 'MAPE (%)'], atol=0.006)
assert np.allclose(of['MAE mediana'].round(2), t3.loc[of.index, 'MAE'], atol=0.006)
cols_num = [c for c in dist.columns if dist[c].dtype.kind == 'f']
dist[cols_num] = dist[cols_num].round(4)
dist.to_csv(SAIDA / 'm4_limpo_distribuicao_por_serie.csv', index=False)

# ---- 2. MAPE máximo do SARIMA automático e séries acima de 100%
sa = of.loc['SARIMA automático']
sa_jan = jan[(jan.model == 'sarima_auto') & (jan.unique_id == sa['MAPE máx série'])]
acima = (dist[['modelo', 'versão', 'n_series', 'séries com MAPE > 100%']]
         .assign(**{'MAPE máx': dist['MAPE máx'], 'série do MAPE máx': dist['MAPE máx série']}))
acima.to_csv(SAIDA / 'm4_limpo_mape_acima_100.csv', index=False)

# ---- 3. tabela por série e Figura 9
ps = por_serie_oficial.copy()
cont = jan.groupby(['model', 'unique_id']).agg(n_janelas=('janela', 'size'), n_fallback=('fallback', 'sum'),
                                               n_explosivas=('explosiva', 'sum')).reset_index()
ps = ps.merge(cont, on=['model', 'unique_id'])
ps.insert(1, 'modelo', ps['model'].map(NOMES)); ps.insert(2, 'família', ps['model'].map(pv.FAMILIA))
ps = ps[['unique_id', 'modelo', 'família', 'mape', 'mase', 'mae', 'r2', 'n_janelas', 'n_fallback', 'n_explosivas']]
ps = ps.rename(columns={'mape': 'MAPE', 'mase': 'MASE', 'mae': 'MAE', 'r2': 'R2'}).sort_values(['modelo', 'unique_id'])
ps.to_csv(SAIDA / 'm4_limpo_por_serie.csv', index=False)

ordem, _ = pv.bootstrap_ranks(por_serie_oficial, MODS, 'mape', n=2000)
ordem = list(ordem.sort_values('rank').index)
fig, axes = plt.subplots(2, 1, figsize=(15, 10))
for ax, met, clip, fmt, rot in [(axes[0], 'mape', 100, '{:.2f}', 'MAPE (%) por série, truncado em 100'),
                                (axes[1], 'mase', 6, '{:.3f}', 'MASE por série, truncado em 6')]:
    dados = [por_serie_oficial[por_serie_oficial.model == m][met].dropna() for m in ordem]
    bp = ax.boxplot([d.clip(upper=clip) for d in dados], patch_artist=True, medianprops=dict(color='black', linewidth=2),
                    flierprops=dict(marker='o', markersize=2.5, alpha=0.35))
    for patch, m in zip(bp['boxes'], ordem):
        patch.set_facecolor(CORES_FAM[pv.FAMILIA[m]]); patch.set_alpha(0.75)
    for i, d in enumerate(dados):
        ax.text(i + 1, d.median(), fmt.format(d.median()), ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(range(1, len(ordem) + 1)); ax.set_xticklabels([NOMES[m] for m in ordem], rotation=30, ha='right', fontsize=8)
    ax.set_xlim(0.5, len(ordem) + 0.5)
    ax.set_ylabel(rot); ax.grid(axis='y', linestyle='--', alpha=0.4)
axes[0].legend(handles=[Patch(facecolor=c, alpha=0.75, label=f) for f, c in CORES_FAM.items()],
               fontsize=8, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False)
fig.suptitle('M4, v2: MAPE e MASE dos 10 modelos (mediana por série; ordenados pelo MAPE mediano)', fontsize=11)
fig.savefig(SAIDA / 'fig09_v2_m4_10_modelos.png', dpi=140, bbox_inches='tight'); plt.close(fig)

# ---- 4. desempenho por janela de origem (formato de desempenho_por_janela_tm.csv, com os postos)
pj = jan.groupby(['model', 'janela']).agg(n_train=('n_train', 'median'), MAPE=('mape', 'median'),
                                          MAE=('mae', 'median'), MASE=('mase', 'median')).reset_index()
pj['linhas ML'] = (pj['n_train'] - pv.historico_minimo()).where(pj.model.isin(['random_forest', 'gbt', 'svr']))
pj['posto MAPE'] = pj.groupby('janela')['MAPE'].rank(method='min').astype(int)
pj['posto MASE'] = pj.groupby('janela')['MASE'].rank(method='min').astype(int)
pj['modelo'] = pj['model'].map(NOMES)
pj = pj.drop(columns='model')[['janela', 'n_train', 'MAPE', 'MAE', 'MASE', 'linhas ML', 'modelo', 'posto MAPE', 'posto MASE']]
pj = pj.sort_values(['modelo', 'janela'])
pj.to_csv(SAIDA / 'desempenho_por_janela_m4.csv', index=False)

# ---- 5. denominador da proporção de janelas explosivas
ex = jan.groupby('model').agg(janelas=('janela', 'size'), janelas_OK=('status', lambda s: int((s == pv.OK).sum())),
                              fallback=('fallback', 'sum'), explosivas=('explosiva', 'sum'))
ex['explosivas em janelas OK'] = jan[jan.status == pv.OK].groupby('model')['explosiva'].sum()
ex['% sobre todas as janelas (usado na Tabela 3 v2)'] = 100 * ex['explosivas'] / ex['janelas']
ex['% sobre as janelas OK'] = 100 * ex['explosivas em janelas OK'] / ex['janelas_OK']
assert np.allclose(ex['% sobre todas as janelas (usado na Tabela 3 v2)'].round(1).values,
                   t3.loc[[NOMES[m] for m in ex.index], 'Explosivas (% jan.)'].values)
ex.index = [NOMES[m] for m in ex.index]
ex.index.name = 'modelo'
ex.round(3).to_csv(SAIDA / 'm4_limpo_explosivas_denominador.csv')

# ---- resumo no console
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print('1. Distribuição entre séries (mediana por série):')
for met in METRICAS:
    cols = ['modelo', 'versão'] + [f'{ROT[met]} {k}' for k in ('mín', 'Q1', 'mediana', 'Q3', 'máx')]
    if met in ('mape', 'r2'):
        cols += [f'{ROT[met]} mín série', f'{ROT[met]} máx série']
    print(f'--- {ROT[met]}'); print(dist[cols].round(3).to_string(index=False))
print(f"\n2. SARIMA automático: MAPE máximo (mediana por série) {sa['MAPE máx']:.2f}% na série {sa['MAPE máx série']}; "
      f"MAPE por janela nessa série: {sa_jan.sort_values('janela')['mape'].round(2).tolist()} "
      f"(explosivas: {sa_jan.sort_values('janela')['explosiva'].tolist()})")
print(acima.to_string(index=False))
print('\n4. Desempenho por janela (posto entre os 10 modelos):')
print(pj.pivot(index='modelo', columns='janela', values=['posto MAPE', 'posto MASE']).to_string())
print('\n5. Explosivas:'); print(ex.round(3).to_string())
print('\nSaídas em', SAIDA.relative_to(RAIZ))
