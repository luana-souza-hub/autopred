#!/usr/bin/env python
"""Conta, nos caches v2 já existentes (sem rodar modelos), limites de tempo e motivos de fallback do SARIMA.

SARIMA automático: busca incompleta = info.busca_completa == false (limite de 300 s por janela,
verificado antes de cada candidato). SARIMA heurístico: estouro = info.motivo == "tempo" (ajuste > 60 s);
exec_time_s > 60 também é contado como conferência.

Motivos de fallback lidos de info.motivo. Exceção: no SARIMA automático o fallback por treino constante
retorna sem info (pipeline_v2.py, model_sarima_auto), então FALLBACK_SNAIVE sem motivo e sem
busca_completa é classificado como treino constante; o script confere que essas janelas coincidem com
as marcadas "treino_constante" pelo SARIMA heurístico na mesma base e condição.

Saída: results/v2/sarima_limites_fallback_v2.csv (copiada para results/v2/pacote_escrita/01_documentos/).
Uso (na raiz do projeto):  .venv\\Scripts\\python.exe tools\\contar_limites_sarima_v2.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src import pipeline_v2 as pv  # noqa: E402

V2 = RAIZ / 'results' / 'v2'
BASES = {'m4': 'M4', 'cp': 'Car Parts', 'tm': 'Tourism 200', 'tm_366': 'Tourism 366'}
MODELOS = {'sarima_auto': 'SARIMA automático', 'sarima_heuristic': 'SARIMA heurístico'}
MOTIVOS = ['treino_constante', 'sem_candidato_valido', 'excecao', 'verossimilhanca_degenerada', 'tempo',
           'previsao_invalida']

frames = []
for sub, nome_base in BASES.items():
    for modelo in MODELOS:
        for arq in sorted((V2 / sub).glob(f'*/{sub}_*_{modelo}_by_window.csv')):
            cond = arq.parent.name
            d = pd.read_csv(arq)
            info = d['info'].fillna('').map(lambda s: json.loads(s) if s else {})
            motivo = info.map(lambda x: x.get('motivo', ''))
            if modelo == 'sarima_auto':
                sem_info = (d.status == pv.FB_SNAIVE) & (motivo == '') & info.map(lambda x: 'busca_completa' not in x)
                motivo = motivo.mask(sem_info, 'treino_constante')
            d = d.assign(base=nome_base, pasta=sub, condicao=cond,
                         suplementar=cond.endswith('_alvo_ruidoso'), modelo=MODELOS[modelo],
                         motivo=motivo,
                         busca_incompleta=info.map(lambda x: x.get('busca_completa') is False),
                         sel_s=info.map(lambda x: x.get('sel_s', np.nan)),
                         explosiva=info.map(lambda x: bool(x.get('explosiva'))),
                         arquivo=str(arq.relative_to(V2)).replace('\\', '/'))
            frames.append(d)
df = pd.concat(frames, ignore_index=True)

# Conferência: treino constante inferido no automático = marcado no heurístico (mesma base, condição, série, janela)
chave = ['pasta', 'condicao', 'unique_id', 'janela']
tc = {m: set(map(tuple, df[(df.modelo == MODELOS[m]) & (df.motivo == 'treino_constante')][chave].values)) for m in MODELOS}
auto_conds = set(df[df.modelo == MODELOS['sarima_auto']][['pasta', 'condicao']].itertuples(index=False, name=None))
heur_tc = {k for k in tc['sarima_heuristic'] if (k[0], k[1]) in auto_conds}
assert tc['sarima_auto'] == heur_tc, 'treino constante inferido no automático difere do heurístico'


def resumir(g):
    out = {'n_janelas': len(g), 'ok': int((g.status == pv.OK).sum()),
           'busca_incompleta_300s': int(g.busca_incompleta.sum()) if g.model.iloc[0] == 'sarima_auto' else np.nan,
           'estouro_60s': int((g.motivo == 'tempo').sum()) if g.model.iloc[0] == 'sarima_heuristic' else np.nan,
           'exec_time_s_acima_60': int((g.exec_time_s > pv.SARIMA_TIME_LIMIT_S).sum())
           if g.model.iloc[0] == 'sarima_heuristic' else np.nan}
    for m in MOTIVOS:
        out[f'fallback_{m}'] = int(((g.status == pv.FB_SNAIVE) & (g.motivo == m)).sum())
    out['fallback_total'] = int(g.status.isin([pv.FB_SNAIVE, pv.FB_MEAN]).sum())
    out['status_erro'] = int((g.status == pv.ERRO).sum())
    out['sem_janela'] = int((g.status == pv.SEM_JANELA).sum())
    out['explosivas'] = int(g.explosiva.sum())
    out['max_exec_time_s'] = round(float(g.exec_time_s.max()), 2)
    out['max_selecao_s'] = round(float(g.sel_s.max()), 2) if g.sel_s.notna().any() else np.nan
    soma = sum(out[f'fallback_{m}'] for m in MOTIVOS)
    assert soma == out['fallback_total'], (g.name, soma, out['fallback_total'])
    return pd.Series(out)


grupos = ['base', 'pasta', 'condicao', 'suplementar', 'modelo']
df['janela_txt'] = df.janela.map(lambda j: 'sem janela' if pd.isna(j) else str(int(j)))
por_janela = df.groupby(grupos + ['janela_txt'], sort=False).apply(resumir, include_groups=False).reset_index()
total = df.groupby(grupos, sort=False).apply(resumir, include_groups=False).reset_index().assign(janela_txt='todas')
tab = pd.concat([por_janela, total]).rename(columns={'janela_txt': 'janela'})
ordem_base = {b: i for i, b in enumerate(BASES.values())}
ordem_cond = {c: (0 if c == 'limpo' else 1) for c in tab.condicao}
tab = tab.sort_values(['base', 'modelo', 'suplementar', 'condicao', 'janela'],
                      key=lambda s: s.map(ordem_base) if s.name == 'base'
                      else s.map(ordem_cond).astype(str) + s if s.name == 'condicao' else s).reset_index(drop=True)
arqs = df.groupby(grupos, sort=False).arquivo.first().reset_index()
tab = tab.merge(arqs, on=grupos)
contagens = [c for c in tab.columns if c not in grupos + ['janela', 'arquivo', 'max_exec_time_s', 'max_selecao_s']]
tab[contagens] = tab[contagens].astype('Int64')

saida = V2 / 'sarima_limites_fallback_v2.csv'
tab.to_csv(saida, index=False, encoding='utf-8')
shutil.copy2(saida, V2 / 'pacote_escrita' / '01_documentos' / saida.name)
print(f'{len(tab)} linhas -> {saida.relative_to(RAIZ)} (+ cópia no pacote)')
print(tab[tab.janela == 'todas'][['base', 'condicao', 'modelo', 'n_janelas', 'busca_incompleta_300s', 'estouro_60s',
                                  'fallback_total'] + [f'fallback_{m}' for m in MOTIVOS] + ['status_erro']]
      .to_string(index=False))
