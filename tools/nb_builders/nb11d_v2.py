import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nbtools import md, code, check

NB = Path(r'C:\Users\Usuario\Desktop\Mestrado\Mestrado - aula\dissertação\autopred\notebooks\11d_sensibilidade_366_tourism_v2.ipynb')
cells = []

cells.append(md(r'''
# 11d v2: Tourism Monthly, sensibilidade com as 366 séries (Fase 4d)

**Análise de SENSIBILIDADE.** Repete a linha de base limpa (10 modelos, inclusive o SARIMA automático), os 9 cenários adversos e a explicabilidade (condições oficiais limpo e ruído σ=30%) com **todas as 366 séries** do Tourism Monthly, em vez da amostra de 200 (SEED=42). A avaliação oficial das previsões continua sendo a das 200 séries (`tm/replicacao_previsoes_tm.csv`); nada aqui altera `PREVISOES_tourism.md` nem os rótulos já fixados.

- Resultados: `results/v2/tm_366/` (pasta própria; `results/v2/tm/` não é tocado).
- As 200 séries da amostra oficial estão contidas nas 366, então as duas análises **não são independentes**: os IC das correlações entre os rankings abaixo são aproximados (reamostragem independente das duas bases).
- XAI: 30 séries sorteadas com SEED+1 entre as 366 (outras séries que as da Fase 4c).

Itens: (a) Spearman entre o ranking das 200 e o das 366 (MASE e MAPE); (b) mudanças de posição; (c) condições de H1 a H3 nas 366, como sensibilidade; (d) Spearman do escore de robustez entre 200 e 366.
'''))

cells.append(code(r'''
import sys, json, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_root = Path.cwd().resolve()
PROJECT_ROOT = next(c for c in (_root, *_root.parents) if (c / 'src' / 'pipeline_v2.py').is_file())
sys.path.insert(0, str(PROJECT_ROOT))
from src import pipeline_v2 as pv

V2 = PROJECT_ROOT / 'results' / 'v2'
DATA = PROJECT_ROOT / 'data' / 'raw'
DIR = V2 / 'tm_366' / 'sensibilidade'; DIR.mkdir(parents=True, exist_ok=True)
NOMES = {'holt_winters': 'Holt-Winters', 'persistence': 'Persistência', 'ses': 'SES', 'sarima_auto': 'SARIMA automático',
         'gbt': 'Gradient Boosting', 'sarima_heuristic': 'SARIMA heurístico', 'random_forest': 'Random Forest',
         'seasonal_naive': 'Sazonal ingênuo', 'svr': 'SVR', 'mean': 'Média histórica'}
pd.set_option('display.width', 260)

todas = pv.load_tsf(DATA / 'tourism_monthly_dataset.tsf', 'latin-1')
ids200, _ = pv.amostra_tsf(todas, 200)

def carregar(sub, pasta, modelos=None):
    d = pd.concat([pd.read_csv(f) for f in sorted((V2 / sub / pasta).glob('*_by_window.csv'))], ignore_index=True)
    return d if modelos is None else d[d.model.isin(modelos)]

t200 = pv.adicionar_mase(carregar('tm', 'limpo'), todas)
t366 = pv.adicionar_mase(carregar('tm_366', 'limpo'), todas)
MODELOS = [m for m in pv.MODELOS if m in t366.model.unique()]
assert set(ids200) <= set(t366.unique_id) and t366.unique_id.nunique() == 366
metas = {json.loads(f.read_text(encoding='utf-8'))['pipeline_sha256'][:8] for f in (V2 / 'tm_366').rglob('*.meta.json')}
print(f'366 séries: {len(t366):,} janelas, {len(MODELOS)} modelos | 200 séries contidas nas 366: True | hashes do pipeline nos caches: {metas}')
'''))

cells.append(md(r'''
## 1. Status, fallback e previsões explosivas (366 séries)
'''))

cells.append(code(r'''
st = pd.DataFrame({'fallback 200 (%)': 100 * pv.taxas_status(t200)[[pv.FB_SNAIVE, pv.FB_MEAN]].sum(axis=1),
                   'fallback 366 (%)': 100 * pv.taxas_status(t366)[[pv.FB_SNAIVE, pv.FB_MEAN]].sum(axis=1),
                   'erro 366 (%)': 100 * pv.taxas_status(t366)[pv.ERRO],
                   'explosivas 200 (n)': t200.assign(e=pv.flag_explosiva(t200)).groupby('model')['e'].sum(),
                   'explosivas 366 (n)': t366.assign(e=pv.flag_explosiva(t366)).groupby('model')['e'].sum()}).round(1)
st.index = [NOMES[m] for m in st.index]
print(st.to_string()); st.to_csv(DIR / 'status_200_366.csv')
'''))

cells.append(md(r'''
## 2. (a) e (b): ranking das 200 x 366 séries
Medianas entre séries (com fallback; SARIMA e SVR também sem fallback e o SARIMA sem explosivas), rank com IC por bootstrap pareado por série dentro de cada amostra.
'''))

cells.append(code(r'''
def tabela(d):
    r = pv.resumo_modelos(pv.agregar_por_serie(d))
    sf = pv.resumo_modelos(pv.agregar_por_serie(d, incluir_fallback=False))
    se = pv.resumo_modelos(pv.agregar_por_serie(d, excluir_explosivas=True))
    return pd.DataFrame({'MAPE': r['mape'], 'MAE': r['mae'], 'MASE': r['mase'],
                         'MAPE sem fallback': sf['mape'], 'MAPE sem explosivas': se['mape']})
tab = pd.concat({'200': tabela(t200), '366': tabela(t366)}, axis=1).round(3)
tab.index = [NOMES[m] for m in tab.index]
print(tab.to_string()); tab.to_csv(DIR / 'desempenho_200_366.csv')

rk, bt = {}, {}
for met in ['mase', 'mape', 'mae']:
    for nome, d in [('200', t200), ('366', t366)]:
        rk[(met, nome)], bt[(met, nome)] = pv.bootstrap_ranks(pv.agregar_por_serie(d), MODELOS, met, n=2000)

linhas = []
for met in ['mase', 'mape', 'mae']:
    a, b = rk[(met, '200')], rk[(met, '366')]
    rho = spearmanr(a['rank'], b['rank'])[0]
    ra, rb = bt[(met, '200')].rank(axis=1).values, bt[(met, '366')].rank(axis=1).values
    rng = np.random.default_rng(pv.SEED)
    rr = [spearmanr(ra[i], rb[j])[0] for i, j in zip(rng.integers(0, 2000, 2000), rng.integers(0, 2000, 2000))]
    linhas.append({'métrica': met.upper(), 'Spearman 200 x 366': round(rho, 3),
                   'IC 95% (aproximado)': f'[{np.percentile(rr, 2.5):.3f}; {np.percentile(rr, 97.5):.3f}]'})
rho_tab = pd.DataFrame(linhas)
print('\n(a) Spearman entre os rankings dos 10 modelos:'); print(rho_tab.to_string(index=False))
rho_tab.to_csv(DIR / 'spearman_ranking_200_366.csv', index=False)

pos = []
for met in ['mase', 'mape']:
    a, b = rk[(met, '200')], rk[(met, '366')]
    for m in MODELOS:
        pos.append({'métrica': met.upper(), 'modelo': NOMES[m],
                    'rank 200': int(a.loc[m, 'rank']), 'IC 200': f"[{a.loc[m, 'rank_ic_inf']:.0f}; {a.loc[m, 'rank_ic_sup']:.0f}]",
                    'rank 366': int(b.loc[m, 'rank']), 'IC 366': f"[{b.loc[m, 'rank_ic_inf']:.0f}; {b.loc[m, 'rank_ic_sup']:.0f}]",
                    'Δ rank': int(b.loc[m, 'rank'] - a.loc[m, 'rank'])})
pos = pd.DataFrame(pos)
print('\n(b) Mudanças de posição:')
for met, g in pos.groupby('métrica', sort=False):
    print(f'=== {met} ==='); print(g.drop(columns='métrica').sort_values('rank 200').to_string(index=False))
    print(f"  modelos que mudam de posição: {int((g['Δ rank'] != 0).sum())} de {len(g)}; maior |Δ|: {int(g['Δ rank'].abs().max())}"); print()
pos.to_csv(DIR / 'posicoes_200_366.csv', index=False)
'''))

cells.append(md(r'''
### 2.1 H4 nas 366 séries (sensibilidade, mesmo critério registrado)
Ordem das famílias por MASE (melhor modelo de cada família) nas 366 séries, comparada à do M4 e do Car Parts.
'''))

cells.append(code(r'''
def familias(t, boot):
    fam = pd.Series(pv.FAMILIA)[MODELOS]
    pont = t.groupby(fam.values)['mediana'].min().rank(method='min').astype(int)
    rb = boot.T.groupby(fam.values).min().T.rank(axis=1, method='min')
    return pd.DataFrame({'rank': pont, 'ic_inf': rb.quantile(0.025), 'ic_sup': rb.quantile(0.975)})
ref = pd.read_csv(V2 / 'tm' / 'limpo' / 'familias_tres_bases_mase_tm.csv', index_col=0)
f366 = familias(rk[('mase', '366')], bt[('mase', '366')])
f200 = familias(rk[('mase', '200')], bt[('mase', '200')])
print('Ordem 200:', ' > '.join(f200.sort_values('rank').index)); print('Ordem 366:', ' > '.join(f366.sort_values('rank').index))
print('Referências (M4 | Car Parts):'); print(ref[['M4', 'Car Parts']].to_string())
print('Mesma ordem das famílias que nas 200 séries:', tuple(f200.sort_values('rank').index) == tuple(f366.sort_values('rank').index))
'''))

cells.append(md(r'''
## 3. (c) Condições de H1 a H3 nas 366 séries (SENSIBILIDADE)
Calculadas no notebook `11e_explicabilidade_tourism_366_v2.ipynb` com o mesmo código da avaliação oficial (gerador `nb09v2.py`), sobre 30 séries sorteadas entre as 366. Comparação condição a condição com a avaliação oficial (200 séries).
'''))

cells.append(code(r'''
c200 = pd.read_csv(V2 / 'tm' / 'xai' / 'xai_tm_avaliacao_previsoes_condicoes.csv')
c366 = pd.read_csv(V2 / 'tm_366' / 'xai' / 'xai_tm_366_avaliacao_previsoes_condicoes.csv')
comp = c200[['hipotese', 'condicao', 'valor', 'ok']].merge(c366[['condicao', 'valor', 'ok']], on='condicao', suffixes=(' 200', ' 366'))
comp['mesma decisão'] = comp['ok 200'] == comp['ok 366']
pd.set_option('display.max_colwidth', 70)
print(comp.drop(columns=['condicao']).assign(condicao=comp['condicao'].str[:70]).to_string(index=False)); print()
s200 = pd.read_csv(V2 / 'tm' / 'replicacao_previsoes_tm.csv').set_index('hipotese')
s366 = pd.read_csv(V2 / 'tm_366' / 'sensibilidade_previsoes_tm366.csv').set_index('hipotese')
res = pd.DataFrame({'oficial (200)': s200.loc[['H1', 'H2', 'H3'], 'status'],
                    'condições 200': [f"{int(comp[(comp.hipotese == h)]['ok 200'].sum())} de {int((comp.hipotese == h).sum())}" for h in ['H1', 'H2', 'H3']],
                    'sensibilidade (366)': s366.loc[['H1', 'H2', 'H3'], 'status'],
                    'condições 366': s366.loc[['H1', 'H2', 'H3'], 'condicoes_validas']})
print(res.to_string()); comp.to_csv(DIR / 'condicoes_h1_h3_200_366.csv', index=False); res.to_csv(DIR / 'h1_h3_200_366.csv')
'''))

cells.append(md(r'''
## 4. (d) Robustez: escore e Δ% nas 200 x 366 séries
'''))

cells.append(code(r'''
RAP = pv.MODELOS_RAPIDOS
TRIV = ['persistence', 'seasonal_naive', 'mean']
def deltas(sub):
    base = pv.agregar_por_serie(carregar(sub, 'limpo', RAP))
    return pd.concat([pv.tabela_delta(base, pv.agregar_por_serie(carregar(sub, c, RAP))).reset_index().assign(cenario=c)
                      for c in pv.CENARIOS], ignore_index=True)
d200, d366 = deltas('tm'), deltas('tm_366')
esc = pd.DataFrame({'escore 200': d200.groupby('model')['delta_pct'].apply(lambda x: x.abs().mean()),
                    'escore 366': d366.groupby('model')['delta_pct'].apply(lambda x: x.abs().mean()),
                    'Δ% médio com sinal 200': d200.groupby('model')['delta_pct'].mean(),
                    'Δ% médio com sinal 366': d366.groupby('model')['delta_pct'].mean()}).round(1)
esc['grupo'] = ['Baseline trivial' if m in TRIV else 'Modelo ajustado' for m in esc.index]
aj = [m for m in RAP if m not in TRIV]
r_aj = spearmanr(esc.loc[aj, 'escore 200'], esc.loc[aj, 'escore 366'])[0]
r_all = spearmanr(esc['escore 200'], esc['escore 366'])[0]
esc.index = [NOMES[m] for m in esc.index]
print(esc.sort_values(['grupo', 'escore 366'], ascending=[False, True]).to_string())
print(f'\n(d) Spearman do escore de robustez 200 x 366: modelos ajustados (6) = {r_aj:+.3f}; todos (9) = {r_all:+.3f}')
j = d200.merge(d366, on=['model', 'cenario'], suffixes=('_200', '_366'))
print(f"Mesmo sinal de Δ% (81 combinações): {int((np.sign(j.delta_pct_200) == np.sign(j.delta_pct_366)).sum())} de {len(j)}; "
      f"correlação de Pearson dos Δ%: {np.corrcoef(j.delta_pct_200, j.delta_pct_366)[0, 1]:.3f}")
ic366 = pd.concat([pv.bootstrap_delta(pv.agregar_por_serie(carregar('tm_366', 'limpo', RAP)),
                                      pv.agregar_por_serie(carregar('tm_366', c, RAP)), RAP, n=2000)[0].assign(cenario=c)
                   for c in pv.CENARIOS])
print(f'ICs do Δ% incluindo zero: 366 séries = {int(ic366.ic_inclui_zero.sum())} de {len(ic366)} (200 séries: 53 de 81)')
esc.to_csv(DIR / 'robustez_200_366.csv'); j.to_csv(DIR / 'delta_200_366.csv', index=False); ic366.to_csv(DIR / 'delta_ic95_366.csv')
'''))

check(cells)
nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': '.venv', 'language': 'python', 'name': 'python3'},
                                   'language_info': {'name': 'python', 'version': '3.11.0'}},
      'nbformat': 4, 'nbformat_minor': 5}
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('ok', len(cells))
