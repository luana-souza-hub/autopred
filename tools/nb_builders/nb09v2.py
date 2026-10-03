import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nbtools import md, code, check

BASE = sys.argv[1] if len(sys.argv) > 1 else 'm4'
ROTULO = {'m4': 'M4 Monthly', 'cp': 'Car Parts', 'tm': 'Tourism Monthly',
          'tm_366': 'Tourism Monthly, 366 séries (SENSIBILIDADE)'}[BASE]
NB = Path(r'C:\Users\Usuario\Desktop\Mestrado\Mestrado - aula\dissertação\autopred\notebooks') / \
    {'m4': '09_explicabilidade_v2.ipynb', 'cp': '10b_explicabilidade_car_parts_v2.ipynb',
     'tm': '11c_explicabilidade_tourism_v2.ipynb', 'tm_366': '11e_explicabilidade_tourism_366_v2.ipynb'}[BASE]
cells = []

cells.append(md(rf'''
# Explicabilidade v2: {ROTULO} (protocolo único v2)

**O que é explicado.** Para cada uma das 30 séries (sorteio SEED+1 sobre a amostra da base, regra do NB09), RF, GB e SVR são treinados na **última janela** (treino = série[:-H]), limpo ou com ruído σ=30% do desvio do próprio treino. SHAP e LIME explicam **os mesmos 30 pontos** desse modelo: as 29 linhas de treino mais recentes e o **ponto de previsão** (features do 1º passo da previsão, calculadas a partir do histórico que o modelo vê).

### Diferenças em relação ao NB09
| Item | NB09 | v2 |
|---|---|---|
| Pontos explicados | SHAP: todas as linhas de treino (árvores) ou 30 sorteadas (SVR); LIME: só a última linha de treino | os mesmos 30 pontos para SHAP e LIME |
| Features | 17, com lag_season = lag_12 duplicado | 15, sem duplicata; lag_12 é o lag sazonal |
| LIME | associação por substring (bug) | por índice (`as_map()[1]`) |
| SHAP do SVR | sem semente, nsamples=50, escala normalizada | semente fixa, nsamples = 2p + 2048, escala original |
| Agregação | média simples de importâncias em escala original | normalização por série (soma 1) antes da média |
| Ruído da estabilidade | gerador sequencial sobre as séries | semente por série, σ do próprio treino |
| Incerteza | nenhuma | IC de 95% por bootstrap sobre as séries (2000 reamostragens) |

Resultados oficiais: LIME com 200 amostras. Sensibilidade (1000 e 5000) e análise com lag_24 são **suplementares**.
'''))

cells.append(code(rf'''
import sys, json, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

_root = Path.cwd().resolve()
PROJECT_ROOT = next(c for c in (_root, *_root.parents) if (c / 'src' / 'pipeline_v2.py').is_file())
sys.path.insert(0, str(PROJECT_ROOT))
from src import pipeline_v2 as pv

BASE = '{BASE}'
DIR = PROJECT_ROOT / 'results' / 'v2' / BASE / 'xai'
FN, FN24 = pv.feature_names(), pv.feature_names(com_lag24=True)
MOD = pv.MODELOS_XAI
MET = ['shap', 'lime']
pd.set_option('display.width', 250)

def ler(nome):
    p = DIR / f'xai_{{BASE}}_{{nome}}_pontos.csv'
    return pd.read_csv(p) if p.exists() else None

pts = {{c: ler(c) for c in ['limpo', 'ruido30']}}
sens = {{(n, c): ler(f'{{c}}_lime{{n}}') for n in (1000, 5000) for c in ['limpo', 'ruido30']}}
lag24 = {{c: ler(f'{{c}}_lag24') for c in ['limpo', 'ruido30']}}
for c, d in pts.items():
    print(c, d.groupby(['modelo', 'metodo']).serie.nunique().unstack().to_dict(), '| pontos por série:',
          d.groupby(['serie', 'modelo', 'metodo']).size().unique())

def imp_serie(d, nomes=FN, tipo=None):
    """Importância por série (média de |atribuição| nos pontos) normalizada para somar 1."""
    i = pv.importancia_por_serie(d, nomes, tipo)
    return pv.normalizar_por_serie(i, nomes)

imp = pd.concat([imp_serie(d) for d in pts.values()], ignore_index=True)
imp_prev = pd.concat([imp_serie(d, tipo='previsao') for d in pts.values()], ignore_index=True)
'''))

cells.append(md(r'''
## 1. Importância global (normalizada por série) e top-10
'''))

cells.append(code(r'''
def glob(i, modelo, metodo, cond, nomes=FN):
    return i[(i.modelo == modelo) & (i.metodo == metodo) & (i.condicao == cond)][nomes].mean()

linhas = []
for cond in ['limpo', 'ruido30']:
    for m in MOD:
        for met in MET:
            g = glob(imp, m, met, cond)
            linhas.append({'condicao': cond, 'modelo': m, 'metodo': met,
                           'top10': ', '.join(g.sort_values(ascending=False).index[:10]),
                           'pos_lag_12': int(g.rank(ascending=False, method='min')['lag_12']),
                           'lag_1': g['lag_1'], 'roll_mean_3': g['roll_mean_3'], 'roll_mean_6': g['roll_mean_6'],
                           'lag_12': g['lag_12']})
top = pd.DataFrame(linhas)
top.to_csv(DIR / f'xai_{BASE}_top10_global.csv', index=False)
for cond in ['limpo', 'ruido30']:
    print(f'=== {cond} ===')
    for _, r in top[top.condicao == cond].iterrows():
        print(f"{r.modelo:<17} {r.metodo:<5} lag_12 em {r.pos_lag_12:>2}º | {r.top10}")
    print()
'''))

cells.append(md(r'''
## 2. Estabilidade entre limpo e ruído σ=30% (por série), com IC por bootstrap
Spearman entre os vetores de importância da mesma série (limpo x ruído) e Jaccard dos conjuntos top-3 e top-5. Estatística: média entre as 30 séries; IC 95% por bootstrap (2000). A mediana também é reportada.
'''))

cells.append(code(r'''
def tab_estab(i, nomes=FN):
    linhas = []
    for met in MET:
        sub = i[i.metodo == met]
        e = pv.estabilidade(sub[sub.condicao == 'limpo'], sub[sub.condicao == 'ruido30'], nomes)
        for m in MOD:
            em = e[e.modelo == m]
            lin = {'modelo': m, 'metodo': met, 'n_series': len(em)}
            for col in ['spearman', 'jaccard_top3', 'jaccard_top5']:
                est, lo, hi = pv.bootstrap_ic(em[col].values)
                lin[col] = f'{est:.3f} [{lo:.3f}; {hi:.3f}]'
                lin[f'{col}_mediana'] = round(float(em[col].median()), 3)
            linhas.append(lin)
    return pd.DataFrame(linhas)

estab = tab_estab(imp)
estab.to_csv(DIR / f'xai_{BASE}_estabilidade.csv', index=False)
print(estab.to_string(index=False))
'''))

cells.append(md(r'''
## 3. Concordância SHAP x LIME
(a) Entre as importâncias globais (normalizadas por série): Pearson (normalização pelo máximo, como na Fig. 20) e Spearman. (b) Por série: Spearman entre o vetor SHAP e o vetor LIME da mesma série, média com IC.
'''))

cells.append(code(r'''
def tab_conc(i, nomes=FN):
    linhas = []
    for cond in ['limpo', 'ruido30']:
        for m in MOD:
            s, l = glob(i, m, 'shap', cond, nomes), glob(i, m, 'lime', cond, nomes)
            pear = np.corrcoef((s / s.max()).values, (l / l.max()).values)[0, 1]
            spe = spearmanr(s.values, l.values)[0]
            a = i[(i.modelo == m) & (i.condicao == cond)]
            j = a[a.metodo == 'shap'].merge(a[a.metodo == 'lime'], on='serie', suffixes=('_s', '_l'))
            por_serie = [spearmanr(r[[f'{n}_s' for n in nomes]].astype(float), r[[f'{n}_l' for n in nomes]].astype(float))[0]
                         for _, r in j.iterrows()]
            est, lo, hi = pv.bootstrap_ic(por_serie)
            linhas.append({'condicao': cond, 'modelo': m, 'pearson_global': round(pear, 3), 'spearman_global': round(spe, 3),
                           'spearman_por_serie': f'{est:.3f} [{lo:.3f}; {hi:.3f}]'})
    return pd.DataFrame(linhas)

conc = tab_conc(imp)
conc.to_csv(DIR / f'xai_{BASE}_concordancia_shap_lime.csv', index=False)
print(conc.to_string(index=False))
'''))

cells.append(md(r'''
## 4. Posição do lag_12 (lag sazonal): global e por série
'''))

cells.append(code(r'''
def pos_lag(i, feat='lag_12', nomes=FN):
    linhas = []
    for cond in ['limpo', 'ruido30']:
        for m in MOD:
            for met in MET:
                sub = i[(i.modelo == m) & (i.metodo == met) & (i.condicao == cond)]
                r = sub[nomes].rank(axis=1, ascending=False, method='min')[feat]
                g = glob(i, m, met, cond, nomes)
                linhas.append({'condicao': cond, 'modelo': m, 'metodo': met,
                               'pos_global': int(g.rank(ascending=False, method='min')[feat]),
                               'pos_mediana_por_serie': float(r.median()),
                               '% séries no top-3': round(100 * (r <= 3).mean(), 1),
                               'importância normalizada': round(g[feat], 4)})
    return pd.DataFrame(linhas)

lag12 = pos_lag(imp)
lag12.to_csv(DIR / f'xai_{BASE}_posicao_lag12.csv', index=False)
print(lag12.to_string(index=False))
'''))

cells.append(md(r'''
## 5. Redistribuição de importância sob ruído (equivalente à Fig. 22)
Importância normalizada (fração da soma por série) de lag_1, roll_mean_3 e roll_mean_6, limpo x ruído, com IC 95% por bootstrap sobre as séries.
'''))

cells.append(code(r'''
FEATS = ['lag_1', 'roll_mean_3', 'roll_mean_6', 'lag_12']
linhas = []
for m in MOD:
    for met in MET:
        for f in FEATS:
            lin = {'modelo': m, 'metodo': met, 'feature': f}
            for cond in ['limpo', 'ruido30']:
                v = imp[(imp.modelo == m) & (imp.metodo == met) & (imp.condicao == cond)][f].values
                est, lo, hi = pv.bootstrap_ic(v)
                lin[cond], lin[f'{cond}_ic'] = round(est, 3), f'[{lo:.3f}; {hi:.3f}]'
            a = imp[(imp.modelo == m) & (imp.metodo == met)]
            j = a[a.condicao == 'limpo'][['serie', f]].merge(a[a.condicao == 'ruido30'][['serie', f]], on='serie')
            est, lo, hi = pv.bootstrap_ic((j[f'{f}_y'] - j[f'{f}_x']).values)
            lin['Δ (ruído − limpo)'] = f'{est:+.3f} [{lo:+.3f}; {hi:+.3f}]'
            lin['IC inclui zero'] = lo <= 0 <= hi
            linhas.append(lin)
redis = pd.DataFrame(linhas)
redis.to_csv(DIR / f'xai_{BASE}_redistribuicao_ruido.csv', index=False)
print(redis.to_string(index=False))

fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
for r_, met in enumerate(MET):
    for c_, m in enumerate(MOD):
        ax = axes[r_, c_]; sub = redis[(redis.modelo == m) & (redis.metodo == met)]
        x = np.arange(len(FEATS))
        ax.bar(x - 0.2, sub['limpo'], 0.4, label='limpo', color='#5B8DB8')
        ax.bar(x + 0.2, sub['ruido30'], 0.4, label='ruído σ=30%', color='#E8834E')
        ax.set_xticks(x); ax.set_xticklabels(FEATS, fontsize=8); ax.set_title(f'{m} | {met.upper()}', fontsize=10)
        if c_ == 0: ax.set_ylabel('importância normalizada (fração por série)')
axes[0, 0].legend(fontsize=8)
fig.suptitle(f'{BASE.upper()}, v2: redistribuição de importância sob ruído (30 séries, mesmos pontos para SHAP e LIME)', fontsize=11)
plt.tight_layout(); plt.savefig(DIR / f'fig_v2_{BASE}_redistribuicao_ruido.png', dpi=150, bbox_inches='tight'); plt.show()
'''))

cells.append(md(r'''
### 5.1 Só o ponto de previsão (suplementar)
As mesmas métricas usando apenas o ponto de previsão (1 ponto por série), em vez dos 30 pontos.
'''))

cells.append(code(r'''
print('Estabilidade (só o ponto de previsão):'); print(tab_estab(imp_prev).to_string(index=False)); print()
print('Concordância SHAP x LIME (só o ponto de previsão):'); print(tab_conc(imp_prev).to_string(index=False))
'''))

cells.append(md(r'''
## 6. Sensibilidade do LIME (1000 e 5000 amostras), separada do resultado oficial
O SHAP é o mesmo do resultado oficial; só o LIME muda.
'''))

cells.append(code(r'''
linhas = []
for n in (200, 1000, 5000):
    if n == 200:
        lime_d = {c: pts[c][pts[c].metodo == 'lime'] for c in pts}
    else:
        if any(sens[(n, c)] is None for c in pts):
            print(f'sem cache para {n} amostras'); continue
        lime_d = {c: sens[(n, c)] for c in pts}
    comb = pd.concat([imp_serie(pts[c][pts[c].metodo == 'shap']) for c in pts] +
                     [imp_serie(lime_d[c]) for c in pts], ignore_index=True)
    cc = tab_conc(comb); pl = pos_lag(comb); es = tab_estab(comb)
    for m in MOD:
        for cond in ['limpo', 'ruido30']:
            g = glob(comb, m, 'lime', cond)
            linhas.append({'lime_samples': n, 'modelo': m, 'condicao': cond,
                           'pearson_shap_lime': cc[(cc.modelo == m) & (cc.condicao == cond)].pearson_global.iloc[0],
                           'spearman_shap_lime': cc[(cc.modelo == m) & (cc.condicao == cond)].spearman_global.iloc[0],
                           'pos_lag_12_lime': pl[(pl.modelo == m) & (pl.condicao == cond) & (pl.metodo == 'lime')].pos_global.iloc[0],
                           'estab_spearman_lime': es[(es.modelo == m) & (es.metodo == 'lime')].spearman.iloc[0] if cond == 'limpo' else '',
                           'top10_lime': ', '.join(g.sort_values(ascending=False).index[:10])})
sens_tab = pd.DataFrame(linhas)
sens_tab.to_csv(DIR / f'xai_{BASE}_sensibilidade_lime.csv', index=False)
print(sens_tab.drop(columns='top10_lime').to_string(index=False)); print()
for _, r in sens_tab[sens_tab.condicao == 'limpo'].iterrows():
    print(f'{r.lime_samples:>5} {r.modelo:<17} {r.top10_lime}')
'''))

cells.append(md(r'''
## 7. Análise SUPLEMENTAR (não oficial): última janela com lag_24 incluído
16 features (as 15 oficiais + lag_24). Os números não substituem os oficiais.
'''))

cells.append(code(r'''
if all(lag24[c] is not None for c in lag24):
    imp24 = pd.concat([imp_serie(lag24[c], FN24) for c in lag24], ignore_index=True)
    p12, p24 = pos_lag(imp24, 'lag_12', FN24), pos_lag(imp24, 'lag_24', FN24)
    sup = p12[['condicao', 'modelo', 'metodo', 'pos_global', '% séries no top-3']].rename(
        columns={'pos_global': 'pos lag_12', '% séries no top-3': 'lag_12 top-3 (%)'})
    sup['pos lag_24'] = p24['pos_global'].values
    sup['lag_24 top-3 (%)'] = p24['% séries no top-3'].values
    sup['lag_12 + lag_24 (fração)'] = (p12['importância normalizada'] + p24['importância normalizada']).round(3).values
    print(sup.to_string(index=False))
    print('\nConcordância SHAP x LIME com lag_24:'); print(tab_conc(imp24, FN24).to_string(index=False))
    print('\nEstabilidade com lag_24:'); print(tab_estab(imp24, FN24).to_string(index=False))
    sup.to_csv(DIR / f'xai_{BASE}_suplementar_lag24.csv', index=False)
else:
    print('caches com lag_24 ausentes')
'''))

if BASE in ('tm', 'tm_366'):
    if BASE == 'tm_366':
        cells.append(md(r'''
> **SENSIBILIDADE (Fase 4d).** As mesmas condições de H1 a H3 recalculadas com as 366 séries do Tourism (30 séries explicadas sorteadas com SEED+1 entre as 366). A avaliação oficial das previsões é a das 200 séries (Fase 4c, `tm/replicacao_previsoes_tm.csv`); os rótulos abaixo **não** a substituem nem alteram `PREVISOES_tourism.md`.
'''))
    cells.append(md(r'''
## 8. Avaliação das previsões H1, H2 e H3 (critérios registrados)
Critérios de `results/v2/PREVISOES_tourism.md` (commit 1a08fa1) com a interpretação de `results/v2/INTERPRETACAO_criterios_4c.md` (commit 1e3d9f0): faixa da H1 **fechada** [0,65; 0,85]; um Spearman por série acima de 0,85 conta como fora da faixa (concordância maior que a prevista); rótulos pelas contagens registradas, sem reinterpretação. Configuração oficial: 200 séries, 30 explicadas, 15 features, LIME com 200 amostras; lag_24 e sensibilidade do LIME **não** entram.
'''))
    cells.append(code(r'''
def rotulo(n_ok, n_total, faixa_parcial):
    if n_ok == n_total:
        return 'confirmou'
    return 'confirmou parcialmente' if faixa_parcial[0] <= n_ok <= faixa_parcial[1] else 'não confirmou'

cond = []
# ---- H1: Pearson agregado > 0,90 e Spearman por série em [0,65; 0,85], condição limpa, por modelo
for m in MOD:
    s, l = glob(imp, m, 'shap', 'limpo'), glob(imp, m, 'lime', 'limpo')
    pear = float(np.corrcoef((s / s.max()).values, (l / l.max()).values)[0, 1])
    a = imp[(imp.modelo == m) & (imp.condicao == 'limpo')]
    j = a[a.metodo == 'shap'].merge(a[a.metodo == 'lime'], on='serie', suffixes=('_s', '_l'))
    sp = [spearmanr(r[[f'{n}_s' for n in FN]].astype(float), r[[f'{n}_l' for n in FN]].astype(float))[0] for _, r in j.iterrows()]
    sp_m, sp_lo, sp_hi = pv.bootstrap_ic(sp)
    cond.append({'hipotese': 'H1', 'condicao': f'{m}: Pearson agregado > 0,90', 'valor': round(pear, 3), 'ok': pear > 0.90})
    obs = '' if sp_m <= 0.85 else 'acima da faixa: concordância maior que a prevista'
    cond.append({'hipotese': 'H1', 'condicao': f'{m}: Spearman por série em [0,65; 0,85]',
                 'valor': f'{sp_m:.3f} [{sp_lo:.3f}; {sp_hi:.3f}]', 'ok': 0.65 <= sp_m <= 0.85, 'obs': obs})
# ---- H2 (a): Spearman médio limpo x ruído < 0,80, por modelo e método
for met in MET:
    sub = imp[imp.metodo == met]
    e = pv.estabilidade(sub[sub.condicao == 'limpo'], sub[sub.condicao == 'ruido30'], FN)
    for m in MOD:
        v, lo, hi = pv.bootstrap_ic(e[e.modelo == m]['spearman'].values)
        cond.append({'hipotese': 'H2', 'condicao': f'(a) {m} {met.upper()}: estabilidade (Spearman médio) < 0,80',
                     'valor': f'{v:.3f} [{lo:.3f}; {hi:.3f}]', 'ok': v < 0.80})
# ---- H2 (b): RF e GB, SHAP e LIME: Δ lag_1 < 0 e Δ roll_mean_3 ou roll_mean_6 > 0, com IC excluindo zero
def delta_ic(m, met, f):
    a = imp[(imp.modelo == m) & (imp.metodo == met)]
    j = a[a.condicao == 'limpo'][['serie', f]].merge(a[a.condicao == 'ruido30'][['serie', f]], on='serie')
    return pv.bootstrap_ic((j[f'{f}_y'] - j[f'{f}_x']).values)
for m in ['RandomForest', 'GradientBoosting']:
    for met in MET:
        d1, d3, d6 = delta_ic(m, met, 'lag_1'), delta_ic(m, met, 'roll_mean_3'), delta_ic(m, met, 'roll_mean_6')
        perde = d1[2] < 0
        ganha = (d3[1] > 0) or (d6[1] > 0)
        cond.append({'hipotese': 'H2', 'condicao': f'(b) {m} {met.upper()}: lag_1 perde e média móvel ganha (IC excluindo zero)',
                     'valor': f'Δlag_1 {d1[0]:+.3f} [{d1[1]:+.3f}; {d1[2]:+.3f}]; Δroll_mean_3 {d3[0]:+.3f} [{d3[1]:+.3f}; {d3[2]:+.3f}]; '
                              f'Δroll_mean_6 {d6[0]:+.3f} [{d6[1]:+.3f}; {d6[2]:+.3f}]',
                     'ok': bool(perde and ganha),
                     'obs': ('' if perde else 'lag_1 não perde com IC excluindo zero; ') + ('' if ganha else 'nenhuma média móvel ganha com IC excluindo zero')})
# ---- H3: fração de séries com lag_12 no top-3 > 30,0%, condição limpa, por modelo e método
for m in MOD:
    for met in MET:
        sub = imp[(imp.modelo == m) & (imp.metodo == met) & (imp.condicao == 'limpo')]
        r = sub[FN].rank(axis=1, ascending=False, method='min')['lag_12']
        frac = 100 * float((r <= 3).mean())
        cond.append({'hipotese': 'H3', 'condicao': f'{m} {met.upper()}: lag_12 no top-3 em > 30,0% das séries',
                     'valor': f'{frac:.1f}% ({int((r <= 3).sum())} de {len(r)})', 'ok': frac > 30.0})

cond = pd.DataFrame(cond).fillna('')
cond.to_csv(DIR / f'xai_{BASE}_avaliacao_previsoes_condicoes.csv', index=False)   # tm: xai_tm_...; tm_366: xai_tm_366_...
FAIXAS = {'H1': (6, (3, 5)), 'H2': (10, (5, 9)), 'H3': (6, (1, 5))}
res_h = []
for h, (n_tot, faixa) in FAIXAS.items():
    c = cond[cond.hipotese == h]
    assert len(c) == n_tot, (h, len(c))
    n_ok = int(c['ok'].sum())
    res_h.append({'hipotese': h, 'condicoes_validas': f'{n_ok} de {n_tot}', 'status': rotulo(n_ok, n_tot, faixa)})
    print(f'=== {h}: {n_ok} de {n_tot} condições -> {rotulo(n_ok, n_tot, faixa).upper()} ===')
    print(c[['condicao', 'valor', 'ok', 'obs']].to_string(index=False)); print()
res_h = pd.DataFrame(res_h)
print(res_h.to_string(index=False))
'''))
    if BASE == 'tm_366':
        cells.append(code(r'''
sens = res_h.assign(rotulo='SENSIBILIDADE (366 séries); não substitui a avaliação oficial (200 séries)')
sens.to_csv(PROJECT_ROOT / 'results' / 'v2' / 'tm_366' / 'sensibilidade_previsoes_tm366.csv', index=False)
print(sens.to_string(index=False))
'''))
    else:
      cells.append(code(r'''
# Atualiza a tabela de replicação (H4 vem da 4a e não muda)
arq = PROJECT_ROOT / 'results' / 'v2' / 'tm' / 'replicacao_previsoes_tm.csv'
rep = pd.read_csv(arq)
for _, r in res_h.iterrows():
    c = cond[cond.hipotese == r.hipotese]
    falhas = '; '.join(f"{x.condicao} = {x.valor}" for x in c.itertuples() if not x.ok)
    rep.loc[rep.hipotese == r.hipotese, 'status'] = r.status
    rep.loc[rep.hipotese == r.hipotese, 'evidencia'] = f"{r.condicoes_validas} condições; não valem: {falhas if falhas else 'nenhuma'}"
    rep.loc[rep.hipotese == r.hipotese, 'parte'] = '4c'
    rep.loc[rep.hipotese == r.hipotese, 'fonte'] = 'tm/xai/xai_tm_avaliacao_previsoes_condicoes.csv'
rep.to_csv(arq, index=False)
pd.set_option('display.max_colwidth', 120)
print(rep[['hipotese', 'status', 'evidencia', 'parte']].to_string(index=False))
'''))

if BASE == 'm4':
    cells.append(md(r'''
## 8. Comparação com a dissertação (Fig. 20, Fig. 22 e Spearman de estabilidade)
Valores antigos: dissertação (com o bug do LIME e a feature duplicada) e NB09 corrigido em out/2026 (LIME por índice, mas ainda com os demais problemas do NB09).
'''))
    cells.append(code(r'''
ANT = pd.DataFrame({
    'Pearson SHAP×LIME, dissertação (Fig. 20)': {'RandomForest': 0.959, 'GradientBoosting': 0.952, 'SVR': 0.908},
    'Pearson, NB09 com LIME corrigido': {'RandomForest': 0.992, 'GradientBoosting': 0.994, 'SVR': 0.980},
    'Spearman estabilidade SHAP (mediana), dissertação': {'RandomForest': 0.572, 'GradientBoosting': 0.344, 'SVR': 0.520},
})
c_l = conc[conc.condicao == 'limpo'].set_index('modelo')
e_s = estab[estab.metodo == 'shap'].set_index('modelo')
ANT['Pearson v2 (limpo)'] = c_l['pearson_global']
ANT['Spearman SHAP×LIME v2 (limpo)'] = c_l['spearman_global']
ANT['Spearman estabilidade SHAP v2 (mediana)'] = e_s['spearman_mediana']
ANT['Spearman estabilidade SHAP v2 (média [IC])'] = e_s['spearman']
print(ANT.T.to_string()); print()

# Fig. 22: valores antigos (medianas brutas de |SHAP|) e a mesma informação normalizada
st = pd.read_csv(PROJECT_ROOT / 'results' / 'explicabilidade' / 'shap_estabilidade_adversa.csv')
FN_ANT = [c.replace('shap_clean_', '') for c in st.columns if c.startswith('shap_clean_')]
linhas = []
for m in MOD:
    s = st[st.modelo == m]
    cl = s[[f'shap_clean_{f}' for f in FN_ANT]].values; pe = s[[f'shap_pert_{f}' for f in FN_ANT]].values
    cln, pen = cl / cl.sum(1, keepdims=True), pe / pe.sum(1, keepdims=True)
    for f in ['lag_1', 'roll_mean_6']:
        k = FN_ANT.index(f)
        v2 = redis[(redis.modelo == m) & (redis.metodo == 'shap') & (redis.feature == f)].iloc[0]
        linhas.append({'modelo': m, 'feature': f,
                       'antigo bruto (mediana) limpo→ruído': f'{np.median(cl[:, k]):.3f} → {np.median(pe[:, k]):.3f}',
                       'antigo normalizado (média) limpo→ruído': f'{cln[:, k].mean():.3f} → {pen[:, k].mean():.3f}',
                       'v2 normalizado (média) limpo→ruído': f"{v2['limpo']:.3f} → {v2['ruido30']:.3f}",
                       'v2 Δ [IC 95%]': v2['Δ (ruído − limpo)']})
fig22 = pd.DataFrame(linhas)
print(fig22.to_string(index=False))
fig22.to_csv(DIR / 'xai_m4_comparacao_fig22.csv', index=False)
ANT.to_csv(DIR / 'xai_m4_comparacao_dissertacao.csv')
'''))

check(cells)
nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': '.venv', 'language': 'python', 'name': 'python3'},
                                   'language_info': {'name': 'python', 'version': '3.11.0'}},
      'nbformat': 4, 'nbformat_minor': 5}
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('ok', NB.name, len(cells))
