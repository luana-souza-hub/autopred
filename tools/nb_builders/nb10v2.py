import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nbtools import md, code, check

NB = Path(r'C:\Users\Usuario\Desktop\Mestrado\Mestrado - aula\dissertação\autopred\notebooks\10_car_parts_pipeline_v2.ipynb')
cells = []

cells.append(md(r'''
# 10 v2: Car Parts com o protocolo único v2 (desempenho em dados limpos)

**Fase 3** da correção. O NB10 original foi mantido.

### O que mudou em relação ao NB10
| Item | NB10 | v2 |
|---|---|---|
| Random Forest | 30 árvores (só no Car Parts) | 100 árvores, iguais nas três bases (C15) |
| Features de ML | lag_1..12, roll_mean e roll_std (12), lag_season (= lag_12), lag_season2 condicional | 15 features sem duplicata (lag_1..12, roll_mean_3, roll_std_3, roll_mean_6) |
| SARIMA automático | excluído | incluído (seleção de ordem por janela, só com o treino) |
| SARIMA heurístico | fallback silencioso em exceções | ajustes degenerados rejeitados; fallbacks registrados no status (C14) |
| R² | forçado a 0 ou 1 pelo sklearn com teste constante (15,8% das janelas, C13) | indefinido (NaN) nessas janelas; cobertura reportada |
| Cobertura do MAPE | por série, mas rotulada como "por janela" (C20) | reportada por janela **e** por série |

Inalterado: 2.674 séries, H=18, janelas por `linspace(24, 33, 3)` = origens 24, 28 e 33, MAE como métrica principal, MAPE com máscara de zeros.

### Sem cenários adversos (decisão de escopo 3A, 2026-10-02)
Os cenários do NB08 escalam a perturbação pelo desvio padrão da série (ruído σ, outliers ±3σ). Numa série com ~80% de zeros e demanda de 1 a 2 unidades, isso gera valores negativos e fracionários sem interpretação física e "outliers" do tamanho da própria demanda; a degradação medida refletiria o artefato da perturbação, não a robustez do algoritmo. O ruído σ=30% é usado apenas na explicabilidade exploratória, para medir a estabilidade das atribuições.
'''))

cells.append(code(r'''
import sys, json, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_root = Path.cwd().resolve()
PROJECT_ROOT = next(c for c in (_root, *_root.parents) if (c / 'src' / 'pipeline_v2.py').is_file())
sys.path.insert(0, str(PROJECT_ROOT))
from src import pipeline_v2 as pv

DIR = PROJECT_ROOT / 'results' / 'v2' / 'cp' / 'limpo'
DIR_M4 = PROJECT_ROOT / 'results' / 'v2' / 'm4' / 'limpo'
ANT = PROJECT_ROOT / 'results' / 'car_parts'
TSF = PROJECT_ROOT / 'data' / 'raw' / 'car_parts' / 'car_parts_dataset_without_missing_values.tsf'
NOMES = {'holt_winters': 'Holt-Winters', 'persistence': 'Persistência', 'ses': 'SES', 'sarima_auto': 'SARIMA automático',
         'gbt': 'Gradient Boosting', 'sarima_heuristic': 'SARIMA heurístico', 'random_forest': 'Random Forest',
         'seasonal_naive': 'Sazonal ingênuo', 'svr': 'SVR', 'mean': 'Média histórica'}
pd.set_option('display.width', 250)

df = pd.concat([pd.read_csv(f) for f in sorted(DIR.glob('cp_limpo_*_by_window.csv'))], ignore_index=True)
MODELOS = [m for m in pv.MODELOS if m in df.model.unique()]
print(f'{len(df):,} janelas | {df.unique_id.nunique():,} séries | modelos: {MODELOS}')
if 'sarima_auto' not in MODELOS:
    print('ATENÇÃO: SARIMA automático ainda sem cache')

serie = pv.load_tsf(TSF)
zp = pd.Series({u: float((s == 0).mean()) for u, s in serie.items()}, name='zero_prop')
faixa = pd.cut(zp, [0, 0.5, 0.8, 1.01], right=False, labels=['Baixo (<50%)', 'Alto (50% a <80%)', 'Extremo (>=80%)'])
print(faixa.value_counts().sort_index().to_string())
'''))

cells.append(md(r'''
## 1. Status e taxa de fallback por modelo e por janela
'''))

cells.append(code(r'''
taxas = (100 * pv.taxas_status(df)).round(1); taxas.index = [NOMES[m] for m in taxas.index]
print('Status (% das janelas):'); print(taxas.to_string())
fbj = (100 * pv.taxas_fallback_por_janela(df)).round(1); fbj.index = [NOMES[m] for m in fbj.index]
fbj.columns = [f'janela {int(c)} (origem {o})' for c, o in zip(fbj.columns, [24, 28, 33])]
print('\nFallback por janela (% das séries):'); print(fbj.to_string())
motivos = df[df.status.isin([pv.FB_SNAIVE, pv.FB_MEAN]) & (df['info'].fillna('') != '')]
motivos = motivos.assign(motivo=motivos['info'].map(lambda s: json.loads(s).get('motivo', 'n/d')))
print('\nMotivos de fallback:'); print(motivos.groupby(['model', 'motivo']).size().rename('janelas').to_string())
explo = df.assign(e=pv.flag_explosiva(df)).groupby('model')['e'].agg(['sum', 'mean'])
print('\nPrevisões explosivas (sinalizadas, status OK):', {NOMES[m]: f'{int(r["sum"])} ({100*r["mean"]:.2f}%)' for m, r in explo.iterrows() if r['sum'] > 0})
taxas.to_csv(DIR / 'cp_limpo_taxas_status.csv'); fbj.to_csv(DIR / 'cp_limpo_fallback_por_janela.csv')
'''))

cells.append(md(r'''
## 2. Tabela 7 v2 (MAE como métrica principal)
- **MAPE**: calculado só sobre passos com valor real não nulo; janelas com teste todo zero ficam sem MAPE. Cobertura reportada **por janela** (% de janelas com MAPE definido) e **por série** (% de séries com mediana de MAPE definida), corrigindo o rótulo do item C20.
- **R²**: indefinido nas janelas com teste constante (C13); cobertura por janela reportada.
- MAE também sem as janelas de fallback e, para os SARIMA, sem as janelas explosivas.
'''))

cells.append(code(r'''
bs = pv.agregar_por_serie(df)
bs_sem_fb = pv.agregar_por_serie(df, incluir_fallback=False)
bs_sem_exp = pv.agregar_por_serie(df, excluir_explosivas=True)
validos = df[df.status.isin([pv.OK, pv.FB_SNAIVE, pv.FB_MEAN])]
g = bs.groupby('model')
tab7 = pd.DataFrame({
    'Família'              : pd.Series(pv.FAMILIA),
    'MAE mediano'          : g['mae'].median(),
    'MAPE mediano (%)'     : g['mape'].median(),
    'Cob. MAPE janelas (%)': 100 * validos.groupby('model')['mape'].apply(lambda x: x.notna().mean()),
    'Cob. MAPE séries (%)' : 100 * g['mape'].apply(lambda x: x.notna().mean()),
    'R² mediano'           : g['r2'].median(),
    'Cob. R² janelas (%)'  : 100 * validos.groupby('model')['r2'].apply(lambda x: x.notna().mean()),
    'MAE sem fallback'     : bs_sem_fb.groupby('model')['mae'].median(),
    'MAE sem explosivas'   : bs_sem_exp.groupby('model')['mae'].median(),
    'Fallback (% jan.)'    : 100 * pv.taxas_status(df)[[pv.FB_SNAIVE, pv.FB_MEAN]].sum(axis=1),
}).loc[MODELOS]
for f_ in faixa.cat.categories:
    ids_f = faixa[faixa == f_].index
    tab7[f'MAE {f_.split()[0]}'] = bs[bs.unique_id.isin(ids_f)].groupby('model')['mae'].median()
tab7['Rank (MAE)'] = tab7['MAE mediano'].rank(method='min').astype(int)
tab7 = tab7.sort_values('Rank (MAE)').round(3)
tab7.index = [NOMES[m] for m in tab7.index]
print(f'Tabela 7 v2: Car Parts, {bs.unique_id.nunique():,} séries, H=18, 3 origens')
print(tab7.to_string())
tab7.to_csv(DIR / 'tabela7_v2_cp.csv')
'''))

cells.append(md(r'''
## 3. Comparação com a Tabela 7 antiga
'''))

cells.append(code(r'''
TAB7_ANT = {   # dissertação, Tabela 7: MAE, MAPE, R², MAE Baixo, Alto, Extremo
    'persistence': (0.333, 100.0, -0.20, 0.889, 0.667, 0.111), 'random_forest': (0.485, 78.6, -0.14, 0.928, 0.719, 0.217),
    'svr': (0.488, 80.3, -0.10, 0.903, 0.693, 0.250), 'gbt': (0.499, 91.3, -0.29, 1.011, 0.776, 0.167),
    'ses': (0.536, 75.8, -0.09, 1.016, 0.733, 0.250), 'seasonal_naive': (0.556, 100.0, -0.72, 1.278, 0.833, 0.222),
    'mean': (0.576, 78.2, -0.11, 1.206, 0.778, 0.270), 'holt_winters': (0.741, 90.4, -0.80, 1.387, 1.027, 0.431),
    'sarima_heuristic': (0.885, 100.0, -1.67, 1.779, 1.261, 0.389),
}
ant = pd.DataFrame(TAB7_ANT, index=['MAE', 'MAPE', 'R2', 'MAE Baixo', 'MAE Alto', 'MAE Extremo']).T
ant['Rank antigo'] = ant['MAE'].rank(method='min').astype(int)
t7 = tab7.copy(); t7.index = [next(k for k, v in NOMES.items() if v == n) for n in t7.index]
comp = pd.DataFrame({
    'MAE antigo': ant['MAE'], 'MAE v2': t7['MAE mediano'], 'MAE v2 sem fallback': t7['MAE sem fallback'],
    'Fallback v2 (% jan.)': t7['Fallback (% jan.)'], 'Rank antigo': ant['Rank antigo'], 'Rank v2': t7['Rank (MAE)'],
    'MAPE antigo': ant['MAPE'], 'MAPE v2': t7['MAPE mediano (%)'], 'R² antigo': ant['R2'], 'R² v2': t7['R² mediano'],
    'MAE Extremo antigo': ant['MAE Extremo'], 'MAE Extremo v2': t7['MAE Extremo'],
}).reindex(MODELOS)
comp['Δ MAE'] = comp['MAE v2'] - comp['MAE antigo']
comp['Δ rank'] = comp['Rank v2'] - comp['Rank antigo']
comp['Mudou de posição'] = np.where(comp['Δ rank'].fillna(0) != 0, 'SIM', '')
comp = comp.sort_values('Rank v2'); comp.index = [NOMES[m] for m in comp.index]
print(comp.round(3).to_string())
comp.to_csv(DIR / 'comparacao_tabela7_antiga_v2_cp.csv')
'''))

cells.append(md(r'''
### 3.1 Sanidade: onde o protocolo não mudou, o v2 reproduz o NB10?
Persistência, média, sazonal ingênuo, SES e Holt-Winters usam as mesmas funções, séries e janelas do NB10: o MAE por janela deve ser idêntico. SARIMA heurístico: comparação só nas janelas com status OK no v2.
'''))

cells.append(code(r'''
linhas = []
for m in ['persistence', 'mean', 'seasonal_naive', 'ses', 'holt_winters', 'sarima_heuristic']:
    a = pd.read_csv(ANT / f'results_cp_{m}_by_window.csv')[['unique_id', 'origin', 'mae']]
    j = df[(df.model == m) & (df.status == pv.OK)].merge(a, on=['unique_id', 'origin'], suffixes=('_v2', '_antigo'))
    d = (j['mae_v2'] - j['mae_antigo']).abs()
    linhas.append({'modelo': NOMES[m], 'janelas': len(j), 'dif. máx. MAE': f'{d.max():.2e}', 'iguais (%)': round(100 * (d < 1e-9).mean(), 2)})
print(pd.DataFrame(linhas).to_string(index=False))
'''))

cells.append(md(r'''
## 4. Dependência do regime em relação ao M4
A expressão "inversão de hierarquia" é reservada à queda do Holt-Winters (robusta a qualquer métrica) e ao par Tourism x Car Parts (única comparação com separação por IC); para o restante usamos "dependência do regime".
**4.0 (referência, métricas diferentes):** ranking pela métrica principal de cada base (M4: MAPE; Car Parts: MAE), com IC 95% do rank por bootstrap pareado por série **dentro** de cada base (2000 reamostragens; as mesmas séries para todos os modelos). As bases são independentes, então o IC do Spearman entre os rankings reamostra as duas bases de forma independente.
'''))

cells.append(code(r'''
from scipy.stats import spearmanr
m4 = pd.concat([pd.read_csv(f) for f in sorted(DIR_M4.glob('m4_limpo_*_by_window.csv'))], ignore_index=True)
COMUNS = [m for m in MODELOS if m in m4.model.unique()]
tab_m4, boot_m4 = pv.bootstrap_ranks(pv.agregar_por_serie(m4), COMUNS, 'mape', n=2000)
tab_cp, boot_cp = pv.bootstrap_ranks(bs, COMUNS, 'mae', n=2000)
rev = pd.DataFrame({
    'Família': pd.Series(pv.FAMILIA)[COMUNS],
    'Rank M4 (MAPE)': tab_m4['rank'], 'IC M4': tab_m4.apply(lambda r: f'[{r.rank_ic_inf:.0f}; {r.rank_ic_sup:.0f}]', axis=1),
    'Rank Car Parts (MAE)': tab_cp['rank'], 'IC CP': tab_cp.apply(lambda r: f'[{r.rank_ic_inf:.0f}; {r.rank_ic_sup:.0f}]', axis=1),
})
rev['Δ rank (CP − M4)'] = rev['Rank Car Parts (MAE)'] - rev['Rank M4 (MAPE)']
rev['ICs separados'] = (tab_cp['rank_ic_inf'] > tab_m4['rank_ic_sup']) | (tab_cp['rank_ic_sup'] < tab_m4['rank_ic_inf'])
rev = rev.sort_values('Rank M4 (MAPE)'); rev.index = [NOMES[m] for m in rev.index]
print(rev.to_string())

# Família: rank da família = posição do seu melhor modelo, por réplica bootstrap
def rank_familias(boot):
    fam = pd.Series(pv.FAMILIA)[boot.columns]
    melhor = boot.T.groupby(fam.values).min().T            # mediana do melhor membro, por réplica
    return melhor.rank(axis=1, method='min')
fm4, fcp = rank_familias(boot_m4), rank_familias(boot_cp)
pt_m4 = tab_m4.groupby(pd.Series(pv.FAMILIA)[COMUNS].values)['mediana'].min().rank(method='min')
pt_cp = tab_cp.groupby(pd.Series(pv.FAMILIA)[COMUNS].values)['mediana'].min().rank(method='min')
fam_tab = pd.DataFrame({'Rank M4': pt_m4.astype(int), 'IC M4': fm4.quantile([0.025, 0.975]).T.apply(lambda r: f'[{r.iloc[0]:.0f}; {r.iloc[1]:.0f}]', axis=1),
                        'Rank Car Parts': pt_cp.astype(int), 'IC CP': fcp.quantile([0.025, 0.975]).T.apply(lambda r: f'[{r.iloc[0]:.0f}; {r.iloc[1]:.0f}]', axis=1)})
print('\nFamílias (posição do melhor modelo de cada família):'); print(fam_tab.sort_values('Rank M4').to_string())

rho = spearmanr(tab_m4['rank'], tab_cp['rank'])[0]
rk_m4 = boot_m4.rank(axis=1).values; rk_cp = boot_cp.rank(axis=1).values
rng = np.random.default_rng(pv.SEED)
rhos = [spearmanr(rk_m4[i], rk_cp[j])[0] for i, j in zip(rng.integers(0, 2000, 2000), rng.integers(0, 2000, 2000))]
print(f'\nSpearman entre os rankings M4 e Car Parts: {rho:.3f} [IC 95%: {np.percentile(rhos, 2.5):.3f}; {np.percentile(rhos, 97.5):.3f}]')
hw = rev.loc['Holt-Winters']
print(f"Holt-Winters: {hw['Rank M4 (MAPE)']}º no M4 {hw['IC M4']} x {hw['Rank Car Parts (MAE)']}º no Car Parts {hw['IC CP']} | ICs separados: {hw['ICs separados']}")
rev.to_csv(DIR / 'reversao_hierarquia_m4_cp.csv'); fam_tab.to_csv(DIR / 'reversao_familias_m4_cp.csv')
'''))

cells.append(md(r'''
### 4.1 Mesma métrica nas duas bases: MAE e MASE
A comparação 4.0 usa métricas diferentes em cada base (MAPE x MAE), o que confunde mudança de regime com mudança de métrica. Aqui as duas bases são ranqueadas pela **mesma** métrica:
- **MAE** nas duas bases (no M4 a escala varia entre séries, mas o bootstrap é pareado por série dentro de cada base);
- **MASE** nas duas bases: MAE da janela dividido pelo erro in-sample do sazonal ingênuo no treino da janela, média de |y_t − y_(t−12)|. Indefinido quando essa escala é zero (treino sem nenhuma variação sazonal), caso frequente no Car Parts; essas janelas saem e as séries sem nenhuma janela válida também (`n_series`).

Duas versões: **com fallback** (como na Tabela 7) e **SVR e SARIMA sem as janelas de fallback** (para os demais modelos não há fallback).
'''))

cells.append(code(r'''
ids_m4, ser_m4 = pv.load_m4_piloto(PROJECT_ROOT / 'data' / 'raw', 200)
m4x = pv.adicionar_mase(m4, dict(zip(ids_m4, ser_m4)))
cpx = pv.adicionar_mase(df, serie)
COM_FB = ['svr', 'sarima_heuristic', 'sarima_auto']

def by_series_variante(d, sem_fb):
    b = pv.agregar_por_serie(d)
    if not sem_fb:
        return b
    b_sf = pv.agregar_por_serie(d, incluir_fallback=False)
    return pd.concat([b[~b.model.isin(COM_FB)], b_sf[b_sf.model.isin(COM_FB)]], ignore_index=True)

linhas, fam_linhas = [], []
for metrica in ['mae', 'mase']:
    for sem_fb in [False, True]:
        tm4, bm4 = pv.bootstrap_ranks(by_series_variante(m4x, sem_fb), COMUNS, metrica, n=2000)
        tcp, bcp = pv.bootstrap_ranks(by_series_variante(cpx, sem_fb), COMUNS, metrica, n=2000)
        versao = 'SVR e SARIMA sem fallback' if sem_fb else 'com fallback'
        for m in COMUNS:
            linhas.append({'métrica': metrica.upper(), 'versão': versao, 'modelo': NOMES[m], 'família': pv.FAMILIA[m],
                           'rank M4': int(tm4.loc[m, 'rank']), 'IC M4': f"[{tm4.loc[m, 'rank_ic_inf']:.0f}; {tm4.loc[m, 'rank_ic_sup']:.0f}]",
                           'rank CP': int(tcp.loc[m, 'rank']), 'IC CP': f"[{tcp.loc[m, 'rank_ic_inf']:.0f}; {tcp.loc[m, 'rank_ic_sup']:.0f}]",
                           'Δ rank': int(tcp.loc[m, 'rank'] - tm4.loc[m, 'rank']),
                           'ICs separados': bool(tcp.loc[m, 'rank_ic_inf'] > tm4.loc[m, 'rank_ic_sup'] or tcp.loc[m, 'rank_ic_sup'] < tm4.loc[m, 'rank_ic_inf']),
                           'mediana M4': tm4.loc[m, 'mediana'], 'mediana CP': tcp.loc[m, 'mediana'],
                           'n séries M4': int(tm4.loc[m, 'n_series']), 'n séries CP': int(tcp.loc[m, 'n_series'])})
        fr4, frc = rank_familias(bm4), rank_familias(bcp)
        p4 = tm4.groupby(pd.Series(pv.FAMILIA)[COMUNS].values)['mediana'].min().rank(method='min')
        pc = tcp.groupby(pd.Series(pv.FAMILIA)[COMUNS].values)['mediana'].min().rank(method='min')
        for fam in p4.index:
            fam_linhas.append({'métrica': metrica.upper(), 'versão': versao, 'família': fam,
                               'rank M4': int(p4[fam]), 'IC M4': f'[{fr4[fam].quantile(.025):.0f}; {fr4[fam].quantile(.975):.0f}]',
                               'rank CP': int(pc[fam]), 'IC CP': f'[{frc[fam].quantile(.025):.0f}; {frc[fam].quantile(.975):.0f}]'})
        rho = spearmanr(tm4['rank'], tcp['rank'])[0]
        r4, rc = bm4.rank(axis=1).values, bcp.rank(axis=1).values
        rng = np.random.default_rng(pv.SEED)
        rhos = [spearmanr(r4[i], rc[j])[0] for i, j in zip(rng.integers(0, 2000, 2000), rng.integers(0, 2000, 2000))]
        print(f'{metrica.upper():<4} | {versao:<25} | Spearman entre rankings M4 x CP: {rho:+.3f} '
              f'[{np.percentile(rhos, 2.5):+.3f}; {np.percentile(rhos, 97.5):+.3f}] | séries M4={int(tm4.n_series.iloc[0])}, CP={int(tcp.n_series.iloc[0])}')
mm = pd.DataFrame(linhas); fam_mm = pd.DataFrame(fam_linhas)
mm.to_csv(DIR / 'reversao_hierarquia_m4_cp_mesma_metrica.csv', index=False)
fam_mm.to_csv(DIR / 'reversao_familias_m4_cp_mesma_metrica.csv', index=False)
print()
for (met, ver), g in mm.groupby(['métrica', 'versão'], sort=False):
    print(f'=== {met} | {ver} ===')
    print(g.drop(columns=['métrica', 'versão']).sort_values('rank M4').round(3).to_string(index=False)); print()
print('Famílias (posição do melhor modelo):')
print(fam_mm.pivot_table(index='família', columns=['métrica', 'versão'], values=['rank M4', 'rank CP'], aggfunc='first').to_string())
print()
print('Holt-Winters, rank nas duas bases:')
print(mm[mm.modelo == 'Holt-Winters'][['métrica', 'versão', 'rank M4', 'IC M4', 'rank CP', 'IC CP', 'ICs separados']].to_string(index=False))
'''))

cells.append(md(r'''
## 5. Figura: distribuição do MAE por modelo (equivalente à Fig. 24)
'''))

cells.append(code(r'''
ordem = [next(k for k, v in NOMES.items() if v == n) for n in tab7.index]
cores = {'Baseline': '#607D8B', 'Suavização exponencial': '#2196F3', 'ARIMA': '#FF9800', 'ML': '#9C27B0'}
dados = [bs[bs.model == m]['mae'].dropna() for m in ordem]
clip = max(d.quantile(0.95) for d in dados)
fig, ax = plt.subplots(figsize=(15, 5))
bp = ax.boxplot([d.clip(upper=clip) for d in dados], patch_artist=True, medianprops=dict(color='black', linewidth=2),
                flierprops=dict(marker='o', markersize=2, alpha=0.3))
for patch, m in zip(bp['boxes'], ordem):
    patch.set_facecolor(cores[pv.FAMILIA[m]]); patch.set_alpha(0.75)
for i, d in enumerate(dados):
    ax.text(i + 1, d.median(), f'{d.median():.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xticks(range(1, len(ordem) + 1)); ax.set_xticklabels([NOMES[m] for m in ordem], rotation=25, fontsize=9)
for f_, c in cores.items():
    ax.bar(0, 0, color=c, alpha=0.75, label=f_)
ax.legend(fontsize=8); ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.set_ylabel('MAE por série (truncado no p95)')
ax.set_title(f'Car Parts, protocolo v2: MAE por modelo ({bs.unique_id.nunique():,} séries, H=18, 3 origens)', fontsize=11)
plt.tight_layout(); plt.savefig(DIR / 'fig_v2_cp_boxplot_mae.png', dpi=150, bbox_inches='tight'); plt.show()
'''))

check(cells)
nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': '.venv', 'language': 'python', 'name': 'python3'},
                                   'language_info': {'name': 'python', 'version': '3.11.0'}},
      'nbformat': 4, 'nbformat_minor': 5}
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('ok', len(cells))
