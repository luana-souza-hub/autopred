import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nbtools import md, code, check

NB = Path(r'C:\Users\Usuario\Desktop\Mestrado\Mestrado - aula\dissertação\autopred\notebooks\12_fase5_comparativos_v2.ipynb')
cells = []

cells.append(md(r'''
# 12 v2: Fase 5, tabelas comparativas (dissertação x protocolo v2) e replicação de H1 a H4

Consolida em `results/v2/fase5/` (i) as Tabelas 3 a 7 da dissertação com o valor antigo ao lado do valor v2, (ii) as figuras de resultados (Figs. 8 a 26) refeitas no v2, cada uma com a versão antiga ao lado quando o PNG antigo existe, e (iii) a tabela de replicação de H1 a H4 nas três bases.

**Regras desta fase**
- Só leitura dos caches v2 (nenhum modelo de previsão é reexecutado). Cálculo novo, leve: bootstrap de ranks, CV da Fig. 18 a partir dos caches da 2c, importância nativa da Fig. 12 (um ajuste de RF e GB numa série) e as previsões das séries de exemplo das Figs. 10 e 26.
- Valores antigos lidos do `.docx` da dissertação **só para leitura** (o arquivo não é alterado; o SHA-256 vai no `.meta.json`).
- SARIMA e SVR sempre com e sem fallback; SARIMA também sem as janelas explosivas; baselines triviais separados nos rankings de robustez; Δ% com sinal; IC 95% por bootstrap pareado por série (2000).
- Replicação: no Tourism, os rótulos oficiais das **200 séries** (`tm/replicacao_previsoes_tm.csv`), sem reinterpretação; no M4 e no Car Parts, os **critérios numéricos registrados para o Tourism aplicados a posteriori**, com a formulação qualitativa da dissertação ao lado. A sensibilidade com 366 séries fica em arquivo separado.
- Limiares da Tabela 4 e critérios da Tabela 5 foram definidos depois de ver o M4 (não são pré-registrados).
'''))

cells.append(code(r'''
import sys, json, re, zipfile, hashlib, warnings
import xml.etree.ElementTree as ET
from datetime import datetime
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

_root = Path.cwd().resolve()
PROJECT_ROOT = next(c for c in (_root, *_root.parents) if (c / 'src' / 'pipeline_v2.py').is_file())
sys.path.insert(0, str(PROJECT_ROOT))
from src import pipeline_v2 as pv

V2, OLD, DATA = PROJECT_ROOT / 'results' / 'v2', PROJECT_ROOT / 'results', PROJECT_ROOT / 'data' / 'raw'
F5 = V2 / 'fase5'; FIG = F5 / 'figuras'; ANTD = F5 / 'antigo_dissertacao'
for d in (F5, FIG, ANTD):
    d.mkdir(parents=True, exist_ok=True)
DOCX = PROJECT_ROOT.parent / 'dissertacao' / 'Dissertacao_PPGI_AutoPred_2.docx'
NOMES = {'holt_winters': 'Holt-Winters', 'persistence': 'Persistência', 'ses': 'SES', 'sarima_auto': 'SARIMA automático',
         'gbt': 'Gradient Boosting', 'sarima_heuristic': 'SARIMA heurístico', 'random_forest': 'Random Forest',
         'seasonal_naive': 'Sazonal ingênuo', 'svr': 'SVR', 'mean': 'Média histórica'}
COD = {v: k for k, v in NOMES.items()}
MODS = list(pv.MODELOS)
RAP = pv.MODELOS_RAPIDOS
TRIV = ['persistence', 'seasonal_naive', 'mean']
AJUST = [m for m in RAP if m not in TRIV]
CORES_FAM = {'Baseline': '#607D8B', 'Suavização exponencial': '#2196F3', 'ARIMA': '#FF9800', 'ML': '#9C27B0'}
FN = pv.feature_names()
MOD_XAI = pv.MODELOS_XAI
pd.set_option('display.width', 260); pd.set_option('display.max_colwidth', 90)

def num(s):
    """Número do texto da dissertação (vírgula decimal, sinal tipográfico, ordinal); NaN para traço."""
    s = str(s).strip().replace('\u2212', '-').replace('°', '').replace('º', '').replace(',', '.')
    try:
        return float(s)
    except ValueError:
        return np.nan

# ---- tabelas da dissertação (somente leitura do .docx)
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
DOCX_SHA = hashlib.sha256(DOCX.read_bytes()).hexdigest()
_root_xml = ET.fromstring(zipfile.ZipFile(DOCX).read('word/document.xml'))
def _tab(t):
    rows = [[''.join(x.text or '' for x in c.iter(W + 't')).strip() for c in r.findall(W + 'tc')] for r in t.findall(W + 'tr')]
    return pd.DataFrame(rows[1:], columns=rows[0])
TD = [_tab(t) for t in _root_xml.iter(W + 'tbl')]
ANT = {3: TD[3], 4: TD[4], 5: TD[5], 6: TD[6], 7: TD[7]}   # TD[0] licença, TD[1] Tabela 1, TD[2] Tabela 2
assert list(ANT[3].columns) == ['Algoritmo', 'Família', 'MAPE (%)', 'MAE', 'R² mediano', 'Rank']
assert ANT[4].columns[2] == 'Desemp. limpo' and ANT[5].columns[2] == 'Algoritmo recomendado'
assert ANT[6].columns[0] == 'Característica' and ANT[7].columns[1] == 'MAE med.'
for k, t in ANT.items():
    t.to_csv(ANTD / f'tabela{k}_dissertacao.csv', index=False)
print(f'.docx lido (sha256 {DOCX_SHA[:16]}); Tabelas 3 a 7 gravadas em {ANTD.relative_to(PROJECT_ROOT)}')

# ---- janelas v2 das três bases (com MASE)
def carregar(sub, pasta, modelos=None):
    d = pd.concat([pd.read_csv(f) for f in sorted((V2 / sub / pasta).glob('*_by_window.csv'))], ignore_index=True)
    return d if modelos is None else d[d.model.isin(modelos)]

ids_m4, ser_m4 = pv.load_m4_piloto(DATA, 200); S_M4 = dict(zip(ids_m4, ser_m4))
S_CP = pv.load_tsf(DATA / 'car_parts' / 'car_parts_dataset_without_missing_values.tsf')
S_TM_TODAS = pv.load_tsf(DATA / 'tourism_monthly_dataset.tsf', 'latin-1')
ids_tm, ser_tm = pv.amostra_tsf(S_TM_TODAS, 200); S_TM = dict(zip(ids_tm, ser_tm))
JAN = {'M4': pv.adicionar_mase(carregar('m4', 'limpo'), S_M4),
       'Car Parts': pv.adicionar_mase(carregar('cp', 'limpo'), S_CP),
       'Tourism': pv.adicionar_mase(carregar('tm', 'limpo'), S_TM)}
BS = {b: pv.agregar_por_serie(d) for b, d in JAN.items()}
hashes = sorted({json.loads(f.read_text(encoding='utf-8'))['pipeline_sha256'][:8]
                 for s in ('m4', 'cp', 'tm') for f in (V2 / s).rglob('*.meta.json')})
for b, d in JAN.items():
    print(f'{b:<9}: {len(d):>6,} janelas | {d.unique_id.nunique():>4} séries | {d.model.nunique()} modelos')
print('hashes do pipeline nos caches lidos:', hashes, '| pipeline atual:', pv.hash_pipeline()[:8])
'''))

# ---------------------------------------------------------------- Tabela 3
cells.append(md(r'''
## 1. Tabela 3: desempenho em dados limpos no M4 (antigo x v2)
Valores antigos: Tabela 3 do `.docx`. No antigo, os estatísticos e o ML vinham de amostras diferentes (item C6); no v2, amostra única de 200 séries. Rank e IC por bootstrap pareado por série; MASE acrescentado como métrica comum às três bases.
'''))

cells.append(code(r'''
d = JAN['M4']; bs = BS['M4']
r = pv.resumo_modelos(bs)
r_sf = pv.resumo_modelos(pv.agregar_por_serie(d, incluir_fallback=False))
r_se = pv.resumo_modelos(pv.agregar_por_serie(d, excluir_explosivas=True))
rk_mape, _ = pv.bootstrap_ranks(bs, MODS, 'mape'); rk_mase, _ = pv.bootstrap_ranks(bs, MODS, 'mase')
fb = 100 * pv.taxas_status(d)[[pv.FB_SNAIVE, pv.FB_MEAN]].sum(axis=1)
ex = 100 * d.assign(e=pv.flag_explosiva(d)).groupby('model')['e'].mean()
a3 = ANT[3].set_index('Algoritmo')
amostra_ant = pd.read_csv(V2 / 'm4' / 'limpo' / 'comparacao_tabela3_antiga_v2_m4.csv', index_col=0)['Amostra antiga']
ic = lambda t, m, c: f"[{t.loc[m, c + '_ic_inf']:.0f}; {t.loc[m, c + '_ic_sup']:.0f}]" if c == 'rank' else \
    f"[{t.loc[m, 'mediana_ic_inf']:.2f}; {t.loc[m, 'mediana_ic_sup']:.2f}]"
linhas = []
for m in MODS:
    n = NOMES[m]
    linhas.append({'Algoritmo': n, 'Família (dissertação)': a3.loc[n, 'Família'], 'Família (v2)': pv.FAMILIA[m],
                   'Amostra antiga': amostra_ant[n],
                   'MAPE antigo': num(a3.loc[n, 'MAPE (%)']), 'MAPE v2': round(r.loc[m, 'mape'], 2), 'IC MAPE v2': ic(rk_mape, m, 'mediana'),
                   'Δ MAPE (p.p.)': round(r.loc[m, 'mape'] - num(a3.loc[n, 'MAPE (%)']), 2),
                   'Rank antigo': int(num(a3.loc[n, 'Rank'])), 'Rank v2 (MAPE)': int(rk_mape.loc[m, 'rank']), 'IC rank v2': ic(rk_mape, m, 'rank'),
                   'MAPE v2 sem fallback': round(r_sf.loc[m, 'mape'], 2), 'MAPE v2 sem explosivas': round(r_se.loc[m, 'mape'], 2),
                   'Fallback v2 (% jan.)': round(fb[m], 1), 'Explosivas v2 (% jan.)': round(ex[m], 1),
                   'MAE antigo': num(a3.loc[n, 'MAE']), 'MAE v2': round(r.loc[m, 'mae'], 2),
                   'R² antigo': num(a3.loc[n, 'R² mediano']), 'R² v2': round(r.loc[m, 'r2'], 2),
                   'MASE v2': round(r.loc[m, 'mase'], 3), 'Rank v2 (MASE)': f"{int(rk_mase.loc[m, 'rank'])} {ic(rk_mase, m, 'rank')}"})
t3 = pd.DataFrame(linhas).sort_values('Rank v2 (MAPE)')
ref = pd.read_csv(V2 / 'm4' / 'limpo' / 'tabela3_v2_m4.csv', index_col=0)
assert np.allclose(t3.set_index('Algoritmo')['MAPE v2'], ref.loc[t3.Algoritmo, 'MAPE (%)'], atol=0.006)
t3['Mudou de posição'] = np.where(t3['Rank antigo'] != t3['Rank v2 (MAPE)'], 'SIM', '')
t3.to_csv(F5 / 'tabela3_antigo_v2_m4.csv', index=False)
print(t3.drop(columns=['Família (dissertação)', 'Amostra antiga']).to_string(index=False))
print('\nMAE e R² antigos só existem para os baselines e o HW (traço na dissertação para os demais).')
'''))

# ---------------------------------------------------------------- tempos
cells.append(md(r'''
## 2. Custo computacional medido no v2 (usado nas Tabelas 4, 5 e 6)
Duas medições, ambas das execuções v2: (a) **tempo de parede** por modelo, lido dos logs de `scripts/run_v2.py` (soma dos trechos quando a execução foi retomada do `.parcial.csv`); (b) **tempo de ajuste por janela** (`exec_time_s` dos caches: relógio de parede de cada ajuste dentro do processo). Os dois dependem da máquina e do paralelismo (`n_jobs` registrado no log); comparações relativas entre modelos são as mais confiáveis.
'''))

cells.append(code(r'''
LOGS = {'M4': ['m4_limpo_rapidos.log', 'm4_limpo_sarima_auto.log'], 'Car Parts': ['cp.log'], 'Tourism': ['tm.log']}
def tempos_log(arqs):
    vals, nj = {}, {}
    for a in arqs:
        limpo = True   # só a linha de base limpa entra
        for l in (V2 / 'logs' / a).read_text(encoding='utf-8', errors='replace').splitlines():
            if 'etapa=' in l:
                limpo = 'etapa=limpo' in l and 'alvo_ruidoso=True' not in l
            h = re.search(r'modelos=\[(.*?)\] n_jobs=(\d+)', l)
            if h:
                if limpo:
                    for m in re.findall(r"'(\w+)'", h.group(1)):
                        nj[m] = int(h.group(2))
                continue
            g = re.match(r'\[(\w+)\] (?:\d+/\d+ \S+ \| ([\d.]+) min|\S+ em ([\d.]+) min)', l)
            if limpo and g and g.group(1) in pv.MODELOS:
                vals.setdefault(g.group(1), []).append(float(g.group(2) or g.group(3)))
    out = {}
    for m, vs in vals.items():
        total, ant, trechos = 0.0, 0.0, 1
        for v in vs:
            if v < ant:          # contador reiniciado: execução retomada
                total += ant; trechos += 1
            ant = v
        out[m] = {'parede_min': round(total + ant, 1), 'trechos': trechos, 'n_jobs': nj.get(m)}
    return pd.DataFrame(out).T

TEMPO = {}
for b, arqs in LOGS.items():
    t = tempos_log(arqs)
    t['ajuste_mediano_s'] = JAN[b].groupby('model')['exec_time_s'].median()
    t['ajuste_somado_min'] = JAN[b].groupby('model')['exec_time_s'].sum() / 60
    TEMPO[b] = t.reindex(MODS)
tempo = pd.concat(TEMPO, names=['base', 'model']).reset_index()
tempo['modelo'] = tempo['model'].map(NOMES)
tempo = tempo[['base', 'modelo', 'parede_min', 'trechos', 'n_jobs', 'ajuste_mediano_s', 'ajuste_somado_min']]
tempo.to_csv(F5 / 'custo_computacional_v2.csv', index=False)
print(tempo.round(3).to_string(index=False))
for b in TEMPO:
    t = TEMPO[b]
    print(f"{b:<9}: parede total {t.parede_min.sum():.1f} min (SARIMA automático {t.loc['sarima_auto', 'parede_min']:.1f} min); "
          f"ajuste somado {t.ajuste_somado_min.sum() / 60:.2f} h")
'''))

# ---------------------------------------------------------------- Tabela 4
cells.append(md(r'''
## 3. Tabela 4: mapa síntese (antigo x v2, com Tourism ao lado)
- **Robustez**: categorias do v2 com os limiares declarados no notebook 08 v2 (definidos depois de ver o M4) sobre a média do Δ% relativo de cada tipo de cenário; entre parênteses, o Δ% com sinal; ao lado, quantos níveis do tipo têm IC excluindo zero. Mesmos limiares no Tourism. SARIMA automático fora dos cenários (restrição operacional).
- **Desempenho limpo**: o v2 traz o rank (MAPE) com IC, em vez do rótulo qualitativo (a dissertação não declara a regra do rótulo).
- **Explicabilidade**: só RF, GB e SVR têm medição (estabilidade SHAP e concordância SHAP x LIME no M4); os demais ficam N.A.
- **Custo**: só entra com tempo medido no v2 (seção 2); "não remedido" quando não há medição.
'''))

cells.append(code(r'''
NOME_T4 = {'Holt-Winters': 'Holt-Winters', 'SES': 'SES', 'SARIMA auto.': 'SARIMA automático', 'SARIMA heur.': 'SARIMA heurístico',
           'Gradient Boost.': 'Gradient Boosting', 'Random Forest': 'Random Forest', 'SVR': 'SVR', 'Persistência': 'Persistência',
           'Sazonal ingênuo': 'Sazonal ingênuo', 'Média histórica': 'Média histórica'}
a4 = ANT[4].assign(Algoritmo=ANT[4]['Algoritmo'].map(NOME_T4)).set_index('Algoritmo')
assert a4.index.notna().all()
TIPOS = {'Ruído': 'ruido', 'Ausências': 'ausencia', 'Outliers': 'outlier', 'Truncamento': 'trunca'}
COL_ANT = {'Ruído': 'Rob. ruído', 'Ausências': 'Rob. ausências', 'Outliers': 'Rob. outliers', 'Truncamento': 'Rob. truncamento'}
cat_m4 = pd.read_csv(V2 / 'm4' / 'adversos' / 'comparacao_tabela4_robustez.csv').set_index('Modelo')
cat_tm = pd.read_csv(V2 / 'tm' / 'adversos' / 'tm_adversos_categorias.csv', index_col=0)
def ic_niveis(arq):
    t = pd.read_csv(arq)
    t['tipo'] = t['cenario'].map(lambda c: pv.CENARIOS[c][0])
    g = t.groupby(['model', 'tipo'])
    return (g['ic_inclui_zero'].apply(lambda x: f'{int((~x.astype(bool)).sum())} de {len(x)}')).unstack()
icn_m4 = ic_niveis(V2 / 'm4' / 'adversos' / 'm4_adversos_delta_ic95.csv')
icn_tm = ic_niveis(V2 / 'tm' / 'adversos' / 'tm_adversos_delta_ic95.csv')
est = pd.read_csv(V2 / 'm4' / 'xai' / 'xai_m4_estabilidade.csv'); cc = pd.read_csv(V2 / 'm4' / 'xai' / 'xai_m4_concordancia_shap_lime.csv')
XAI_COD = {'random_forest': 'RandomForest', 'gbt': 'GradientBoosting', 'svr': 'SVR'}
rk_tm, _ = pv.bootstrap_ranks(BS['Tourism'], MODS, 'mase')
linhas = []
for m in MODS:
    n = NOMES[m]
    lin = {'Algoritmo': n, 'Família': pv.FAMILIA[m],
           'Desemp. limpo antigo': a4.loc[n, 'Desemp. limpo'],
           'Desemp. limpo v2 M4 (rank MAPE [IC])': f"{int(rk_mape.loc[m, 'rank'])} {ic(rk_mape, m, 'rank')}",
           'Desemp. limpo v2 Tourism (rank MASE [IC])': f"{int(rk_tm.loc[m, 'rank'])} {ic(rk_tm, m, 'rank')}"}
    for tipo, cod in TIPOS.items():
        lin[f'{tipo} antigo'] = a4.loc[n, COL_ANT[tipo]]
        if m == 'sarima_auto':
            lin[f'{tipo} v2 M4'] = lin[f'{tipo} v2 Tourism'] = 'não avaliado (restrição operacional)'
            continue
        lin[f'{tipo} v2 M4'] = f"{cat_m4.loc[n, f'{tipo} v2']}; IC sem zero em {icn_m4.loc[m, cod]} níveis"
        lin[f'{tipo} v2 Tourism'] = f"{cat_tm.loc[n, tipo]}; IC sem zero em {icn_tm.loc[m, cod]} níveis"
    lin['Explicab. antigo'] = a4.loc[n, 'Explicab.']
    if m in XAI_COD:
        e = est[(est.modelo == XAI_COD[m]) & (est.metodo == 'shap')].iloc[0]; c = cc[(cc.modelo == XAI_COD[m]) & (cc.condicao == 'limpo')].iloc[0]
        lin['Explicab. v2 M4'] = f"estab. SHAP {e.spearman}; Pearson SHAP×LIME {c.pearson_global:.3f}; Spearman por série {c.spearman_por_serie}"
    else:
        lin['Explicab. v2 M4'] = 'N.A. (sem medição; modelo sem explicação pós-hoc no protocolo)'
    lin['Custo antigo'] = a4.loc[n, 'Custo comp.']
    tm4 = TEMPO['M4'].loc[m]
    lin['Custo v2 M4 (medido)'] = ('não remedido' if pd.isna(tm4.ajuste_mediano_s) or pd.isna(tm4.parede_min) else
                                   f"{tm4.ajuste_mediano_s:.3f} s por janela (mediana); {tm4.parede_min:.1f} min de parede para 200 séries (n_jobs={int(tm4.n_jobs)})")
    linhas.append(lin)
t4 = pd.DataFrame(linhas)
t4.to_csv(F5 / 'tabela4_antigo_v2.csv', index=False)
print(t4[['Algoritmo', 'Desemp. limpo antigo', 'Desemp. limpo v2 M4 (rank MAPE [IC])'] +
         [c for t in TIPOS for c in (f'{t} antigo', f'{t} v2 M4')]].to_string(index=False))
print(); print(t4[['Algoritmo', 'Explicab. antigo', 'Explicab. v2 M4', 'Custo antigo', 'Custo v2 M4 (medido)']].to_string(index=False))
'''))

# ---------------------------------------------------------------- Tabela 5
cells.append(md(r'''
## 4. Tabela 5: recomendações (antigo x v2, Tourism ao lado)
Cada recomendação vira um critério verificável (os cinco primeiros são os do notebook 08 v2; as duas recomendações que faltavam, restrição operacional e explicabilidade, ganham critério aqui). Os critérios foram escritos depois de ver o M4; no Tourism são aplicados sem mudança e **a posteriori**. No Car Parts, só onde não dependem de cenários adversos (decisão 3A).
'''))

cells.append(code(r'''
hier = {'M4': pd.read_csv(V2 / 'm4' / 'adversos' / 'm4_adversos_hierarquia_robustez.csv', index_col=0),
        'Tourism': pd.read_csv(V2 / 'tm' / 'adversos' / 'tm_adversos_hierarquia_robustez.csv', index_col=0)}
longo = {'M4': pd.read_csv(V2 / 'm4' / 'adversos' / 'm4_adversos_delta_longo.csv'),
         'Tourism': pd.read_csv(V2 / 'tm' / 'adversos' / 'tm_adversos_delta_longo.csv')}
xai_est = {b: pd.read_csv(V2 / s / 'xai' / f'xai_{s}_estabilidade.csv') for b, s in [('M4', 'm4'), ('Tourism', 'tm'), ('Car Parts', 'cp')]}
xai_cc = {b: pd.read_csv(V2 / s / 'xai' / f'xai_{s}_concordancia_shap_lime.csv') for b, s in [('M4', 'm4'), ('Tourism', 'tm'), ('Car Parts', 'cp')]}
f0 = lambda s: float(str(s).split()[0])

def r1(b):
    met = 'mae' if b == 'Car Parts' else 'mape'
    return NOMES[pv.resumo_modelos(BS[b], met)[met].idxmin()]
def r2(b): return hier[b]['Ruído Δ%'].idxmin()
def r3(b): return hier[b].loc[['Random Forest', 'Gradient Boosting', 'SVR'], 'Outliers Δ%'].idxmin()
def r4(b):
    l = longo[b][longo[b].cenario.str.startswith('B_')]
    return NOMES[l.groupby('model')['mape_adv'].mean().idxmin()]
def r5(b):
    t = TEMPO[b]['ajuste_mediano_s']
    outros = t.drop(TRIV + ['ses', 'holt_winters'])
    rapidos = t[['ses', 'holt_winters']].max() < outros.min()
    sarima_lento = t.drop(TRIV).idxmax() == 'sarima_auto'
    return (f"SES {t['ses']:.3f} s, HW {t['holt_winters']:.3f} s, mais rápido dos demais ajustados {NOMES[outros.idxmin()]} {outros.min():.3f} s; "
            f"mais lento: {NOMES[t.drop(TRIV).idxmax()]} {t.drop(TRIV).max():.2f} s"), bool(rapidos and sarima_lento)
def r6(b):
    e = xai_est[b][xai_est[b].metodo == 'shap'].set_index('modelo')['spearman'].map(f0)
    c = xai_cc[b][xai_cc[b].condicao == 'limpo'].set_index('modelo')['pearson_global']
    ok = e.idxmax() == 'RandomForest' and c.idxmax() == 'RandomForest' and c.max() > c.drop('RandomForest').max()
    return (f"estab. SHAP: " + ', '.join(f'{k} {v:.3f}' for k, v in e.items()) + "; Pearson: " + ', '.join(f'{k} {v:.3f}' for k, v in c.items())), bool(ok)
def r7(b):
    h = hier[b]
    ok = h.loc['SES', 'Ruído Δp.p.'] < h.loc['Holt-Winters', 'Ruído Δp.p.'] and h.loc['SES', 'Outliers Δp.p.'] < h.loc['Holt-Winters', 'Outliers Δp.p.']
    return (f"SES ruído {h.loc['SES', 'Ruído Δp.p.']:+.2f} p.p. x HW {h.loc['Holt-Winters', 'Ruído Δp.p.']:+.2f}; "
            f"outliers {h.loc['SES', 'Outliers Δp.p.']:+.2f} x {h.loc['Holt-Winters', 'Outliers Δp.p.']:+.2f}"), bool(ok)

a5 = ANT[5]
REG = [  # (linha da Tabela 5, critério, esperado, função, bases)
    (0, 'menor MAPE limpo', {'Holt-Winters'}, r1, ['M4', 'Tourism']),
    (1, 'menor Δ% médio sob ruído (todos os 9 modelos)', {'SES'}, r2, ['M4', 'Tourism']),
    (2, 'menor Δ% médio sob outliers entre RF, GB e SVR', {'SVR'}, r3, ['M4', 'Tourism']),
    (3, 'menor MAPE mediano sob ausências (média de 10, 20 e 30%)', {'Gradient Boosting', 'Persistência'}, r4, ['M4', 'Tourism']),
    (4, 'SES e HW com tempo de ajuste por janela menor que o de todos os demais modelos ajustados, e o SARIMA automático o mais lento', None, r5, ['M4', 'Tourism', 'Car Parts']),
    (5, 'RF com a maior estabilidade SHAP e a maior concordância SHAP×LIME (Pearson) entre RF, GB e SVR', None, r6, ['M4', 'Tourism', 'Car Parts']),
    (6, 'SES degrada menos que HW sob ruído e sob outliers (Δ p.p.)', None, r7, ['M4', 'Tourism']),
]
linhas = []
for i, crit, esperado, fn, bases in REG:
    lin = {'Perfil (Tabela 5)': a5.iloc[i, 0], 'Adversidade': a5.iloc[i, 1], 'Recomendação antiga': a5.iloc[i, 2],
           'Justificativa antiga': a5.iloc[i, 3], 'Critério verificável': crit}
    for b in ['M4', 'Tourism', 'Car Parts']:
        if b not in bases:
            lin[f'{b}: resultado'] = ('não se aplica (perfil sem sazonalidade regular; menor MAE: ' + r1(b) + ')' if i == 0 else
                                      'não se aplica (sem cenários adversos, decisão 3A)')
            lin[f'{b}: sustenta?'] = ''
            continue
        out = fn(b)
        if esperado is not None:
            lin[f'{b}: resultado'], lin[f'{b}: sustenta?'] = out, 'SIM' if out in esperado else 'NÃO'
        else:
            lin[f'{b}: resultado'], lin[f'{b}: sustenta?'] = out[0], 'SIM' if out[1] else 'NÃO'
    linhas.append(lin)
t5 = pd.DataFrame(linhas)
t5['Nota'] = ['', 'diferenças pequenas entre os primeiros colocados; ver ICs em m4/ e tm/adversos/*_delta_ic95.csv', 'ICs sobrepostos no M4',
              'o "benefício paradoxal" vinha do teste interpolado (C2)',
              'o critério exige as duas partes: SES é o mais rápido dos ajustados e o SARIMA automático o mais lento, mas o HW é mais lento que SVR e GB; tempos dependem da máquina e do paralelismo (seção 2)',
              'Car Parts: XAI exploratória', 'no Tourism o SES é o 8º em dados limpos']
ant5 = pd.read_csv(V2 / 'm4' / 'adversos' / 'comparacao_tabela5_recomendacoes.csv')
assert list(ant5['Sustenta?']) == list(t5.loc[[0, 1, 2, 3, 6], 'M4: sustenta?']), 'divergência com o notebook 08 v2'
t5.to_csv(F5 / 'tabela5_antigo_v2.csv', index=False)
print(t5[['Recomendação antiga', 'Critério verificável', 'M4: resultado', 'M4: sustenta?', 'Tourism: resultado', 'Tourism: sustenta?',
          'Car Parts: sustenta?']].to_string(index=False))
'''))

# ---------------------------------------------------------------- Tabela 6
cells.append(md(r'''
## 5. Tabela 6: características das bases (antigo x v2, três bases)
Valores antigos: Tabela 6 do `.docx` (M4 e Car Parts). Caracterização v2: `results/tourism/caracterizacao_bases.csv` (F_T e F_S por STL; piso de ruído de F_S por simulação, NB11 Etapa 1). Tempo: linha de base limpa com os 10 modelos, **incluindo o SARIMA automático**; depende da máquina e do paralelismo.
'''))

cells.append(code(r'''
car = pd.read_csv(OLD / 'tourism' / 'caracterizacao_bases.csv').set_index('Base')
CB = {'M4': car.loc['M4 Monthly'], 'Car Parts': car.loc['Car Parts'], 'Tourism': car.loc['Tourism Monthly']}
a6 = ANT[6].set_index('Característica')
def comp(b):
    c = CB[b]; return f"{int(c['Comprimento mín.'])} a {int(c['Comprimento máx.'])} obs. (mediana {c['Comprimento mediano']:.0f})"
def zeros(b):
    c = CB[b]; return f"mediana {c['Zeros (mediana, %)']:.1f}%; {c['Séries com zeros (%)']:.1f}% das séries com algum zero"
def saz(b):
    c = CB[b]; return (f"F_S mediana {c['F_S mediana']:.2f}; F_S > 0,5 em {c['F_S > 0,5 (%)']:.0f}% das séries; "
                       f"acima do piso de ruído (p95) em {c['F_S acima do piso p95 (%)']:.0f}%")
def tempo_txt(b):
    t = TEMPO[b]; tot, sa = t.parede_min.sum(), t.loc['sarima_auto', 'parede_min']
    extra = ' (SARIMA automático em execução retomada; soma dos trechos)' if t.loc['sarima_auto', 'trechos'] > 1 else ''
    nj = f"n_jobs {sorted(set(t.drop('sarima_auto').n_jobs.dropna().astype(int)))} nos demais modelos; SARIMA automático " +          (f"n_jobs {int(t.loc['sarima_auto', 'n_jobs'])}" if pd.notna(t.loc['sarima_auto', 'n_jobs']) else 'com n_jobs não registrado no log')
    return f"{tot:.1f} min de parede, dos quais {sa:.1f} min do SARIMA automático{extra}; {nj}"
mape_nan_cp = 100 * JAN['Car Parts'].loc[JAN['Car Parts'].status.isin([pv.OK, pv.FB_SNAIVE, pv.FB_MEAN]), 'mape'].isna().mean()
linhas = [
    ('Número de séries', '200 (amostra SEED=42 de 48.000)', f"{int(CB['Car Parts']['Séries analisadas'])} (base completa)",
     '200 (amostra SEED=42 de 366); sensibilidade com as 366', ''),
    ('Comprimento por série', comp('M4'), comp('Car Parts'), comp('Tourism'), 'antigo do M4 (42 a 126) era erro de texto'),
    ('Horizonte H', '18', '18', '24', ''),
    ('Natureza das séries', a6.loc['Natureza das séries', 'M4 mensal (NB 02–07)'], a6.loc['Natureza das séries', 'Car Parts (NB 10)'],
     'Demanda turística mensal (Tourism Forecasting Competition)', ''),
    ('Proporção de zeros', zeros('M4'), zeros('Car Parts'), zeros('Tourism'), ''),
    ('Tendência (F_T mediana)', f"{CB['M4']['F_T mediana']:.2f}", f"{CB['Car Parts']['F_T mediana']:.2f}", f"{CB['Tourism']['F_T mediana']:.2f}", ''),
    ('Sazonalidade', saz('M4'), saz('Car Parts'), saz('Tourism'), 'no Car Parts, F_S quase não supera o piso de ruído'),
    ('Métrica principal', 'MAPE; MASE como métrica comum', f'MAE; MASE como métrica comum; MAPE indefinido em {mape_nan_cp:.1f}% das janelas',
     'MAPE; MASE como métrica comum', ''),
    ('Perfil de adversidade', '9 cenários controlados (perturbação só no treino, teste limpo)', 'sem cenários adversos (decisão 3A); esparsidade natural',
     '9 cenários controlados (mesmo protocolo do M4)', ''),
    ('Modelos', '10 (inclui SARIMA automático)', '10 (inclui SARIMA automático)', '10 (inclui SARIMA automático)', 'antigo: 9 no Car Parts'),
    ('Tempo de execução (linha de base limpa)', tempo_txt('M4'), tempo_txt('Car Parts'), tempo_txt('Tourism'),
     'depende da máquina e do paralelismo; ver custo_computacional_v2.csv'),
]
t6 = pd.DataFrame(linhas, columns=['Característica', 'M4 v2', 'Car Parts v2', 'Tourism v2', 'Nota'])
MAP6 = {'Número de séries': 'Número de séries', 'Comprimento por série': 'Comprimento por série', 'Natureza das séries': 'Natureza das séries',
        'Proporção de zeros': 'Proporção de zeros', 'Métrica principal': 'Métrica principal', 'Sazonalidade': 'Sazonalidade',
        'Perfil de adversidade': 'Perfil de adversidade', 'Tempo de execução (linha de base limpa)': 'Tempo de execução (9 mod.)'}
t6.insert(1, 'M4 antigo', t6['Característica'].map(lambda k: a6.loc[MAP6[k], 'M4 mensal (NB 02–07)'] if k in MAP6 else ''))
t6.insert(3, 'Car Parts antigo', t6['Característica'].map(lambda k: a6.loc[MAP6[k], 'Car Parts (NB 10)'] if k in MAP6 else ''))
t6.to_csv(F5 / 'tabela6_antigo_v2.csv', index=False)
print(t6.to_string(index=False))
'''))

# ---------------------------------------------------------------- Tabela 7
cells.append(md(r'''
## 6. Tabela 7: Car Parts (antigo x v2)
Valores antigos: Tabela 7 do `.docx`. Rank por MAE com IC (bootstrap pareado por série); SVR e SARIMA com e sem fallback, SARIMA sem explosivas; MASE acrescentado. Cobertura do MAPE: proporção de janelas e de séries (o antigo rotulava como janelas o que era séries).
'''))

cells.append(code(r'''
d = JAN['Car Parts']; bs = BS['Car Parts']
r = pv.resumo_modelos(bs, 'mae')
r_sf = pv.resumo_modelos(pv.agregar_por_serie(d, incluir_fallback=False), 'mae')
r_se = pv.resumo_modelos(pv.agregar_por_serie(d, excluir_explosivas=True), 'mae')
rk_mae, _ = pv.bootstrap_ranks(bs, MODS, 'mae'); rk_mase_cp, _ = pv.bootstrap_ranks(bs, MODS, 'mase')
v2_7 = pd.read_csv(V2 / 'cp' / 'limpo' / 'tabela7_v2_cp.csv', index_col=0)
a7 = ANT[7].set_index('Algoritmo')
fb = 100 * pv.taxas_status(d)[[pv.FB_SNAIVE, pv.FB_MEAN]].sum(axis=1)
linhas = []
for m in MODS:
    n = NOMES[m]; tem = n in a7.index
    g = lambda c: num(a7.loc[n, c]) if tem else np.nan
    linhas.append({'Algoritmo': n, 'MAE antigo': g('MAE med.'), 'MAE v2': round(r.loc[m, 'mae'], 3), 'IC MAE v2': ic(rk_mae, m, 'mediana'),
                   'Rank antigo': int(a7.index.get_loc(n) + 1) if tem else np.nan, 'Rank v2 (MAE)': int(rk_mae.loc[m, 'rank']),
                   'IC rank v2': ic(rk_mae, m, 'rank'),
                   'MAE v2 sem fallback': round(r_sf.loc[m, 'mae'], 3), 'MAE v2 sem explosivas': round(r_se.loc[m, 'mae'], 3),
                   'Fallback v2 (% jan.)': round(fb[m], 1),
                   'MAPE antigo': g('MAPE med. (%)'), 'MAPE v2': round(r.loc[m, 'mape'], 1),
                   'R² antigo': g('R² med.'), 'R² v2': round(r.loc[m, 'r2'], 3),
                   'Cob. MAPE antigo (%)': g('Cob. MAPE (%)'), 'Cob. MAPE v2 janelas (%)': v2_7.loc[n, 'Cob. MAPE janelas (%)'],
                   'Cob. MAPE v2 séries (%)': v2_7.loc[n, 'Cob. MAPE séries (%)'],
                   'MAE Baixo antigo': g('MAE Baixo'), 'MAE Baixo v2': v2_7.loc[n, 'MAE Baixo'],
                   'MAE Alto antigo': g('MAE Alto'), 'MAE Alto v2': v2_7.loc[n, 'MAE Alto'],
                   'MAE Extremo antigo': g('MAE Extremo'), 'MAE Extremo v2': v2_7.loc[n, 'MAE Extremo'],
                   'MASE v2': round(r.loc[m, 'mase'], 3), 'Rank v2 (MASE)': f"{int(rk_mase_cp.loc[m, 'rank'])} {ic(rk_mase_cp, m, 'rank')}"})
t7 = pd.DataFrame(linhas).sort_values('Rank v2 (MAE)')
assert np.allclose(t7.set_index('Algoritmo')['MAE v2'], v2_7.loc[t7.Algoritmo, 'MAE mediano'], atol=0.0006)
t7['Mudou de posição'] = np.where(t7['Rank antigo'].notna() & (t7['Rank antigo'] != t7['Rank v2 (MAE)']), 'SIM', '')
t7.to_csv(F5 / 'tabela7_antigo_v2_cp.csv', index=False)
print(t7.to_string(index=False))
print('\nRank antigo = posição na Tabela 7 da dissertação (9 modelos, sem o SARIMA automático).')
'''))

# ---------------------------------------------------------------- figuras: utilitários
cells.append(md(r'''
## 7. Figuras de resultados (Figs. 8 a 26), refeitas no v2
Cada figura v2 é gravada como `fase5/figuras/figNN_v2_<nome>.png`, com os dados em `..._dados.csv`. Quando o PNG antigo existe em `results/`, também é gravada `figNN_antiga_x_v2.png` (antiga à esquerda, v2 à direita). O índice `figuras_indice.csv` traz o valor-chave antigo e o v2 de cada figura.
'''))

cells.append(code(r'''
ANTIGAS = {8: 'fig_boxplot_comparacao.png', 9: 'fig_mape_acumulado_COMPLETO.png', 10: 'fig_exemplo_4modelos_M32652.png',
           11: 'fig_ganho_holt_winters.png', 12: 'fig_gbt_vs_rf_importancia.png',
           13: 'adversos/fig1_heatmap_delta_mape.png', 14: 'adversos/fig4_taxa_sucesso.png', 15: 'adversos/fig2_boxplot_mape_cenarios.png',
           16: 'adversos/fig3_perfil_robustez.png', 17: 'explicabilidade/fig1_shap_global.png', 18: 'explicabilidade/fig2_shap_consistencia.png',
           19: 'explicabilidade/fig3_lime_global.png', 20: 'explicabilidade/fig4_shap_vs_lime.png', 21: 'explicabilidade/fig5_estabilidade_shap.png',
           22: 'explicabilidade/fig6_shap_clean_vs_pert.png', 23: 'car_parts/fig_boxplot_cp_mape.png', 24: 'car_parts/fig_boxplot_cp_mae.png',
           25: 'car_parts/fig_comparacao_3metricas_cp.png', 26: 'car_parts/fig_exemplos_cp.png'}
CORRESP_PROVAVEL = {9}   # identificada pelo conteúdo, não pelo nome do arquivo
FIG_REG = []

def salvar(fig, n, nome, titulo, chave_ant, chave_v2, dados=None, nota=''):
    p = FIG / f'fig{n:02d}_v2_{nome}.png'
    fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
    if dados is not None:
        dados.to_csv(FIG / f'fig{n:02d}_v2_{nome}_dados.csv', index=False)
    ant = OLD / ANTIGAS[n]; lado = ''
    if ant.exists():
        a, b = plt.imread(ant), plt.imread(p)
        h = max(a.shape[0] / a.shape[1], b.shape[0] / b.shape[1])
        f2, ax = plt.subplots(1, 2, figsize=(22, 11 * h + 1))
        for x, img, t in [(ax[0], a, f'Dissertação (antiga): {ANTIGAS[n]}'), (ax[1], b, 'Protocolo v2')]:
            x.imshow(img); x.axis('off'); x.set_title(t, fontsize=11)
        f2.suptitle(f'Fig. {n}: {titulo}', fontsize=12)
        lado = FIG / f'fig{n:02d}_antiga_x_v2.png'
        f2.savefig(lado, dpi=110, bbox_inches='tight'); plt.close(f2)
    FIG_REG.append({'figura': n, 'título (dissertação)': titulo, 'arquivo antigo': ANTIGAS[n] if ant.exists() else 'não encontrado',
                    'correspondência': 'provável (pelo conteúdo)' if n in CORRESP_PROVAVEL else 'pelo nome do arquivo',
                    'arquivo v2': p.name, 'lado a lado': lado.name if lado else '', 'valor-chave antigo': chave_ant,
                    'valor-chave v2': chave_v2, 'nota': nota})
    print(f'Fig. {n:>2}: {p.name}' + (f' | {lado.name}' if lado else ''))

def boxplot_modelos(ax, bs, modelos, met, clip=None, anot=True, fmt='{:.2f}'):
    dados = [bs[bs.model == m][met].dropna() for m in modelos]
    bp = ax.boxplot([x.clip(upper=clip) if clip else x for x in dados], patch_artist=True,
                    medianprops=dict(color='black', linewidth=2), flierprops=dict(marker='o', markersize=2.5, alpha=0.35))
    for patch, m in zip(bp['boxes'], modelos):
        patch.set_facecolor(CORES_FAM[pv.FAMILIA[m]]); patch.set_alpha(0.75)
    if anot:
        for i, x in enumerate(dados):
            ax.text(i + 1, x.median(), fmt.format(x.median()), ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(range(1, len(modelos) + 1)); ax.set_xticklabels([NOMES[m] for m in modelos], rotation=30, ha='right', fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
'''))

# ---------------------------------------------------------------- Figs 8-12
cells.append(md(r'''
### 7.1 M4, dados limpos (Figs. 8 a 12)
'''))

cells.append(code(r'''
bs = BS['M4']; d = JAN['M4']
# Fig. 8: baselines + HW, MAPE, MAE e R²
SEL8 = ['persistence', 'mean', 'seasonal_naive', 'holt_winters']
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, (met, rot) in zip(axes, [('mape', 'MAPE (%)'), ('mae', 'MAE'), ('r2', 'R²')]):
    boxplot_modelos(ax, bs, SEL8, met); ax.set_title(f'{rot} por modelo', fontsize=10)
fig.suptitle('M4, v2: baselines e Holt-Winters (200 séries, amostra única, H=18)', fontsize=11)
med8 = bs[bs.model.isin(SEL8)].groupby('model')[['mape', 'mae', 'r2']].median().rename(index=NOMES).round(2)
salvar(fig, 8, 'm4_baselines_hw', 'Distribuição do MAPE, MAE e R² para os baselines e o Holt-Winters (M4)',
       'MAPE: Persistência 8,70; Média 18,60; Sazonal ingênuo 11,06; HW 8,26',
       'MAPE: ' + '; '.join(f'{k} {v:.2f}' for k, v in med8['mape'].items()), med8.reset_index())

# Fig. 9: 10 modelos (MAPE e MASE)
ordem9 = list(rk_mape.sort_values('rank').index)
fig, axes = plt.subplots(2, 1, figsize=(15, 10))
boxplot_modelos(axes[0], bs, ordem9, 'mape', clip=100); axes[0].set_ylabel('MAPE (%) por série, truncado em 100')
boxplot_modelos(axes[1], bs, ordem9, 'mase', clip=6, fmt='{:.3f}'); axes[1].set_ylabel('MASE por série, truncado em 6')
for f_, c in CORES_FAM.items():
    axes[0].bar(0, 0, color=c, alpha=0.75, label=f_)
axes[0].legend(fontsize=8)
fig.suptitle('M4, v2: MAPE e MASE dos 10 modelos (ordenados pelo MAPE mediano)', fontsize=11)
med9 = bs.groupby('model')[['mape', 'mase']].median().loc[ordem9].rename(index=NOMES).round(3)
salvar(fig, 9, 'm4_10_modelos', 'Distribuição completa do MAPE para os dez algoritmos (M4)',
       'HW 8,26 (1º) ... Média histórica 18,60 (10º); Persistência 2ª', 'HW 8,36 (1º) ... Média histórica 21,82 (10º); Persistência 6ª',
       med9.reset_index())

# Fig. 10: série de exemplo; M32652 não está na amostra v2
ex10 = 'M32652' if 'M32652' in S_M4 else ids_m4[0]
s = np.asarray(S_M4[ex10], float); o = pv.make_rolling_windows(len(s), 18)[-1]
tr, te = s[:o], s[o:o + 18]
fig, ax = plt.subplots(figsize=(14, 5)); x0 = max(0, o - 60)
ax.plot(range(x0, o), tr[x0:], color='#4682B4', label='treino'); ax.plot(range(o, o + 18), te, color='black', lw=2, label='real (teste)')
dados10 = []
for m, c in zip(SEL8, ['#E8834E', '#8BC34A', '#9C27B0', '#2196F3']):
    p_ = np.asarray(pv.MODELOS[m](tr, 18)[0], float); mt = pv.calc_metrics(te, p_)
    ax.plot(range(o, o + 18), p_, color=c, label=f"{NOMES[m]} (MAPE {mt['mape']:.1f}%)")
    dados10 += [{'modelo': NOMES[m], 'passo': k + 1, 'previsao': v, 'real': te[k]} for k, v in enumerate(p_)]
ax.axvline(o - 0.5, ls=':', color='grey'); ax.legend(fontsize=8); ax.set_title(f'M4, v2: série {ex10}, última janela (origem {o})', fontsize=11)
salvar(fig, 10, f'm4_exemplo_{ex10}', 'Série M32652: previsões de persistência, média, sazonal ingênuo e HW na última janela',
       'série M32652 (origem 300)', f'série {ex10} (origem {o})', pd.DataFrame(dados10),
       nota=('M32652 não está na amostra v2 (vinha da amostra alfabética antiga); substituída pela 1ª série da amostra v2 '
             '(regra fixada antes de gerar a figura)') if ex10 != 'M32652' else '')

# Fig. 11: ganho do HW sobre os baselines
P = bs.pivot(index='unique_id', columns='model', values='mape')
lin11, dist = [], []
for b in ['persistence', 'seasonal_naive', 'mean']:
    rel = 100 * (P[b] - P['holt_winters']) / P[b]
    lin11.append({'baseline': NOMES[b], 'redução relativa por série, mediana (%)': round(rel.median(), 2),
                  'diferença das medianas (p.p.)': round(P[b].median() - P['holt_winters'].median(), 2),
                  'redução relativa das medianas (%)': round(100 * (P[b].median() - P['holt_winters'].median()) / P[b].median(), 2),
                  'séries em que o HW é melhor (%)': round(100 * (rel > 0).mean(), 1)})
    dist.append(rel.clip(-200, 100))
t11 = pd.DataFrame(lin11)
fig, ax = plt.subplots(figsize=(10, 5))
ax.boxplot(dist, patch_artist=True, medianprops=dict(color='black', lw=2)); ax.axhline(0, color='grey', ls='--')
ax.set_xticks([1, 2, 3]); ax.set_xticklabels(t11['baseline']); ax.set_ylabel('redução relativa do MAPE pelo HW (%), por série')
for i, v in enumerate(t11['redução relativa por série, mediana (%)']):
    ax.text(i + 1, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('M4, v2: ganho relativo do Holt-Winters sobre os baselines (200 séries; truncado em [-200; 100])', fontsize=10)
pers = t11.iloc[0]
salvar(fig, 11, 'm4_ganho_hw', 'Ganho relativo do Holt-Winters sobre os baselines (redução de MAPE)',
       'ganho sobre a Persistência de "1,1 p.p." (era % relativo; 0,06 p.p. na amostra antiga)',
       f"sobre a Persistência: diferença das medianas {pers['diferença das medianas (p.p.)']:.2f} p.p. = "
       f"{pers['redução relativa das medianas (%)']:.1f}% relativo; mediana por série {pers['redução relativa por série, mediana (%)']:.1f}%",
       t11, nota='rótulo de unidade corrigido: % relativo e p.p. reportados separadamente')
print(t11.to_string(index=False))

# Fig. 12: importância nativa (impureza) de GB e RF, última janela de M19247
ex12 = 'M19247' if 'M19247' in S_M4 else ids_m4[0]
fit = pv.treinar_modelos_xai(np.asarray(S_M4[ex12], float)[:-18])
fig, axes = plt.subplots(1, 2, figsize=(15, 5)); dados12 = []
for ax, nome, cor in [(axes[0], 'GradientBoosting', '#E8834E'), (axes[1], 'RandomForest', '#4CAF50')]:
    imp_ = pd.Series(fit[nome]['model'].feature_importances_, index=FN).sort_values(ascending=False)
    top = imp_.iloc[:10][::-1]
    ax.barh(top.index, top.values, color=[('#333333' if f == 'lag_12' else cor) for f in top.index])
    ax.set_title(f'{nome}: importância nativa (redução de impureza)', fontsize=10); ax.set_xlabel('feature_importances_ (soma 1)')
    dados12 += [{'modelo': nome, 'feature': f, 'importancia_impureza': v, 'rank': i + 1} for i, (f, v) in enumerate(imp_.items())]
fig.suptitle(f'M4, v2: série {ex12}, treino da última janela (série[:-18]); barra escura = lag_12 (lag sazonal). '
             'Importância por impureza, diferente do SHAP.', fontsize=10)
d12 = pd.DataFrame(dados12)
ant_gb = pd.read_csv(OLD / 'gbt_feature_importance_example.csv'); ant_rf = pd.read_csv(OLD / 'rf_feature_importance_example.csv')
top3 = lambda t, mod: ', '.join(t[t.modelo == mod].sort_values('rank')['feature'].iloc[:3])
salvar(fig, 12, f'm4_importancia_nativa_{ex12}', 'Importância de características: GB (série M19247) e RF',
       f"GB top-3: {', '.join(ant_gb.feature.iloc[:3])}; RF top-3: {', '.join(ant_rf.feature.iloc[:3])} (17 features, com duplicatas)",
       f"GB top-3: {top3(d12, 'GradientBoosting')}; RF top-3: {top3(d12, 'RandomForest')} (15 features)", d12,
       nota=('legenda: importância nativa baseada em impureza (feature_importances_), diferente do SHAP; '
             'mesma série para GB e RF no v2 (no antigo, o painel do RF era de outra série, rotulada "série anterior" e não identificada)'
             + ('' if ex12 == 'M19247' else '; M19247 fora da amostra v2, substituída pela 1ª série')))
'''))

# ---------------------------------------------------------------- Figs 13-16
cells.append(md(r'''
### 7.2 M4, cenários adversos (Figs. 13 a 16)
'''))

cells.append(code(r'''
CEN = list(pv.CENARIOS)
dic = pd.read_csv(V2 / 'm4' / 'adversos' / 'm4_adversos_delta_ic95.csv')
# Fig. 13: heatmap do Δ% com marca de IC excluindo zero
H = dic.pivot(index='model', columns='cenario', values='delta_pct').loc[RAP, CEN]
Z = dic.pivot(index='model', columns='cenario', values='ic_inclui_zero').loc[RAP, CEN].astype(bool)
fig, ax = plt.subplots(figsize=(14, 6.5))
lim = np.nanmax(np.abs(H.values)); im = ax.imshow(H.values, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        ax.text(j, i, f'{H.iloc[i, j]:+.1f}' + ('' if Z.iloc[i, j] else '*'), ha='center', va='center', fontsize=8)
ax.set_xticks(range(len(CEN))); ax.set_xticklabels([pv.CENARIOS[c][2] for c in CEN], rotation=30, ha='right', fontsize=8)
ax.set_yticks(range(len(RAP))); ax.set_yticklabels([NOMES[m] for m in RAP], fontsize=9)
plt.colorbar(im, ax=ax, label='ΔMAPE relativo (%)')
ax.set_title('M4, v2: ΔMAPE relativo (%) por modelo e cenário; * = IC 95% exclui zero (perturbação só no treino, teste limpo)', fontsize=10)
salvar(fig, 13, 'm4_heatmap_delta', 'Mapa de calor do ΔMAPE relativo (%) por algoritmo e cenário adverso',
       'HW +49,2 (ruído 30%) e +45,7 (outliers 2%), rotulados p.p.; ausências reduzem o MAPE de GB, RF e Persistência',
       f"HW {H.loc['holt_winters', 'A_ruido_30pct']:+.1f}% (ruído 30%) e {H.loc['holt_winters', 'C_outliers_2pct']:+.1f}% (outliers 2%); "
       f"{int((~Z).values.sum())} de {Z.size} combinações com IC excluindo zero", dic)

# Fig. 14: taxa de ajuste sem fallback
st = pd.read_csv(V2 / 'm4' / 'adversos' / 'm4_adversos_status.csv')
S = st.pivot(index='modelo', columns='cenario', values='OK (%)').loc[[NOMES[m] for m in RAP], ['limpo'] + CEN] / 100
fig, ax = plt.subplots(figsize=(14, 6.5)); im = ax.imshow(S.values, cmap='Greens', vmin=0, vmax=1, aspect='auto')
for i in range(S.shape[0]):
    for j in range(S.shape[1]):
        ax.text(j, i, f'{S.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8)
ax.set_xticks(range(S.shape[1])); ax.set_xticklabels(['Limpo'] + [pv.CENARIOS[c][2] for c in CEN], rotation=30, ha='right', fontsize=8)
ax.set_yticks(range(S.shape[0])); ax.set_yticklabels(S.index, fontsize=9); plt.colorbar(im, ax=ax, label='fração de janelas sem fallback')
ax.set_title('M4, v2: fração de janelas com ajuste próprio (1,0 = nenhum fallback); o fallback do SARIMA heurístico é a 1ª janela', fontsize=10)
salvar(fig, 14, 'm4_taxa_sem_fallback', 'Taxa de sucesso do ajuste por algoritmo e cenário adverso',
       'taxas 0,93 e 0,85 atribuídas a "falha de convergência"',
       f"1,00 para todos exceto SARIMA heurístico ({S.loc['SARIMA heurístico'].min():.2f} a {S.loc['SARIMA heurístico'].max():.2f}) e SVR (≥ {S.loc['SVR'].min():.2f})",
       st, nota='no antigo, a taxa refletia séries curtas, não falha de convergência')

# Fig. 15: MAPE por modelo no nível mais forte de cada tipo
FORTE = ['A_ruido_30pct', 'B_ausencias_30pct', 'C_outliers_5pct', 'D_trunca_25pct']
fig, axes = plt.subplots(2, 2, figsize=(17, 10)); d15 = []
for ax, c in zip(axes.flat, FORTE):
    b_ = pv.agregar_por_serie(carregar('m4', c, RAP))
    boxplot_modelos(ax, b_, RAP, 'mape', clip=150); ax.set_title(f'{pv.CENARIOS[c][2]} (MAPE truncado em 150%)', fontsize=10)
    d15.append(b_.groupby('model')['mape'].median().rename(c))
fig.suptitle('M4, v2: MAPE por série no nível mais forte de cada tipo de cenário (teste limpo)', fontsize=11)
d15 = pd.concat(d15, axis=1).rename(index=NOMES).round(2)
salvar(fig, 15, 'm4_mape_cenarios', 'Distribuição do MAPE por algoritmo e tipo de cenário adverso',
       'truncamento sem dados (o cenário nunca rodou)', 'truncamento medido: HW ' + f"{d15.loc['Holt-Winters', 'D_trunca_25pct']:.2f}% com 25% do treino",
       d15.reset_index().rename(columns={'model': 'modelo'}))

# Fig. 16: perfil de robustez com IC
TIPO_CEN = {'Ruído': ['A_ruido_10pct', 'A_ruido_30pct'], 'Ausências': ['B_ausencias_10pct', 'B_ausencias_20pct', 'B_ausencias_30pct'],
            'Outliers': ['C_outliers_2pct', 'C_outliers_5pct'], 'Truncamento (fração mantida)': ['D_trunca_50pct', 'D_trunca_25pct']}
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False); cmap = plt.get_cmap('tab10')
for ax, (t, cs) in zip(axes, TIPO_CEN.items()):
    xs = [0] + [pv.CENARIOS[c][1] for c in cs]
    for k, m in enumerate(RAP):
        sub = dic[dic.model == m].set_index('cenario').loc[cs]
        y = [0] + list(sub.delta_pct); lo = [0] + list(sub.ic_pct_inf); hi = [0] + list(sub.ic_pct_sup)
        ax.plot(range(len(xs)), y, marker='o', color=cmap(k), lw=1.5, label=NOMES[m], ls='--' if m in TRIV else '-')
        ax.fill_between(range(len(xs)), lo, hi, color=cmap(k), alpha=0.08)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(['limpo'] + [pv.CENARIOS[c][2].split()[-1] for c in cs], fontsize=8)
    ax.axhline(0, color='grey', lw=0.8); ax.set_title(t, fontsize=10); ax.grid(alpha=0.3)
axes[0].set_ylabel('ΔMAPE relativo (%) [IC 95%]'); axes[-1].legend(fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
fig.suptitle('M4, v2: perfil de robustez por intensidade (linhas tracejadas = baselines triviais; faixas = IC 95%)', fontsize=11)
salvar(fig, 16, 'm4_perfil_robustez', 'Perfil de robustez: trajetória do ΔMAPE relativo por intensidade',
       'truncamento plano em zero (figura vazia)', 'truncamento medido; HW +4,2/+9,1%, SARIMA heurístico +8,1/+15,9%', dic)
'''))

# ---------------------------------------------------------------- Figs 17-22
cells.append(md(r'''
### 7.3 M4, explicabilidade (Figs. 17 a 22) e reanálise do CV da Questão 1 (item C11)
Importâncias normalizadas por série (soma 1), 30 séries, mesmos 30 pontos para SHAP e LIME. A **reanálise do CV** (Fig. 18) usa os caches da 2c sem rodar XAI de novo: CV entre séries de cada feature (desvio padrão / média, ddof=1), médio sobre as 15 features, com IC por bootstrap sobre as séries; reportado na escala original (como no antigo, que mistura escalas entre séries) e com normalização por série (comparável).
'''))

cells.append(code(r'''
def ler_xai(s, cond):
    return pd.read_csv(V2 / s / 'xai' / f'xai_{s}_{cond}_pontos.csv')
def imp_base(s, normalizar=True):
    out = []
    for cond in ['limpo', 'ruido30']:
        i = pv.importancia_por_serie(ler_xai(s, cond), FN)
        out.append(pv.normalizar_por_serie(i, FN) if normalizar else i)
    return pd.concat(out, ignore_index=True)
IMP = {b: imp_base(s) for b, s in [('M4', 'm4'), ('Car Parts', 'cp'), ('Tourism', 'tm')]}
imp = IMP['M4']; imp_bruto = imp_base('m4', normalizar=False)
glob = lambda i, m, met, cond: i[(i.modelo == m) & (i.metodo == met) & (i.condicao == cond)][FN].mean()

def top10(metodo, n, nome, titulo, ch_ant):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5)); dd, txt = [], []
    for ax, m in zip(axes, MOD_XAI):
        g = glob(imp, m, metodo, 'limpo').sort_values(ascending=False)
        t = g.iloc[:10][::-1]
        ax.barh(t.index, t.values, color=['#333333' if f == 'lag_12' else '#5B8DB8' for f in t.index])
        ax.set_title(m, fontsize=10); ax.set_xlabel(f'importância {metodo.upper()} normalizada (fração por série)')
        dd += [{'modelo': m, 'feature': f, 'importancia': v, 'rank': k + 1} for k, (f, v) in enumerate(g.items())]
        txt.append(f"{m}: lag_12 em {int(g.rank(ascending=False, method='min')['lag_12'])}º")
    fig.suptitle(f'M4, v2: top-10 {metodo.upper()} global (30 séries; barra escura = lag_12, o lag sazonal sem duplicata)', fontsize=11)
    salvar(fig, n, nome, titulo, ch_ant, '; '.join(txt), pd.DataFrame(dd))

top10('shap', 17, 'm4_top10_shap', 'Top-10 características por importância SHAP global (M4)', 'lag sazonal em 7º a 8º no SHAP')
top10('lime', 19, 'm4_top10_lime', 'Top-10 características por importância LIME (M4)', 'lag sazonal no top-3 do LIME (bug de associação por substring)')

# Fig. 18: variabilidade entre séries + CV reanalisado
def cv_medio(X):
    X = np.asarray(X, float); mu = X.mean(axis=0)
    return float(np.nanmean(np.where(mu > 0, X.std(axis=0, ddof=1) / mu, np.nan)))
def cv_ic(X, n=2000):
    X = np.asarray(X, float); idx = np.random.default_rng(pv.SEED).integers(0, len(X), size=(n, len(X)))
    b = [cv_medio(X[k]) for k in idx]
    return cv_medio(X), np.percentile(b, 2.5), np.percentile(b, 97.5)
CV_ANT = {'RandomForest': 1.75, 'GradientBoosting': 1.61, 'SVR': 0.51}
lin_cv = []
for m in MOD_XAI:
    for met in ['shap', 'lime']:
        for esc, dd in [('normalizada por série', imp), ('escala original', imp_bruto)]:
            X = dd[(dd.modelo == m) & (dd.metodo == met) & (dd.condicao == 'limpo')][FN].values
            v, lo, hi = cv_ic(X)
            lin_cv.append({'modelo': m, 'metodo': met, 'escala': esc, 'n_series': len(X), 'CV médio (15 features)': round(v, 3),
                           'IC 95%': f'[{lo:.3f}; {hi:.3f}]',
                           'CV antigo (dissertação, SHAP, 17 features, escala original)': CV_ANT[m] if met == 'shap' else np.nan})
cv = pd.DataFrame(lin_cv)
cv['rótulo'] = 'REANÁLISE (item C11): CV recalculado dos caches da 2c, sem rodar XAI de novo'
cv.to_csv(F5 / 'reanalise_cv_q1_m4.csv', index=False)
print(cv.drop(columns='rótulo').to_string(index=False))
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, m in zip(axes, MOD_XAI):
    sub = imp[(imp.modelo == m) & (imp.metodo == 'shap') & (imp.condicao == 'limpo')]
    t8 = list(sub[FN].mean().sort_values(ascending=False).index[:8])
    ax.boxplot([sub[f] for f in t8], patch_artist=True, medianprops=dict(color='black'))
    ax.set_xticks(range(1, 9)); ax.set_xticklabels(t8, rotation=35, fontsize=8)
    c_ = cv[(cv.modelo == m) & (cv.metodo == 'shap') & (cv.escala == 'normalizada por série')].iloc[0]
    ax.set_title(f"{m}: CV médio (normalizado) {c_['CV médio (15 features)']:.2f} {c_['IC 95%']}", fontsize=9)
axes[0].set_ylabel('importância SHAP normalizada (fração por série)')
fig.suptitle('M4, v2: variabilidade das importâncias SHAP entre as 30 séries (top-8 de cada modelo)', fontsize=11)
cn = cv[(cv.metodo == 'shap') & (cv.escala == 'normalizada por série')].set_index('modelo')['CV médio (15 features)']
cb = cv[(cv.metodo == 'shap') & (cv.escala == 'escala original')].set_index('modelo')['CV médio (15 features)']
salvar(fig, 18, 'm4_variabilidade_shap', 'Variabilidade das importâncias SHAP entre séries (Questão 1)',
       'CV 1,75 / 1,61 / 0,51 (RF/GB/SVR; escala original, 17 features)',
       'CV normalizado ' + ' / '.join(f'{cn[m]:.2f}' for m in MOD_XAI) + '; escala original ' + ' / '.join(f'{cb[m]:.2f}' for m in MOD_XAI),
       cv, nota='reanálise rotulada (item C11); a escala original mistura séries de magnitudes diferentes')

# Fig. 20: SHAP x LIME
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2)); d20 = []
for ax, m in zip(axes, MOD_XAI):
    s_, l_ = glob(imp, m, 'shap', 'limpo'), glob(imp, m, 'lime', 'limpo')
    s_, l_ = s_ / s_.max(), l_ / l_.max(); r_ = np.corrcoef(s_, l_)[0, 1]
    ax.scatter(s_, l_, color=['#333333' if f == 'lag_12' else '#5B8DB8' for f in FN])
    for f in FN:
        ax.annotate(f, (s_[f], l_[f]), fontsize=7, alpha=0.8)
    ax.plot([0, 1], [0, 1], ls='--', color='grey'); ax.set_title(f'{m}: r = {r_:.3f}', fontsize=10)
    ax.set_xlabel('SHAP (normalizado pelo máximo)'); ax.set_ylabel('LIME (normalizado pelo máximo)')
    d20 += [{'modelo': m, 'feature': f, 'shap': s_[f], 'lime': l_[f]} for f in FN]
fig.suptitle('M4, v2: coerência SHAP x LIME por feature (mesmos 30 pontos; LIME por índice)', fontsize=11)
cc_m4 = pd.read_csv(V2 / 'm4' / 'xai' / 'xai_m4_concordancia_shap_lime.csv').query("condicao == 'limpo'").set_index('modelo')
salvar(fig, 20, 'm4_shap_x_lime', 'Coerência entre importâncias SHAP e LIME normalizadas por característica',
       'r = 0,959 / 0,952 / 0,908', 'r = ' + ' / '.join(f"{cc_m4.loc[m, 'pearson_global']:.3f}" for m in MOD_XAI)
       + '; Spearman por série ' + ' / '.join(cc_m4.loc[m, 'spearman_por_serie'].split()[0] for m in MOD_XAI), pd.DataFrame(d20),
       nota='Pearson agregado alto convive com concordância por série bem menor')

# Fig. 21: estabilidade por série
fig, axes = plt.subplots(1, 2, figsize=(13, 5)); d21 = []
for ax, met in zip(axes, ['shap', 'lime']):
    sub = imp[imp.metodo == met]
    e = pv.estabilidade(sub[sub.condicao == 'limpo'], sub[sub.condicao == 'ruido30'], FN)
    ax.boxplot([e[e.modelo == m]['spearman'] for m in MOD_XAI], patch_artist=True, medianprops=dict(color='black'))
    for k, m in enumerate(MOD_XAI):
        v = e[e.modelo == m]['spearman']; ax.text(k + 1, v.median(), f'{v.median():.2f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(MOD_XAI); ax.set_title(met.upper()); ax.axhline(0.8, ls=':', color='grey')
    d21.append(e.assign(metodo=met))
axes[0].set_ylabel('Spearman limpo x ruído σ=30% (por série)')
fig.suptitle('M4, v2: estabilidade das atribuições sob ruído σ=30% (só no treino; semente por série)', fontsize=11)
d21 = pd.concat(d21); md21 = d21[d21.metodo == 'shap'].groupby('modelo')['spearman'].median()
salvar(fig, 21, 'm4_estabilidade', 'Estabilidade das atribuições SHAP sob ruído σ=30%',
       'Spearman mediano 0,572 / 0,344 / 0,520', 'SHAP, mediana ' + ' / '.join(f'{md21[m]:.3f}' for m in MOD_XAI), d21)

# Fig. 22: limpo x ruído, todas as features (média normalizada)
fig, axes = plt.subplots(3, 1, figsize=(15, 12)); d22 = []
for ax, m in zip(axes, MOD_XAI):
    gl, gr = glob(imp, m, 'shap', 'limpo'), glob(imp, m, 'shap', 'ruido30'); x = np.arange(len(FN))
    ax.bar(x - 0.2, gl.values, 0.4, label='limpo', color='#5B8DB8'); ax.bar(x + 0.2, gr.values, 0.4, label='ruído σ=30%', color='#E8834E')
    ax.set_xticks(x); ax.set_xticklabels(FN, fontsize=8); ax.set_title(m, fontsize=10); ax.set_ylabel('SHAP normalizado')
    d22 += [{'modelo': m, 'feature': f, 'limpo': gl[f], 'ruido30': gr[f]} for f in FN]
axes[0].legend(fontsize=8)
fig.suptitle('M4, v2: importância SHAP média (normalizada por série), limpo x ruído σ=30%', fontsize=11)
red = pd.read_csv(V2 / 'm4' / 'xai' / 'xai_m4_redistribuicao_ruido.csv')
pick = lambda m, f: red[(red.modelo == m) & (red.metodo == 'shap') & (red.feature == f)].iloc[0]
salvar(fig, 22, 'm4_shap_limpo_x_ruido', 'Importâncias SHAP entre dados limpos e com ruído σ=30%',
       'RF lag_1 160→62 e roll_mean_6 38→104; GB 182→98 e 51→105 (medianas brutas)',
       '; '.join(f"{m} lag_1 {pick(m, 'lag_1')['Δ (ruído − limpo)']}, roll_mean_6 {pick(m, 'roll_mean_6')['Δ (ruído − limpo)']}" for m in MOD_XAI),
       pd.DataFrame(d22), nota='v2 usa a média das importâncias normalizadas (o antigo, medianas brutas); no SVR o ganho do roll_mean_6 tem IC incluindo zero')
'''))

# ---------------------------------------------------------------- Figs 23-26
cells.append(md(r'''
### 7.4 Car Parts (Figs. 23 a 26)
'''))

cells.append(code(r'''
bs = BS['Car Parts']; ordem_cp = list(rk_mae.sort_values('rank').index)
fig, ax = plt.subplots(figsize=(15, 5)); boxplot_modelos(ax, bs, ordem_cp, 'mape', fmt='{:.1f}')
ax.set_ylabel('MAPE (%) por série (só passos não nulos)')
ax.set_title(f"Car Parts, v2: MAPE por modelo (indefinido em {mape_nan_cp:.1f}% das janelas; séries sem MAPE ficam fora)", fontsize=10)
m23 = bs.groupby('model')['mape'].median().loc[ordem_cp].rename(index=NOMES).round(1)
salvar(fig, 23, 'cp_mape', 'Distribuição do MAPE por algoritmo no Car Parts', 'cobertura do MAPE 89,0% (rotulada como janelas)',
       f'MAPE indefinido em {mape_nan_cp:.1f}% das janelas (84,2% de cobertura por janela, 89,0% por série)', m23.reset_index())

fig, ax = plt.subplots(figsize=(15, 5)); p95 = bs['mae'].quantile(0.95)
boxplot_modelos(ax, bs, ordem_cp, 'mae', clip=p95, fmt='{:.3f}'); ax.set_ylabel(f'MAE por série, truncado no p95 ({p95:.2f})')
ax.set_title('Car Parts, v2: MAE por modelo (10 modelos, ordenados pelo MAE mediano)', fontsize=10)
m24 = bs.groupby('model')['mae'].median().loc[ordem_cp].rename(index=NOMES).round(3)
salvar(fig, 24, 'cp_mae', 'Distribuição do MAE por algoritmo no Car Parts',
       'Persistência 0,333 (1º); RF 0,485 (2º); SVR 0,488 (3º); HW 0,741 (8º de 9)',
       '; '.join(f'{k} {v:.3f}' for k, v in m24.iloc[:3].items()) + f"; HW {m24['Holt-Winters']:.3f} (10º de 10)", m24.reset_index(),
       nota='o 2º lugar do SVR depende das janelas com fallback (MAE sem fallback 0,507)')

fig, axes = plt.subplots(2, 2, figsize=(17, 10))
for ax, (met, rot, clip) in zip(axes.flat, [('mape', 'MAPE (%)', None), ('mae', 'MAE', p95), ('r2', 'R²', None), ('mase', 'MASE', 5)]):
    b_ = bs if met != 'r2' else bs.assign(r2=bs['r2'].clip(lower=-10))
    boxplot_modelos(ax, b_, ordem_cp, met, clip=clip, fmt='{:.2f}'); ax.set_title(rot + (' (R² truncado em -10)' if met == 'r2' else ''), fontsize=10)
fig.suptitle('Car Parts, v2: MAPE, MAE, R² e MASE por modelo (2.674 séries, H=18)', fontsize=11)
m25 = bs.groupby('model')[['mape', 'mae', 'r2', 'mase']].median().loc[ordem_cp].rename(index=NOMES).round(3)
salvar(fig, 25, 'cp_4_metricas', 'Comparação de desempenho no Car Parts: MAPE, MAE e R²',
       'R² mediano de -0,09 a -1,67', f"R² mediano de {m25['r2'].max():.2f} a {m25['r2'].min():.2f}; R² indefinido em 15,8% das janelas; MASE acrescentado",
       m25.reset_index())

fig, axes = plt.subplots(2, 1, figsize=(15, 9)); d26 = []
for ax, u in zip(axes, ['T2566', 'T2419']):
    s = np.asarray(S_CP[u], float); o = pv.make_rolling_windows(len(s), 18)[-1]; tr, te = s[:o], s[o:o + 18]
    p_ = np.asarray(pv.model_random_forest(tr, 18)[0], float); mt = pv.calc_metrics(te, p_)
    ax.plot(range(o), tr, color='#4682B4', label='treino'); ax.plot(range(o, o + 18), te, color='black', lw=2, label='real (teste)')
    ax.plot(range(o, o + 18), p_, color='#2E8B57', label=f"Random Forest (MAE={mt['mae']:.2f}, MAPE={mt['mape']:.1f}%)")
    ax.axvline(o - 0.5, ls=':', color='grey'); ax.legend(fontsize=8)
    ax.set_title(f'Série {u} | zeros={100 * np.mean(s == 0):.0f}% | origem={o}', fontsize=10)
    d26.append({'serie': u, 'origem': o, 'zeros (%)': round(100 * np.mean(s == 0), 1), 'MAE RF v2': round(mt['mae'], 3), 'MAPE RF v2': mt['mape']})
fig.suptitle('Car Parts, v2: exemplos de intermitência com previsões do Random Forest (100 árvores, previsão recursiva corrigida)', fontsize=11)
d26 = pd.DataFrame(d26)
salvar(fig, 26, 'cp_exemplos_rf', 'Exemplos de séries Car Parts com diferentes perfis de intermitência (RF)',
       'T2566: MAE 1,16, MAPE 51,1%; T2419: MAE 1,17, MAPE 100,0% (RF com 30 árvores)',
       '; '.join(f"{r.serie}: MAE {r['MAE RF v2']:.2f}, MAPE {r['MAPE RF v2']:.1f}%" for _, r in d26.iterrows()), d26,
       nota='mesmas séries e mesma origem da figura antiga')

fig_idx = pd.DataFrame(FIG_REG).sort_values('figura')
fig_idx.to_csv(F5 / 'figuras_indice.csv', index=False)
print(); print(fig_idx[['figura', 'arquivo antigo', 'valor-chave antigo', 'valor-chave v2']].to_string(index=False))
'''))

# ---------------------------------------------------------------- Replicação H1-H3
cells.append(md(r'''
## 8. Replicação de H1 a H3 nas três bases
Mesmo código de avaliação do notebook 11c (`nb09v2.py`, seção 8), aplicado às importâncias da configuração oficial de cada base (30 séries, 15 features, LIME com 200 amostras, condição limpa e ruído σ=30%).
- **Tourism**: previsão pré-registrada (`PREVISOES_tourism.md`, 1a08fa1), interpretação da faixa da H1 em `INTERPRETACAO_criterios_4c.md` (1e3d9f0). O recálculo aqui só **confere** os valores oficiais; o rótulo é o de `tm/replicacao_previsoes_tm.csv`.
- **M4** (reanálise) e **Car Parts** (exploratória): os mesmos critérios numéricos **aplicados a posteriori**; rótulos "replicou / replicou parcialmente / não replicou" com as mesmas contagens.
- **H3 no M4**: o limiar de 30,0% é o máximo observado no próprio M4, então a aplicação ao M4 é circular (0 de 6 por construção); é reportada só para completar a tabela, com a posição global do lag_12 como evidência.
'''))

cells.append(code(r'''
def condicoes(imp):
    g = lambda m, met, cond: imp[(imp.modelo == m) & (imp.metodo == met) & (imp.condicao == cond)][FN].mean()
    cond = []
    for m in MOD_XAI:
        s, l = g(m, 'shap', 'limpo'), g(m, 'lime', 'limpo')
        pear = float(np.corrcoef((s / s.max()).values, (l / l.max()).values)[0, 1])
        a = imp[(imp.modelo == m) & (imp.condicao == 'limpo')]
        j = a[a.metodo == 'shap'].merge(a[a.metodo == 'lime'], on='serie', suffixes=('_s', '_l'))
        sp = [spearmanr(r[[f'{n}_s' for n in FN]].astype(float), r[[f'{n}_l' for n in FN]].astype(float))[0] for _, r in j.iterrows()]
        sp_m, sp_lo, sp_hi = pv.bootstrap_ic(sp)
        cond.append({'hipotese': 'H1', 'condicao': f'{m}: Pearson agregado > 0,90', 'valor': round(pear, 3), 'ok': pear > 0.90})
        obs = '' if sp_m <= 0.85 else 'acima da faixa: concordância maior que a prevista'
        cond.append({'hipotese': 'H1', 'condicao': f'{m}: Spearman por série em [0,65; 0,85]',
                     'valor': f'{sp_m:.3f} [{sp_lo:.3f}; {sp_hi:.3f}]', 'ok': 0.65 <= sp_m <= 0.85, 'obs': obs})
    for met in ['shap', 'lime']:
        sub = imp[imp.metodo == met]
        e = pv.estabilidade(sub[sub.condicao == 'limpo'], sub[sub.condicao == 'ruido30'], FN)
        for m in MOD_XAI:
            v, lo, hi = pv.bootstrap_ic(e[e.modelo == m]['spearman'].values)
            cond.append({'hipotese': 'H2', 'condicao': f'(a) {m} {met.upper()}: estabilidade (Spearman médio) < 0,80',
                         'valor': f'{v:.3f} [{lo:.3f}; {hi:.3f}]', 'ok': v < 0.80})
    def delta_ic(m, met, f):
        a = imp[(imp.modelo == m) & (imp.metodo == met)]
        j = a[a.condicao == 'limpo'][['serie', f]].merge(a[a.condicao == 'ruido30'][['serie', f]], on='serie')
        return pv.bootstrap_ic((j[f'{f}_y'] - j[f'{f}_x']).values)
    for m in ['RandomForest', 'GradientBoosting']:
        for met in ['shap', 'lime']:
            d1, d3, d6 = delta_ic(m, met, 'lag_1'), delta_ic(m, met, 'roll_mean_3'), delta_ic(m, met, 'roll_mean_6')
            perde, ganha = d1[2] < 0, (d3[1] > 0) or (d6[1] > 0)
            cond.append({'hipotese': 'H2', 'condicao': f'(b) {m} {met.upper()}: lag_1 perde e média móvel ganha (IC excluindo zero)',
                         'valor': f'Δlag_1 {d1[0]:+.3f} [{d1[1]:+.3f}; {d1[2]:+.3f}]; Δroll_mean_3 {d3[0]:+.3f} [{d3[1]:+.3f}; {d3[2]:+.3f}]; '
                                  f'Δroll_mean_6 {d6[0]:+.3f} [{d6[1]:+.3f}; {d6[2]:+.3f}]',
                         'ok': bool(perde and ganha),
                         'obs': ('' if perde else 'lag_1 não perde com IC excluindo zero; ') + ('' if ganha else 'nenhuma média móvel ganha com IC excluindo zero')})
    for m in MOD_XAI:
        for met in ['shap', 'lime']:
            sub = imp[(imp.modelo == m) & (imp.metodo == met) & (imp.condicao == 'limpo')]
            rr = sub[FN].rank(axis=1, ascending=False, method='min')['lag_12']
            frac = 100 * float((rr <= 3).mean())
            pos = int(sub[FN].mean().rank(ascending=False, method='min')['lag_12'])
            cond.append({'hipotese': 'H3', 'condicao': f'{m} {met.upper()}: lag_12 no top-3 em > 30,0% das séries',
                         'valor': f'{frac:.1f}% ({int((rr <= 3).sum())} de {len(rr)}); lag_12 em {pos}º global', 'ok': frac > 30.0})
    return pd.DataFrame(cond).fillna('')

FAIXAS = {'H1': (6, (3, 5)), 'H2': (10, (5, 9)), 'H3': (6, (1, 5))}
def rotulo(n_ok, n_tot, faixa, oficial):
    r = 'confirmou' if n_ok == n_tot else ('confirmou parcialmente' if faixa[0] <= n_ok <= faixa[1] else 'não confirmou')
    return r if oficial else {'confirmou': 'replicou', 'confirmou parcialmente': 'replicou parcialmente', 'não confirmou': 'não replicou'}[r]

COND = {b: condicoes(IMP[b]).assign(base=b) for b in IMP}
# conferência: o recálculo do Tourism reproduz a avaliação oficial (4c)
of = pd.read_csv(V2 / 'tm' / 'xai' / 'xai_tm_avaliacao_previsoes_condicoes.csv')
chk = of.merge(COND['Tourism'], on=['hipotese', 'condicao'], suffixes=('_of', '_v'))
assert len(chk) == 22 and (chk.ok_of == chk.ok_v).all()
assert (chk.valor_of.astype(str) == chk.valor_v.astype(str).str.replace(r'; lag_12 em \d+º global', '', regex=True)).all()
print('Tourism: recálculo idêntico à avaliação oficial (22 condições, valores e decisões)')
cond_all = pd.concat(COND.values(), ignore_index=True)[['base', 'hipotese', 'condicao', 'valor', 'ok', 'obs']]
cond_all.to_csv(F5 / 'replicacao_condicoes_h1_h3.csv', index=False)
for b in IMP:
    c = COND[b]
    print(f"=== {b}: " + ' | '.join(f"{h} {int(c[c.hipotese == h].ok.sum())} de {FAIXAS[h][0]}" for h in FAIXAS))
    print(c[~c.ok.astype(bool)][['hipotese', 'condicao', 'valor', 'obs']].to_string(index=False)); print()
'''))

# ---------------------------------------------------------------- H4
cells.append(md(r'''
## 9. H4 nos três pares de bases (MASE principal, MAE de apoio)
Para cada base, rank dos 10 modelos pela mediana entre séries e IC por bootstrap pareado por série (2000) dentro da base; rank da família = posição do seu melhor modelo. Em cada par: a ordem das famílias difere? alguma família tem IC do rank sem sobreposição entre as bases? E o Spearman entre os rankings dos 10 modelos, com IC (reamostragem independente das duas bases).
- Par **Tourism x M4** e **Tourism x Car Parts**: compõem a H4 registrada (rótulo oficial da 4a, conferido aqui).
- Par **M4 x Car Parts**: achado da dissertação (queda do HW, "dependência do regime"), reanálise com o mesmo critério aplicado a posteriori a um único par: replicou (ordem difere e há IC separado), parcialmente (difere só nas estimativas pontuais), não replicou (ordem idêntica).
'''))

cells.append(code(r'''
RES, BOOT = {}, {}
for met in ['mase', 'mae']:
    for b in JAN:
        RES[(met, b)], BOOT[(met, b)] = pv.bootstrap_ranks(BS[b], MODS, met, n=2000)
def familias(met, b):
    t, boot = RES[(met, b)], BOOT[(met, b)]
    fam = pd.Series(pv.FAMILIA)[MODS]
    pont = t.groupby(fam.values)['mediana'].min().rank(method='min').astype(int)
    rb = boot.T.groupby(fam.values).min().T.rank(axis=1, method='min')
    return pd.DataFrame({'rank': pont, 'ic_inf': rb.quantile(0.025), 'ic_sup': rb.quantile(0.975)})
def rho_ic(a, b, met):
    rho = spearmanr(RES[(met, a)]['rank'], RES[(met, b)]['rank'])[0]
    ra, rb = BOOT[(met, a)].rank(axis=1).values, BOOT[(met, b)].rank(axis=1).values
    rng = np.random.default_rng(pv.SEED)
    rr = [spearmanr(ra[i], rb[j])[0] for i, j in zip(rng.integers(0, 2000, 2000), rng.integers(0, 2000, 2000))]
    return rho, np.percentile(rr, 2.5), np.percentile(rr, 97.5)
FAM = {(met, b): familias(met, b) for met in ['mase', 'mae'] for b in JAN}
fmt_f = lambda f: f.apply(lambda r: f"{int(r['rank'])} [{r.ic_inf:.0f}; {r.ic_sup:.0f}]", axis=1)
for met in ['mase', 'mae']:   # conferência com a 4a
    ref = pd.read_csv(V2 / 'tm' / 'limpo' / f'familias_tres_bases_{met}_tm.csv', index_col=0)
    for b in JAN:
        assert (fmt_f(FAM[(met, b)]).loc[ref.index] == ref[b]).all(), (met, b)
print('famílias por base: idênticas às da Fase 4a (MASE e MAE)')
ordem = lambda met, b: tuple(FAM[(met, b)].sort_values('rank').index)
def par(a, b, met):
    fa, fb = FAM[(met, a)], FAM[(met, b)]
    sep = [f for f in fa.index if fa.loc[f, 'ic_inf'] > fb.loc[f, 'ic_sup'] or fa.loc[f, 'ic_sup'] < fb.loc[f, 'ic_inf']]
    difere = ordem(met, a) != ordem(met, b)
    rho, lo, hi = rho_ic(a, b, met)
    situ = 'ordem difere, com IC separado' if difere and sep else ('ordem difere só nas estimativas pontuais' if difere else 'ordem idêntica')
    return {'par': f'{a} x {b}', 'métrica': met.upper(), 'ordem A': ' > '.join(ordem(met, a)), 'ordem B': ' > '.join(ordem(met, b)),
            'ordem difere': difere, 'famílias com IC separado': ', '.join(sep) if sep else 'nenhuma', 'situação': situ,
            'Spearman modelos': round(rho, 3), 'IC Spearman': f'[{lo:+.2f}; {hi:+.2f}]', 'IC inclui zero': lo <= 0 <= hi}
PARES = [('M4', 'Car Parts'), ('Tourism', 'M4'), ('Tourism', 'Car Parts')]
h4 = pd.DataFrame([par(a, b, met) for met in ['mase', 'mae'] for a, b in PARES])
h4.to_csv(F5 / 'replicacao_h4_pares.csv', index=False)
print(h4.to_string(index=False))
# status oficial da H4 (critério registrado, MASE) a partir dos dois pares do Tourism
pm = h4[h4['métrica'] == 'MASE'].set_index('par')
dif = [pm.loc[p, 'ordem difere'] for p in ['Tourism x M4', 'Tourism x Car Parts']]
sep = [pm.loc[p, 'famílias com IC separado'] != 'nenhuma' for p in ['Tourism x M4', 'Tourism x Car Parts']]
h4_tm = 'não confirmou' if not all(dif) else ('confirmou' if all(sep) else 'confirmou parcialmente')
REP_TM = pd.read_csv(V2 / 'tm' / 'replicacao_previsoes_tm.csv').set_index('hipotese')
assert h4_tm == REP_TM.loc['H4', 'status'], (h4_tm, REP_TM.loc['H4', 'status'])
SIT2ROT = {'ordem difere, com IC separado': 'replicou', 'ordem difere só nas estimativas pontuais': 'replicou parcialmente', 'ordem idêntica': 'não replicou'}
print(f'\nH4 oficial (Tourism, MASE): {h4_tm} | M4 x Car Parts (a posteriori, MASE): {SIT2ROT[pm.loc["M4 x Car Parts", "situação"]]}')
'''))

# ---------------------------------------------------------------- tabela final
cells.append(md(r'''
## 10. Tabela final de replicação H1 a H4 (formato longo e resumo)
'''))

cells.append(code(r'''
TIPO = {'M4': 'reanálise (critério aplicado a posteriori)', 'Car Parts': 'exploratória (critério aplicado a posteriori)',
        'Tourism': 'previsão pré-registrada (1a08fa1)'}
DISSERT = {
    'H1': 'Concordância SHAP x LIME alta: Pearson > 0,90 (Fig. 20: 0,959 / 0,952 / 0,908)',
    'H2': 'Atribuições SHAP moderadamente estáveis sob ruído σ=30% (ρ mediano 0,572 / 0,344 / 0,520); sob ruído o lag_1 perde e as médias móveis ganham (Fig. 22)',
    'H3': 'Lag sazonal: LIME no top-3 e SHAP em 7º a 8º (divergência); reformulada como importância do lag_12 em SHAP e LIME',
    'H4': 'Hierarquia de desempenho depende do regime: HW 1º no M4 e penúltimo no Car Parts; ML mais estável entre regimes (Seção 5.6)'}
CRIT = {'H1': 'Pearson agregado > 0,90 e Spearman por série em [0,65; 0,85], por modelo (6 condições)',
        'H2': '(a) estabilidade < 0,80 por modelo e método; (b) RF e GB, SHAP e LIME: lag_1 perde e roll_mean_3 ou roll_mean_6 ganha, IC excluindo zero (10 condições)',
        'H3': 'lag_12 no top-3 em > 30,0% das séries, por modelo e método (6 condições)',
        'H4': 'ordem das famílias (MASE, rank do melhor modelo) difere e há família com IC do rank separado; MAE de apoio'}
NOTA = {'H1': ('faixa fechada [0,65; 0,85]; valor acima de 0,85 conta como fora da faixa e indica concordância MAIOR que a prevista '
               '(INTERPRETACAO_criterios_4c.md, 1e3d9f0); Pearson agregado alto é puxado pela feature dominante'),
        'H2': 'leitura pós-hoc no Tourism ("a feature dominante perde e as médias móveis ganham") não é previsão registrada e não altera o status',
        'H3': '', 'H4': ''}
linhas = []
for b in ['M4', 'Car Parts', 'Tourism']:
    c = COND[b]
    for h, (n_tot, faixa) in FAIXAS.items():
        ch = c[c.hipotese == h]; n_ok = int(ch.ok.sum())
        if b == 'Tourism':
            st, cont = REP_TM.loc[h, 'status'], f'{n_ok} de {n_tot}'
            assert st == rotulo(n_ok, n_tot, faixa, True)
        else:
            st, cont = rotulo(n_ok, n_tot, faixa, False), f'{n_ok} de {n_tot}'
        falhas = '; '.join(f'{x.condicao} = {x.valor}' + (f' ({x.obs.strip("; ")})' if x.obs else '') for x in ch.itertuples() if not x.ok)
        nota = NOTA[h]
        if h == 'H3' and b == 'M4':
            nota = 'limiar de 30,0% = máximo do próprio M4: aplicação circular, 0 de 6 por construção'
        if h == 'H2' and b == 'Car Parts':
            nota = 'ruído proporcional a σ em série intermitente cria estrutura artificial (decisão 3A); ' + nota
        if b == 'Car Parts':
            nota = 'XAI exploratória (30 de 2.505 séries elegíveis; 21 linhas de treino); ' + nota
        linhas.append({'hipótese': h, 'base / par': b, 'tipo de evidência': TIPO[b], 'formulação da dissertação': DISSERT[h],
                       'critério': CRIT[h], 'condições válidas': cont, 'status': st,
                       'evidência (condições que não valem)': falhas if falhas else 'nenhuma',
                       'fonte': 'fase5/replicacao_condicoes_h1_h3.csv' + (' ; tm/xai/xai_tm_avaliacao_previsoes_condicoes.csv' if b == 'Tourism' else ''),
                       'nota': nota.strip('; ')})
for p in ['M4 x Car Parts', 'Tourism x M4', 'Tourism x Car Parts']:
    rm, ra = pm.loc[p], h4[(h4['métrica'] == 'MAE')].set_index('par').loc[p]
    oficial = p.startswith('Tourism')
    ev = (f"MASE: {rm['ordem A']} | {rm['ordem B']}; IC separado: {rm['famílias com IC separado']}; Spearman modelos {rm['Spearman modelos']:+.2f} {rm['IC Spearman']}. "
          f"MAE (apoio): {ra['situação']}; IC separado: {ra['famílias com IC separado']}; Spearman {ra['Spearman modelos']:+.2f} {ra['IC Spearman']}")
    linhas.append({'hipótese': 'H4', 'base / par': p,
                   'tipo de evidência': 'previsão pré-registrada (1a08fa1)' if oficial else 'reanálise (critério aplicado a posteriori)',
                   'formulação da dissertação': DISSERT['H4'], 'critério': CRIT['H4'], 'condições válidas': rm['situação'],
                   'status': (f'compõe a H4 oficial: {h4_tm}' if oficial else SIT2ROT[rm['situação']]), 'evidência (condições que não valem)': ev,
                   'fonte': 'fase5/replicacao_h4_pares.csv ; tm/replicacao_previsoes_tm.csv' if oficial else 'fase5/replicacao_h4_pares.csv',
                   'nota': ('H4 oficial = Tourism difere do M4 e do Car Parts; IC separado só contra o Car Parts' if oficial else
                            'reproduz a queda do HW (MASE 2º → 10º) mas sem IC separado no nível das famílias')})
rep = pd.DataFrame(linhas)
rep.to_csv(F5 / 'replicacao_H1_H4.csv', index=False)

res = rep[rep['hipótese'] != 'H4'].pivot(index='hipótese', columns='base / par', values='status')
cont = rep[rep['hipótese'] != 'H4'].pivot(index='hipótese', columns='base / par', values='condições válidas')
resumo = (res + ' (' + cont + ')')[['M4', 'Car Parts', 'Tourism']]
resumo.loc['H4'] = [f"par M4 x Car Parts: {SIT2ROT[pm.loc['M4 x Car Parts', 'situação']]}", 'mesmo par M4 x Car Parts (ver coluna M4)',
                    f'{h4_tm} (difere das duas bases; IC separado só vs Car Parts)']
resumo.columns = [f'{c}: {TIPO[c]}' for c in resumo.columns]
resumo.to_csv(F5 / 'replicacao_H1_H4_resumo.csv')
print(resumo.to_string())
'''))

# ---------------------------------------------------------------- sensibilidade 366
cells.append(md(r'''
## 11. Sensibilidade com as 366 séries do Tourism (arquivo separado)
Não substitui os rótulos oficiais (200 séries). H1 a H3 vêm da Fase 4d (`tm_366/`); a H4 é recalculada aqui nas 366 séries com o mesmo critério registrado (MASE).
'''))

cells.append(code(r'''
c200_366 = pd.read_csv(V2 / 'tm_366' / 'sensibilidade' / 'condicoes_h1_h3_200_366.csv')
s366 = pd.read_csv(V2 / 'tm_366' / 'sensibilidade_previsoes_tm366.csv').set_index('hipotese')
j366 = pv.adicionar_mase(carregar('tm_366', 'limpo'), S_TM_TODAS)
assert j366.unique_id.nunique() == 366
RES[('mase', 'Tourism 366')], BOOT[('mase', 'Tourism 366')] = pv.bootstrap_ranks(pv.agregar_por_serie(j366), MODS, 'mase', n=2000)
FAM[('mase', 'Tourism 366')] = familias('mase', 'Tourism 366')
p366 = {b: par('Tourism 366', b, 'mase') for b in ['M4', 'Car Parts']}
dif = [p366[b]['ordem difere'] for b in p366]; sep = [p366[b]['famílias com IC separado'] != 'nenhuma' for b in p366]
h4_366 = 'não confirmou' if not all(dif) else ('confirmou' if all(sep) else 'confirmou parcialmente')
linhas = []
for h in ['H1', 'H2', 'H3']:
    c = c200_366[c200_366.hipotese == h]
    muda = c[~c['mesma decisão']]
    linhas.append({'hipótese': h, 'rótulo oficial (200 séries)': REP_TM.loc[h, 'status'], 'condições 200': f"{int(c['ok 200'].sum())} de {len(c)}",
                   'rótulo sensibilidade (366 séries)': s366.loc[h, 'status'], 'condições 366': s366.loc[h, 'condicoes_validas'],
                   'condições que mudam de decisão': '; '.join(f"{x.condicao}: {x._3} → {x._5}" for x in muda.itertuples()) if len(muda) else 'nenhuma',
                   'fonte': 'tm_366/sensibilidade/condicoes_h1_h3_200_366.csv'})
linhas.append({'hipótese': 'H4', 'rótulo oficial (200 séries)': h4_tm, 'condições 200': f"ordem TM: {pm.loc['Tourism x M4', 'ordem A']}",
               'rótulo sensibilidade (366 séries)': h4_366, 'condições 366': f"ordem TM 366: {p366['M4']['ordem A']}; IC separado vs M4: "
               f"{p366['M4']['famílias com IC separado']}; vs Car Parts: {p366['Car Parts']['famílias com IC separado']}",
               'condições que mudam de decisão': 'nenhuma' if h4_366 == h4_tm else 'rótulo muda', 'fonte': 'recalculado neste notebook (MASE, 366 séries)'})
sens = pd.DataFrame(linhas)
sens['rótulo'] = 'SENSIBILIDADE (366 séries; inclui as 200 oficiais, não independente); não substitui a avaliação oficial'
sens.to_csv(F5 / 'replicacao_sensibilidade_tm366.csv', index=False)
print(sens.drop(columns=['rótulo', 'fonte']).to_string(index=False))
'''))

# ---------------------------------------------------------------- índice e meta
cells.append(md(r'''
## 12. Índice dos arquivos da Fase 5 e metadados
'''))

cells.append(code(r'''
DESC = {'tabela3_antigo_v2_m4.csv': 'Tabela 3 (M4 limpo): antigo x v2, IC, com e sem fallback e explosivas, MASE',
        'tabela4_antigo_v2.csv': 'Tabela 4 (mapa síntese): antigo x v2 M4, com Tourism; custo só medido',
        'tabela5_antigo_v2.csv': 'Tabela 5 (recomendações): critérios verificáveis em M4, Tourism e Car Parts',
        'tabela6_antigo_v2.csv': 'Tabela 6 (características das bases): antigo x v2, três bases, tempo com SARIMA automático',
        'tabela7_antigo_v2_cp.csv': 'Tabela 7 (Car Parts): antigo x v2, IC, com e sem fallback, MASE',
        'custo_computacional_v2.csv': 'tempos medidos no v2 (logs e exec_time_s)',
        'reanalise_cv_q1_m4.csv': 'reanálise do CV da Questão 1 (item C11), Fig. 18',
        'figuras_indice.csv': 'índice das Figs. 8 a 26 (antigo x v2, valores-chave)',
        'replicacao_condicoes_h1_h3.csv': 'condições de H1 a H3 nas três bases',
        'replicacao_h4_pares.csv': 'H4 nos três pares de bases (MASE e MAE, IC)',
        'replicacao_H1_H4.csv': 'tabela final de replicação (formato longo, com evidência e tipo)',
        'replicacao_H1_H4_resumo.csv': 'resumo hipótese x base',
        'replicacao_sensibilidade_tm366.csv': 'sensibilidade com 366 séries (separada dos rótulos oficiais)'}
idx = pd.DataFrame([{'arquivo': k, 'conteúdo': v} for k, v in DESC.items()])
assert all((F5 / k).exists() for k in DESC)
idx.to_csv(F5 / 'indice_fase5.csv', index=False)
meta = {'gerado_em': datetime.now().isoformat(timespec='seconds'), 'notebook': 'notebooks/12_fase5_comparativos_v2.ipynb',
        'pipeline_sha256_atual': pv.hash_pipeline(), 'pipeline_sha256_caches': hashes, 'versoes': pv.versoes(),
        'dissertacao_docx': {'arquivo': DOCX.name, 'sha256': DOCX_SHA, 'uso': 'somente leitura das Tabelas 3 a 7'},
        'bootstrap': {'n': 2000, 'seed': pv.SEED}, 'arquivos': DESC}
(F5 / 'fase5.meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
print(idx.to_string(index=False)); print(f'\nfiguras: {len(list(FIG.glob("*.png")))} PNG em {FIG.relative_to(PROJECT_ROOT)}')
'''))

check(cells)
nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': '.venv', 'language': 'python', 'name': 'python3'},
                                   'language_info': {'name': 'python', 'version': '3.11.0'}},
      'nbformat': 4, 'nbformat_minor': 5}
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('ok', NB.name, len(cells))
