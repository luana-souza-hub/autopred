#!/usr/bin/env python
"""Gera results/v2/hiperparametros_v2.csv e .md lendo src/pipeline_v2.py.

Valores definidos no pipeline vêm das constantes e funções do módulo; valores que o pipeline
deixa no padrão da biblioteca vêm de get_params() dos estimadores instanciados por make_estimator
e de inspect.signature das classes do statsmodels e do LIME, nas versões instaladas na .venv.

Uso (na raiz do projeto):  .venv\\Scripts\\python.exe tools\\gerar_hiperparametros_v2.py
"""
from __future__ import annotations

import inspect
import sys
from itertools import product
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src import pipeline_v2 as pv  # noqa: E402

V = pv.versoes()
SHA = pv.hash_pipeline()


def loc(fn):
    return f'pipeline_v2.py:{inspect.getsourcelines(fn)[1]} ({fn.__name__})'


def padrao(cls, metodo, nome):
    return inspect.signature(getattr(cls, metodo)).parameters[nome].default


from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing  # noqa: E402
from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: E402
from lime.lime_tabular import LimeTabularExplainer  # noqa: E402

SK = f"padrão scikit-learn {V['scikit-learn']}"
SM = f"padrão statsmodels {V['statsmodels']}"

NOMES = {
    'persistence': 'Persistência', 'seasonal_naive': 'Sazonal ingênuo', 'mean': 'Média histórica',
    'ses': 'SES', 'holt_winters': 'Holt-Winters', 'sarima_heuristic': 'SARIMA heurístico',
    'sarima_auto': 'SARIMA automático', 'random_forest': 'RF', 'gbt': 'GB', 'svr': 'SVR',
}
EST = {'random_forest': 'RandomForest', 'gbt': 'GradientBoosting', 'svr': 'SVR'}
CLASSE = {'random_forest': 'sklearn.ensemble.RandomForestRegressor',
          'gbt': 'sklearn.ensemble.GradientBoostingRegressor', 'svr': 'sklearn.svm.SVR'}

linhas = []


def add(modelo, categoria, parametro, valor, origem, obs=''):
    linhas.append({'modelo': NOMES.get(modelo, modelo), 'chave_codigo': modelo,
                   'familia': pv.FAMILIA.get(modelo, 'Todos'), 'categoria': categoria,
                   'parametro': parametro, 'valor': str(valor), 'origem': origem, 'observacao': obs})


SNAIVE_FB = 'previsão do sazonal ingênuo, status FALLBACK_SNAIVE'

# ── Baselines ────────────────────────────────────────────────────────────────
add('persistence', 'hiperparametro', 'regra', 'ŷ(t+k) = último valor do treino, k = 1..H',
    loc(pv.model_persistence))
add('persistence', 'fallback', 'fallback', 'nenhum (sempre OK)', loc(pv.model_persistence))

add('seasonal_naive', 'hiperparametro', 'periodo_sazonal', pv.SEASON, 'pipeline_v2.py (SEASON)')
add('seasonal_naive', 'hiperparametro', 'regra', 'ŷ(t+k) = treino[n − 12 + ((k−1) mod 12)]',
    loc(pv._seasonal_naive), 'repete o último ciclo de 12 meses do treino')
add('seasonal_naive', 'fallback', 'treino_menor_que_periodo',
    'n < 12: repete o último valor (persistência), status continua OK', loc(pv._seasonal_naive),
    f'não ocorre no protocolo: MIN_TRAIN={pv.MIN_TRAIN} e o truncamento mantém ao menos MIN_TRAIN')

add('mean', 'hiperparametro', 'regra', 'ŷ(t+k) = média do treino inteiro da janela', loc(pv.model_mean))
add('mean', 'fallback', 'fallback', 'nenhum (sempre OK)', loc(pv.model_mean))

# ── SES ──────────────────────────────────────────────────────────────────────
add('ses', 'hiperparametro', 'implementacao', 'statsmodels SimpleExpSmoothing', loc(pv.model_ses))
add('ses', 'hiperparametro', 'initialization_method', 'estimated', loc(pv.model_ses))
add('ses', 'hiperparametro', 'smoothing_level (alfa)', 'otimizado (optimized=True; o statsmodels minimiza a soma dos erros quadráticos)',
    loc(pv.model_ses))
add('ses', 'hiperparametro', 'use_brute', padrao(SimpleExpSmoothing, 'fit', 'use_brute'), SM)
add('ses', 'hiperparametro', 'remove_bias', padrao(SimpleExpSmoothing, 'fit', 'remove_bias'), SM)
add('ses', 'fallback', 'excecao_no_ajuste', 'média do treino, status FALLBACK_MEAN', loc(pv.model_ses))

# ── Holt-Winters ─────────────────────────────────────────────────────────────
f = loc(pv.model_holt_winters)
add('holt_winters', 'hiperparametro', 'implementacao', 'statsmodels ExponentialSmoothing', f)
add('holt_winters', 'hiperparametro', 'trend', 'add', f)
add('holt_winters', 'hiperparametro', 'damped_trend', padrao(ExponentialSmoothing, '__init__', 'damped_trend'), SM)
add('holt_winters', 'hiperparametro', 'seasonal_periods', pv.SEASON, f)
add('holt_winters', 'hiperparametro', 'initialization_method', 'estimated', f)
add('holt_winters', 'hiperparametro', 'use_boxcox', padrao(ExponentialSmoothing, '__init__', 'use_boxcox'), SM)
add('holt_winters', 'hiperparametro', 'alfa, beta, gama', 'otimizados (optimized=True; o statsmodels minimiza a soma dos erros quadráticos)', f)
add('holt_winters', 'regra_selecao', 'seasonal', "ajusta 'add' e, se todo o treino > 0, também 'mul'; fica o de menor AIC",
    f, 'tipo escolhido gravado em info: {"sazonal": "add"|"mul"}; seleção só com o treino da janela')
add('holt_winters', 'fallback', 'treino_curto', f'n < 2 x {pv.SEASON} = {2 * pv.SEASON}: {SNAIVE_FB}', f)
add('holt_winters', 'fallback', 'nenhum_ajuste_valido', f'ambos os tipos lançam exceção: {SNAIVE_FB}', f)

# ── SARIMA heurístico ────────────────────────────────────────────────────────
f = loc(pv.model_sarima_heuristic)
fs = loc(pv._sarimax_fit)
add('sarima_heuristic', 'hiperparametro', 'implementacao', 'statsmodels SARIMAX', fs)
add('sarima_heuristic', 'hiperparametro', 'order (p,d,q)', pv.SARIMA_HEUR_ORDER, 'pipeline_v2.py (SARIMA_HEUR_ORDER)')
add('sarima_heuristic', 'hiperparametro', 'seasonal_order (P,D,Q,s)', pv.SARIMA_HEUR_SEASONAL,
    'pipeline_v2.py (SARIMA_HEUR_SEASONAL)', 'ordem fixa para todas as séries e janelas')
for nome, cfg in [('enforce_stationarity', False), ('enforce_invertibility', False)]:
    add('sarima_heuristic', 'hiperparametro', nome, cfg, fs)
add('sarima_heuristic', 'hiperparametro', 'trend', padrao(SARIMAX, '__init__', 'trend'), SM, 'sem termo determinístico')
add('sarima_heuristic', 'hiperparametro', 'otimizador (method)', padrao(SARIMAX, 'fit', 'method'), SM)
add('sarima_heuristic', 'hiperparametro', 'maxiter', pv.SARIMA_MAXITER, 'pipeline_v2.py (SARIMA_MAXITER)')
add('sarima_heuristic', 'hiperparametro', 'limite_tempo_s', pv.SARIMA_TIME_LIMIT_S, 'pipeline_v2.py (SARIMA_TIME_LIMIT_S)',
    'medido depois do ajuste (não interrompe o ajuste)')
add('sarima_heuristic', 'fallback', 'treino_constante', f'desvio padrão do treino = 0: {SNAIVE_FB}', f,
    'info: {"motivo": "treino_constante"}')
add('sarima_heuristic', 'fallback', 'excecao', f'exceção no ajuste: {SNAIVE_FB}', f, 'info: {"motivo": "excecao"}')
add('sarima_heuristic', 'fallback', 'tempo', f'ajuste > {pv.SARIMA_TIME_LIMIT_S:.0f} s: {SNAIVE_FB}', f,
    'info: {"motivo": "tempo"}')
add('sarima_heuristic', 'fallback', 'verossimilhanca_degenerada',
    f'nobs − loglikelihood_burn <= nº de parâmetros, ou llf não finito, ou llf == 0: {SNAIVE_FB}',
    loc(pv.verossimilhanca_degenerada), 'info: {"motivo": "verossimilhanca_degenerada"}; típico de janelas de 24 obs com D=1')
add('sarima_heuristic', 'sinalizacao', 'previsao_explosiva', 'max|previsão| > 10 x max|treino|: status continua OK, info {"explosiva": true}',
    loc(pv._diagnostico_previsao), 'reportado com e sem as explosivas')

# ── SARIMA automático ────────────────────────────────────────────────────────
g = pv.SARIMA_AUTO_GRID
cands = [c for c in product(g['p'], g['d'], g['q'], g['P'], g['D'], g['Q']) if not (c[0] == c[2] == c[3] == c[5] == 0)]
f = loc(pv.model_sarima_auto)
fsel = loc(pv.selecionar_ordem_sarima)
add('sarima_auto', 'hiperparametro', 'implementacao', 'statsmodels SARIMAX', fs)
for k in ['p', 'd', 'q', 'P', 'D', 'Q']:
    add('sarima_auto', 'regra_selecao', f'grade_{k}', g[k], 'pipeline_v2.py (SARIMA_AUTO_GRID)')
add('sarima_auto', 'regra_selecao', 's', pv.SEASON, 'pipeline_v2.py (SEASON)')
add('sarima_auto', 'regra_selecao', 'exclusao', 'p = q = P = Q = 0 (só diferenças) não é avaliado', fsel)
add('sarima_auto', 'regra_selecao', 'n_candidatos', len(cands), fsel,
    f'{len(list(product(*g.values())))} combinações da grade menos as {len(list(product(*g.values()))) - len(cands)} excluídas')
add('sarima_auto', 'regra_selecao', 'criterio', 'menor AIC entre os candidatos que convergem e não são degenerados', fsel)
add('sarima_auto', 'regra_selecao', 'dados_da_selecao', 'só o treino da janela (seleção refeita a cada janela; C4)', fsel)
add('sarima_auto', 'regra_selecao', 'ordem_de_busca', 'itertools.product na ordem p, d, q, P, D, Q', fsel)
add('sarima_auto', 'regra_selecao', 'limite_tempo_s', pv.SARIMA_AUTO_TIME_LIMIT_S, 'pipeline_v2.py (SARIMA_AUTO_TIME_LIMIT_S)',
    'por janela; verificado antes de cada candidato; se estourar, usa o melhor até ali e grava busca_completa=false')
add('sarima_auto', 'regra_selecao', 'ajuste_final', 'reajusta a ordem escolhida no mesmo treino e prevê H passos', f)
for nome, cfg in [('enforce_stationarity', False), ('enforce_invertibility', False)]:
    add('sarima_auto', 'hiperparametro', nome, cfg, fs)
add('sarima_auto', 'hiperparametro', 'trend', padrao(SARIMAX, '__init__', 'trend'), SM)
add('sarima_auto', 'hiperparametro', 'otimizador (method)', padrao(SARIMAX, 'fit', 'method'), SM)
add('sarima_auto', 'hiperparametro', 'maxiter', pv.SARIMA_MAXITER, 'pipeline_v2.py (SARIMA_MAXITER)', 'em cada candidato e no ajuste final')
add('sarima_auto', 'fallback', 'treino_constante', f'desvio padrão do treino = 0: {SNAIVE_FB}', f, 'sem campo motivo no info')
add('sarima_auto', 'fallback', 'sem_candidato_valido', f'nenhum candidato convergiu sem degeneração: {SNAIVE_FB}', f,
    'info: {"motivo": "sem_candidato_valido"}')
add('sarima_auto', 'fallback', 'excecao', f'exceção no ajuste final: {SNAIVE_FB}', f, 'info: {"motivo": "excecao"}')
add('sarima_auto', 'sinalizacao', 'previsao_explosiva', 'max|previsão| > 10 x max|treino|: status continua OK, info {"explosiva": true}',
    loc(pv._diagnostico_previsao), 'reportado com e sem as explosivas')

# ── ML ───────────────────────────────────────────────────────────────────────
for m in ['random_forest', 'gbt', 'svr']:
    est = EST[m]
    definidos = pv.HIPERPARAMETROS[est]
    add(m, 'hiperparametro', 'implementacao', CLASSE[m], loc(pv.make_estimator))
    for k, v in sorted(pv.make_estimator(est).get_params().items()):
        if k in ('verbose',):
            continue
        add(m, 'semente' if k == 'random_state' else 'hiperparametro', k, v,
            'pipeline_v2.py (HIPERPARAMETROS)' if k in definidos else SK)
    add(m, 'entrada', 'features', ', '.join(pv.feature_names()), loc(pv._linha_features),
        'lag_1..lag_12, médias móveis de 3 e 6, desvio padrão (ddof=1) de 3; sem lag_season (C5)')
    add(m, 'entrada', 'historico_minimo', pv.historico_minimo(), loc(pv.historico_minimo),
        'linhas de treino = n_treino − 12 (12 linhas na 1ª janela, com 24 obs)')
    add(m, 'entrada', 'estrategia', 'recursiva: cada passo usa as features de histórico + previsões anteriores', loc(pv.ml_forecast),
        'mesma função monta treino e próximo passo (C1)')
    if m == 'svr':
        add(m, 'entrada', 'escalonamento', 'StandardScaler em X e em y, ajustados só no treino da janela; previsão volta à escala original',
            loc(pv.ml_forecast))
        add(m, 'fallback', 'treino_insuficiente_ou_alvo_constante',
            'menos de 2 linhas de features, ou desvio padrão do alvo = 0: média do treino, status FALLBACK_MEAN', loc(pv.ml_forecast))
    else:
        add(m, 'entrada', 'escalonamento', 'nenhum', loc(pv.ml_forecast))
        add(m, 'fallback', 'treino_insuficiente', 'menos de 2 linhas de features: média do treino, status FALLBACK_MEAN',
            loc(pv.ml_forecast), 'alvo constante não aciona fallback (árvores preveem a constante)')
add('svr', 'semente', 'random_state', 'não se aplica (SVR é determinístico)', SK)

# ── Regras comuns a todos os modelos ─────────────────────────────────────────
T = 'Todos'
add(T, 'protocolo', 'janelas', f'{pv.N_ORIGINS} origens em linspace({pv.MIN_TRAIN}, n − H, {pv.N_ORIGINS}); treino = [:o], teste = [o:o+H]',
    loc(pv.make_rolling_windows), 'H = 18 (M4, Car Parts) e 24 (Tourism), definido em scripts/run_v2.py')
add(T, 'protocolo', 'MIN_TRAIN', pv.MIN_TRAIN, 'pipeline_v2.py (MIN_TRAIN)')
add(T, 'fallback', 'sem_janela', 'n < MIN_TRAIN + H: nenhuma janela, status SEM_JANELA (métricas NaN)', loc(pv.avaliar_serie))
add(T, 'fallback', 'previsao_invalida', f'forma diferente de (H,) ou valor não finito: {SNAIVE_FB}', loc(pv._chamar),
    'info: {"motivo": "previsao_invalida"}; vale para qualquer modelo')
add(T, 'fallback', 'erro', 'exceção não tratada pelo modelo: status ERRO, métricas NaN', loc(pv.avaliar_serie))
add(T, 'semente', 'SEED', pv.SEED, 'pipeline_v2.py (SEED)')
add(T, 'semente', 'sorteio_da_amostra', f'default_rng({pv.SEED}).choice sobre a ordem do arquivo', loc(pv.sortear_indices),
    'M4 e Tourism: 200 séries; Car Parts: todas')
add(T, 'semente', 'perturbacao_cenarios', f'default_rng([{pv.SEED}, serie_idx, origem, id_cenario])', loc(pv.rng_janela),
    'a mesma perturbação para todos os modelos na mesma série, janela e cenário')

# ── Explicabilidade (RF, GB e SVR com os mesmos hiperparâmetros) ─────────────
X = 'XAI'
add(X, 'xai', 'modelos_explicados', 'RF, GB e SVR com HIPERPARAMETROS e make_estimator (os mesmos da previsão)', loc(pv.treinar_modelos_xai))
add(X, 'xai', 'treino', 'série[:-H] (última janela); ruído σ = 30% do desvio do treino só no treino na condição ruido30', loc(pv.treino_xai))
add(X, 'xai', 'minimo_linhas', pv.MIN_LINHAS_XAI, 'pipeline_v2.py (MIN_LINHAS_XAI)', 'abaixo disso a série não é explicada')
add(X, 'xai', 'svr_alvo_constante', 'SVR não é treinado nem explicado (sem fallback)', loc(pv.treinar_modelos_xai))
add(X, 'xai', 'pontos_explicados', f'{pv.N_PONTOS_XAI - 1} linhas de treino mais recentes + ponto de previsão, iguais para SHAP e LIME',
    loc(pv.pontos_explicados))
add(X, 'xai', 'shap_arvores', 'TreeExplainer (exato)', loc(pv.explicar_pontos))
add(X, 'xai', 'shap_svr', f'KernelExplainer, fundo de {pv.SHAP_BACKGROUND} linhas sorteadas, nsamples = 2p + 2048, escala original do alvo',
    loc(pv.explicar_pontos))
add(X, 'xai', 'lime', f'LimeTabularExplainer modo regressão, num_samples={pv.LIME_NUM_SAMPLES}, num_features = p, coeficientes por índice (as_map()[1])',
    loc(pv.explicar_pontos))
for k in ['discretize_continuous', 'kernel_width', 'feature_selection', 'sample_around_instance']:
    add(X, 'xai', f'lime_{k}', padrao(LimeTabularExplainer, '__init__', k), 'padrão lime',
        'kernel_width None = 0,75 x sqrt(p)' if k == 'kernel_width' else '')
add(X, 'semente', 'subconjunto_30_series', f'default_rng({pv.SEED + 1}).choice', 'scripts/run_xai_v2.py')
add(X, 'semente', 'ruido_xai', f'default_rng([{pv.SEED}, serie_idx, n_treino, {pv.CENARIO_ID["XAI_ruido_30pct"]}])', loc(pv.treino_xai))
add(X, 'semente', 'shap_svr', f'default_rng({pv.SEED}) para o fundo e np.random.seed({pv.SEED}) para o KernelExplainer', loc(pv.explicar_pontos))
add(X, 'semente', 'lime', f'random_state={pv.SEED}', loc(pv.explicar_pontos))

df = pd.DataFrame(linhas)
saida = RAIZ / 'results' / 'v2'
df.to_csv(saida / 'hiperparametros_v2.csv', index=False, encoding='utf-8')

# ── Markdown ─────────────────────────────────────────────────────────────────
CAT = {'hiperparametro': 'Hiperparâmetros', 'entrada': 'Entrada', 'regra_selecao': 'Regra de seleção',
       'semente': 'Sementes', 'fallback': 'Fallback', 'sinalizacao': 'Sinalização', 'protocolo': 'Protocolo',
       'xai': 'Configuração'}


def cel(x):
    return str(x).replace('|', '\\|')


md = ['# Hiperparâmetros efetivos do protocolo v2', '',
      f"Gerado por `tools/gerar_hiperparametros_v2.py` a partir de `src/pipeline_v2.py` (sha256 `{SHA[:8]}…`, o mesmo dos caches v2). "
      f"Valores que o pipeline deixa no padrão da biblioteca foram lidos dos objetos instanciados (`get_params()`) e das assinaturas "
      f"das classes, nas versões usadas: scikit-learn {V['scikit-learn']}, statsmodels {V['statsmodels']}, shap {V['shap']}, "
      f"numpy {V['numpy']}, Python {V['python']}. A coluna Origem diz onde cada valor está definido.", '',
      'Fallbacks ficam registrados no status da janela (FALLBACK_SNAIVE ou FALLBACK_MEAN) e o motivo no campo `info`; '
      'as tabelas de resultados trazem as versões com e sem fallback.', '']
ordem = list(NOMES.values()) + ['Todos', 'XAI']
titulos = {'Todos': 'Regras comuns a todos os modelos', 'XAI': 'Explicabilidade (RF, GB e SVR)'}
for mod in ordem:
    sub = df[df.modelo == mod]
    fam = sub.familia.iloc[0]
    md += [f'## {titulos.get(mod, mod)}' + (f' ({fam})' if mod not in titulos else ''), '']
    for cat in CAT:
        s = sub[sub.categoria == cat]
        if s.empty:
            continue
        md += [f'**{CAT[cat]}**', '', '| Parâmetro | Valor | Origem | Observação |', '|---|---|---|---|']
        md += [f'| {cel(r.parametro)} | {cel(r.valor)} | {cel(r.origem)} | {cel(r.observacao)} |' for r in s.itertuples()]
        md.append('')
texto = '\n'.join(md)
assert '—' not in texto and not df.apply(lambda c: c.astype(str).str.contains('—')).any().any(), 'travessão'
(saida / 'hiperparametros_v2.md').write_text(texto, encoding='utf-8')
print(f'{len(df)} linhas; {df.modelo.nunique()} grupos -> results/v2/hiperparametros_v2.csv e .md')
