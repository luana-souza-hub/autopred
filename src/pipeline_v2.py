"""Pipeline v2 do projeto autopred: protocolo único corrigido.

Usado por M4 Monthly, Car Parts e Tourism Monthly (Fases 2 a 4). Correções em
relação aos notebooks 01 a 10 (ver auditoria, itens C1 a C20):

- C1:  previsão recursiva sem deslocamento. O vetor do próximo passo é montado
       pela MESMA função que gera as linhas de treino (_linha_features).
- C2:  perturbações aplicadas só ao treino de cada janela, com estatísticas do
       próprio treino; avaliação sempre contra o teste limpo. "Alvo ruidoso"
       existe apenas como análise suplementar (alvo_ruidoso=True).
- C4:  SARIMA automático escolhe a ordem em cada janela usando só o treino dela.
- C5:  sem a coluna duplicada lag_season (= lag_12); lag_24 só na análise
       suplementar de explicabilidade (com_lag24=True).
- C6:  amostra piloto única do M4 (ordem do arquivo, SEED=42).
- C8:  SHAP do SVR com semente fixa, nsamples >= 2*p + 2048, escala original.
- C10: importâncias normalizadas por série antes da agregação.
- C12: status separa SEM_JANELA, FALLBACK_* e ERRO.
- C13: R² indefinido (NaN) quando o teste é constante.
- C14: fallbacks de HW e SARIMA registrados no status.
- SARIMA com verossimilhança degenerada (observações efetivas <= nº de parâmetros,
       ou llf nulo/não finito, típico de janelas de 24 obs com diferença sazonal)
       é rejeitado na seleção e, no heurístico, vira FALLBACK_SNAIVE registrado.
- S1:  truncamento mantém a fração mais recente do treino; o teste não muda.
- LIME: associação de termos às features por índice (exp.as_map()[1]).
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings('ignore')

# ── Protocolo ────────────────────────────────────────────────────────────────
SEED      = 42
MIN_TRAIN = 24
N_ORIGINS = 3
SEASON    = 12
N_LAGS    = 12

HIPERPARAMETROS = {
    'RandomForest'    : dict(n_estimators=100, min_samples_leaf=2, random_state=SEED, n_jobs=1),
    'GradientBoosting': dict(n_estimators=100, learning_rate=0.1, max_depth=3,
                             subsample=0.8, random_state=SEED),
    'SVR'             : dict(C=1.0, epsilon=0.1, kernel='rbf', gamma='scale'),
}
SARIMA_HEUR_ORDER    = (1, 1, 1)
SARIMA_HEUR_SEASONAL = (1, 1, 0, SEASON)
SARIMA_MAXITER       = 50
SARIMA_TIME_LIMIT_S  = 60.0          # por ajuste do SARIMA heurístico
SARIMA_AUTO_GRID     = dict(p=[0, 1, 2], d=[0, 1], q=[0, 1, 2], P=[0, 1], D=[0, 1], Q=[0, 1])
SARIMA_AUTO_TIME_LIMIT_S = 300.0     # por seleção (por janela), como no NB04b

# Outliers: o NB08 executou média ± 3σ (o texto da dissertação diz 5σ; item C16)
OUTLIER_K = 3.0

LIME_NUM_SAMPLES = 200
SHAP_BACKGROUND  = 50
SHAP_N_PONTOS    = 30
N_EXPL           = 30
MIN_LINHAS_XAI   = 10

# Status por janela
OK, FB_SNAIVE, FB_MEAN, ERRO, SEM_JANELA = 'OK', 'FALLBACK_SNAIVE', 'FALLBACK_MEAN', 'ERRO', 'SEM_JANELA'

# Cenários adversos do NB08 (os 9), com identificador numérico para a semente
CENARIOS = {
    'A_ruido_10pct'    : ('ruido',    0.10, 'Ruído σ=10%'),
    'A_ruido_30pct'    : ('ruido',    0.30, 'Ruído σ=30%'),
    'B_ausencias_10pct': ('ausencia', 0.10, 'Ausências 10%'),
    'B_ausencias_20pct': ('ausencia', 0.20, 'Ausências 20%'),
    'B_ausencias_30pct': ('ausencia', 0.30, 'Ausências 30%'),
    'C_outliers_2pct'  : ('outlier',  0.02, 'Outliers 2%'),
    'C_outliers_5pct'  : ('outlier',  0.05, 'Outliers 5%'),
    'D_trunca_50pct'   : ('trunca',   0.50, 'Truncamento 50%'),
    'D_trunca_25pct'   : ('trunca',   0.25, 'Truncamento 25%'),
}
CENARIO_ID = {nome: i + 1 for i, nome in enumerate(CENARIOS)}
CENARIO_ID['XAI_ruido_30pct'] = 100   # ruído da análise de estabilidade das explicações


# ── Leitura de dados ─────────────────────────────────────────────────────────
def load_tsf(path, encoding='utf-8'):
    """Leitor TSF do NB10 (encoding parametrizável). Retorna {id: array} na ordem do arquivo."""
    series, in_data = {}, False
    with open(path, encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.lower() == '@data':
                in_data = True
                continue
            if line.startswith('@') or not in_data:
                continue
            parts = line.split(':')
            if len(parts) < 3:
                continue
            series[parts[0].strip()] = np.array([float(v) for v in parts[-1].split(',') if v.strip()])
    return series


def read_tsf_header(path, encoding='latin-1'):
    meta = {}
    with open(path, encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if line.lower() == '@data':
                break
            if line.startswith('@') and not line.startswith('@attribute'):
                k, _, v = line[1:].partition(' ')
                meta[k] = v.strip()
    return meta


def sortear_indices(n_total, n, seed=SEED):
    """Regra dos NB05 a 09: default_rng(seed).choice sobre a ordem do arquivo."""
    if n is None or n >= n_total:
        return np.arange(n_total)
    return np.random.default_rng(seed).choice(n_total, size=n, replace=False)


def load_m4_piloto(data_dir, n=200, seed=SEED):
    """Amostra piloto única do M4 (C6): a dos NB05 a 09. Retorna (ids, lista de arrays)."""
    raw = pd.read_csv(Path(data_dir) / 'Monthly-train.csv', header=0, index_col=0)
    idx = sortear_indices(len(raw), n, seed)
    ids = [raw.index[i] for i in idx]
    return ids, [raw.iloc[i].dropna().values.astype(float) for i in idx]


def amostra_tsf(series_dict, n, seed=SEED):
    ids_arquivo = list(series_dict)
    idx = sortear_indices(len(ids_arquivo), n, seed)
    ids = [ids_arquivo[i] for i in idx]
    return ids, [series_dict[u] for u in ids]


# ── Janelas e métricas ───────────────────────────────────────────────────────
def make_rolling_windows(n, h, min_train=MIN_TRAIN, n_origins=N_ORIGINS):
    """Origens por linspace(MIN_TRAIN, n - H) (NB01 e NB10). Treino = [:o], teste = [o:o+h]."""
    if n < min_train + h or n - h <= min_train:
        return []
    origins = sorted(set(np.linspace(min_train, n - h, n_origins, dtype=int).tolist()))
    return [o for o in origins if o + h <= n]


def calc_mape(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def calc_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        'mape': calc_mape(y_true, y_pred),
        'mae' : float(mean_absolute_error(y_true, y_pred)),
        'r2'  : float(r2_score(y_true, y_pred)) if np.std(y_true) > 0 else np.nan,   # C13
    }


# ── Features (C5: sem duplicata) ─────────────────────────────────────────────
def feature_names(n_lags=N_LAGS, com_lag24=False):
    nomes = [f'lag_{i}' for i in range(1, n_lags + 1)] + ['roll_mean_3', 'roll_std_3', 'roll_mean_6']
    return nomes + (['lag_24'] if com_lag24 else [])


def historico_minimo(n_lags=N_LAGS, com_lag24=False):
    return 24 if com_lag24 else max(n_lags, 6)


def _linha_features(hist, n_lags=N_LAGS, com_lag24=False):
    """Features do passo seguinte ao fim de `hist` (usa só valores já conhecidos)."""
    h = np.asarray(hist, dtype=float)
    linha = [h[-i] for i in range(1, n_lags + 1)]
    linha += [h[-3:].mean(), h[-3:].std(ddof=1), h[-6:].mean()]
    if com_lag24:
        linha.append(h[-24])
    return linha


def build_features(series, n_lags=N_LAGS, com_lag24=False):
    """Matriz supervisionada: linha t usa series[:t] e tem alvo series[t]."""
    s = np.asarray(series, dtype=float)
    m = historico_minimo(n_lags, com_lag24)
    nomes = feature_names(n_lags, com_lag24)
    if len(s) <= m:
        return np.empty((0, len(nomes))), np.empty(0), nomes
    X = np.array([_linha_features(s[:t], n_lags, com_lag24) for t in range(m, len(s))])
    return X, s[m:], nomes


def next_step_features(history, n_lags=N_LAGS, com_lag24=False):
    return np.array(_linha_features(history, n_lags, com_lag24)).reshape(1, -1)


# ── Modelos (cada um retorna (previsão, status) ou (previsão, status, info)) ─
def model_persistence(train, h):
    return np.full(h, train[-1]), OK


def model_mean(train, h):
    return np.full(h, np.mean(train)), OK


def _seasonal_naive(train, h, season=SEASON):
    n = len(train)
    if n < season:
        return np.full(h, train[-1])
    return np.array([train[min(n - season + ((k - 1) % season), n - 1)] for k in range(1, h + 1)])


def model_seasonal_naive(train, h):
    return _seasonal_naive(train, h), OK


def model_ses(train, h):
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.asarray(SimpleExpSmoothing(train, initialization_method='estimated')
                              .fit(optimized=True).forecast(h)), OK
    except Exception:
        return np.full(h, np.mean(train)), FB_MEAN


def model_holt_winters(train, h, season=SEASON):
    """NB02: tendência aditiva, sazonalidade aditiva ou multiplicativa por AIC."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    if len(train) < 2 * season:
        return _seasonal_naive(train, h), FB_SNAIVE
    melhor, aic = None, np.inf
    for tipo in ['add'] + (['mul'] if np.all(train > 0) else []):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fit = ExponentialSmoothing(train, trend='add', seasonal=tipo, seasonal_periods=season,
                                           initialization_method='estimated').fit(optimized=True)
            if fit.aic < aic:
                melhor, aic = (fit, tipo), fit.aic
        except Exception:
            continue
    if melhor is None:
        return _seasonal_naive(train, h), FB_SNAIVE
    return np.asarray(melhor[0].forecast(h)), OK, {'sazonal': melhor[1]}


def _sarimax_fit(train, order, seasonal):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():   # o statsmodels reativa ConvergenceWarning ao ser importado
        warnings.simplefilter('ignore')
        return SARIMAX(train, order=order, seasonal_order=seasonal, enforce_stationarity=False,
                       enforce_invertibility=False).fit(disp=False, maxiter=SARIMA_MAXITER)


def verossimilhanca_degenerada(fit):
    """AIC indefinido: a inicialização difusa consome as observações (n efetivo <= nº de parâmetros)."""
    n_ef = fit.nobs - fit.loglikelihood_burn
    return bool(n_ef <= len(fit.params) or not np.isfinite(fit.llf) or fit.llf == 0)


def _diagnostico_previsao(pred, train):
    """Marca previsões explosivas (|previsão| > 10x a amplitude do treino) sem alterá-las."""
    amp = np.max(np.abs(train)) + 1e-12
    return {'explosiva': True} if np.max(np.abs(pred)) > 10 * amp else {}


def model_sarima_heuristic(train, h):
    if np.std(train) == 0:
        return _seasonal_naive(train, h), FB_SNAIVE, {'motivo': 'treino_constante'}
    t0 = time.perf_counter()
    try:
        fit = _sarimax_fit(train, SARIMA_HEUR_ORDER, SARIMA_HEUR_SEASONAL)
    except Exception:
        return _seasonal_naive(train, h), FB_SNAIVE, {'motivo': 'excecao'}
    if time.perf_counter() - t0 > SARIMA_TIME_LIMIT_S:
        return _seasonal_naive(train, h), FB_SNAIVE, {'motivo': 'tempo'}
    if verossimilhanca_degenerada(fit):
        return _seasonal_naive(train, h), FB_SNAIVE, {'motivo': 'verossimilhanca_degenerada'}
    pred = np.asarray(fit.forecast(h))
    return pred, OK, _diagnostico_previsao(pred, train)


def selecionar_ordem_sarima(train, time_limit=SARIMA_AUTO_TIME_LIMIT_S):
    """Busca por AIC (grade do NB04b) usando SÓ o treino da janela (C4)."""
    g = SARIMA_AUTO_GRID
    melhor, aic, n_conv, n_deg, t0, completa = None, np.inf, 0, 0, time.perf_counter(), True
    for p, d, q, P, D, Q in product(g['p'], g['d'], g['q'], g['P'], g['D'], g['Q']):
        if time.perf_counter() - t0 > time_limit:
            completa = False
            break
        if p == q == P == Q == 0:
            continue
        try:
            fit = _sarimax_fit(train, (p, d, q), (P, D, Q, SEASON))
            n_conv += 1
            if verossimilhanca_degenerada(fit):
                n_deg += 1
                continue
            if fit.aic < aic:
                melhor, aic = ((p, d, q), (P, D, Q, SEASON)), fit.aic
        except Exception:
            continue
    return melhor, {'aic': float(aic), 'n_convergiu': n_conv, 'n_degenerados': n_deg, 'busca_completa': completa,
                    'sel_s': round(time.perf_counter() - t0, 2)}


def model_sarima_auto(train, h):
    if np.std(train) == 0:
        return _seasonal_naive(train, h), FB_SNAIVE
    escolha, info = selecionar_ordem_sarima(train)
    if escolha is None:
        return _seasonal_naive(train, h), FB_SNAIVE, {**info, 'motivo': 'sem_candidato_valido'}
    order, seasonal = escolha
    info.update(order=str(order), seasonal=str(seasonal))
    try:
        pred = np.asarray(_sarimax_fit(train, order, seasonal).forecast(h))
    except Exception:
        return _seasonal_naive(train, h), FB_SNAIVE, {**info, 'motivo': 'excecao'}
    return pred, OK, {**info, **_diagnostico_previsao(pred, train)}


def make_estimator(nome):
    if nome == 'RandomForest':
        return RandomForestRegressor(**HIPERPARAMETROS['RandomForest'])
    if nome == 'GradientBoosting':
        return GradientBoostingRegressor(**HIPERPARAMETROS['GradientBoosting'])
    if nome == 'SVR':
        return SVR(**HIPERPARAMETROS['SVR'])
    raise ValueError(nome)


def ml_forecast(train, h, estimator, escalar=False, com_lag24=False):
    """Previsão recursiva (C1): cada passo usa next_step_features(histórico + previsões)."""
    X, y, _ = build_features(train, com_lag24=com_lag24)
    if len(X) < 2 or (escalar and np.std(y) == 0):
        return np.full(h, np.mean(train)), FB_MEAN
    if escalar:
        sx, sy = StandardScaler().fit(X), StandardScaler().fit(y.reshape(-1, 1))
        estimator.fit(sx.transform(X), sy.transform(y.reshape(-1, 1)).ravel())
    else:
        estimator.fit(X, y)
    hist, preds = list(train), []
    for _ in range(h):
        x = next_step_features(hist, com_lag24=com_lag24)
        if escalar:
            p = sy.inverse_transform(estimator.predict(sx.transform(x)).reshape(-1, 1))[0, 0]
        else:
            p = estimator.predict(x)[0]
        preds.append(p)
        hist.append(p)
    return np.array(preds), OK


def model_random_forest(train, h):
    return ml_forecast(train, h, make_estimator('RandomForest'))


def model_gbt(train, h):
    return ml_forecast(train, h, make_estimator('GradientBoosting'))


def model_svr(train, h):
    return ml_forecast(train, h, make_estimator('SVR'), escalar=True)


MODELOS = {
    'persistence'     : model_persistence,
    'mean'            : model_mean,
    'seasonal_naive'  : model_seasonal_naive,
    'ses'             : model_ses,
    'holt_winters'    : model_holt_winters,
    'sarima_heuristic': model_sarima_heuristic,
    'sarima_auto'     : model_sarima_auto,
    'random_forest'   : model_random_forest,
    'gbt'             : model_gbt,
    'svr'             : model_svr,
}
FAMILIA = {
    'persistence': 'Baseline', 'mean': 'Baseline', 'seasonal_naive': 'Baseline',
    'ses': 'Suavização exponencial', 'holt_winters': 'Suavização exponencial',
    'sarima_heuristic': 'ARIMA', 'sarima_auto': 'ARIMA',
    'random_forest': 'ML', 'gbt': 'ML', 'svr': 'ML',
}
MODELOS_RAPIDOS = [m for m in MODELOS if m != 'sarima_auto']


# ── Perturbações (C2, S1) ────────────────────────────────────────────────────
def rng_janela(serie_idx, origin, cenario):
    """Semente por (série, janela, cenário): a mesma perturbação para todos os modelos."""
    return np.random.default_rng([SEED, int(serie_idx), int(origin), CENARIO_ID[cenario]])


def perturbar(segmento, cenario, rng, min_train=MIN_TRAIN):
    """Aplica o cenário a um segmento, com estatísticas do próprio segmento."""
    tipo, frac, _ = CENARIOS[cenario] if cenario in CENARIOS else ('ruido', 0.30, '')
    s = np.asarray(segmento, dtype=float).copy()
    n = len(s)
    if tipo == 'ruido':
        return s + rng.normal(0, np.std(s) * frac, size=n)
    if tipo == 'ausencia':
        idx = rng.choice(n, size=max(1, int(n * frac)), replace=False)
        s[idx] = np.nan
        return pd.Series(s).interpolate(method='linear', limit_direction='both').values
    if tipo == 'outlier':
        idx = rng.choice(n, size=max(1, int(n * frac)), replace=False)
        sinais = rng.choice([-1, 1], size=len(idx))
        s[idx] = np.mean(segmento) + sinais * OUTLIER_K * np.std(segmento)
        return s
    if tipo == 'trunca':
        return s[-max(min_train, int(n * frac)):]
    raise ValueError(cenario)


# ── Avaliação ────────────────────────────────────────────────────────────────
def _chamar(fn, train, h):
    out = fn(train, h)
    pred, status = out[0], out[1]
    info = out[2] if len(out) > 2 else {}
    pred = np.asarray(pred, dtype=float)
    if pred.shape != (h,) or not np.all(np.isfinite(pred)):
        return _seasonal_naive(train, h), FB_SNAIVE, {**info, 'motivo': 'previsao_invalida'}
    return pred, status, info


def avaliar_serie(series, unique_id, serie_idx, modelos, h, cenario=None, alvo_ruidoso=False,
                  min_train=MIN_TRAIN, n_origins=N_ORIGINS):
    """Avalia uma série em todas as janelas. `modelos`: nomes de MODELOS ou dict nome->função."""
    fns = modelos if isinstance(modelos, dict) else {m: MODELOS[m] for m in modelos}
    series = np.asarray(series, dtype=float)
    base = {'unique_id': unique_id, 'serie_idx': serie_idx, 'cenario': cenario or 'limpo',
            'alvo_ruidoso': bool(alvo_ruidoso)}
    origens = make_rolling_windows(len(series), h, min_train, n_origins)
    if not origens:
        return [{**base, 'model': m, 'janela': None, 'origin': None, 'n_train': None, 'n_train_efetivo': None,
                 'mape': np.nan, 'mae': np.nan, 'r2': np.nan, 'exec_time_s': np.nan,
                 'status': SEM_JANELA, 'info': ''} for m in fns]
    regs = []
    for k, o in enumerate(origens, start=1):
        train, test = series[:o], series[o:o + h]
        if cenario is not None:
            rng = rng_janela(serie_idx, o, cenario)
            if alvo_ruidoso:
                if CENARIOS[cenario][0] == 'trunca':
                    raise ValueError('alvo_ruidoso não se aplica ao truncamento')
                seg = perturbar(series[:o + h], cenario, rng, min_train)
                train, test = seg[:o], seg[o:]
            else:
                train = perturbar(train, cenario, rng, min_train)   # teste continua limpo
        for nome, fn in fns.items():
            t0 = time.perf_counter()
            try:
                pred, status, info = _chamar(fn, train, h)
                met = calc_metrics(test, pred)
            except Exception as e:
                met, status, info = {'mape': np.nan, 'mae': np.nan, 'r2': np.nan}, ERRO, {'erro': str(e)[:120]}
            regs.append({**base, 'model': nome, 'janela': k, 'origin': o, 'n_train': o, 'n_train_efetivo': len(train),
                         **met, 'exec_time_s': time.perf_counter() - t0, 'status': status,
                         'info': json.dumps(info, ensure_ascii=False) if info else ''})
    return regs


def avaliar_base(series_list, ids, modelos, h, cenario=None, alvo_ruidoso=False, n_jobs=10,
                 min_train=MIN_TRAIN, n_origins=N_ORIGINS):
    """Avalia todas as séries em paralelo por série (joblib, processos)."""
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=n_jobs)(
        delayed(avaliar_serie)(s, u, i, modelos, h, cenario, alvo_ruidoso, min_train, n_origins)
        for i, (u, s) in enumerate(zip(ids, series_list)))
    return pd.DataFrame([r for regs in res for r in regs])


def flag_explosiva(df):
    """Janelas com previsão explosiva sinalizada no campo info (status continua OK)."""
    return df['info'].fillna('').astype(str).str.contains('"explosiva": true', regex=False)


def agregar_por_serie(df, incluir_fallback=True, excluir_explosivas=False):
    """Mediana das janelas por (série, modelo). ERRO e SEM_JANELA ficam fora."""
    validos = [OK, FB_SNAIVE, FB_MEAN] if incluir_fallback else [OK]
    d = df[df['status'].isin(validos)]
    if excluir_explosivas:
        d = d[~flag_explosiva(d)]
    return d.groupby(['model', 'unique_id'])[['mape', 'mae', 'r2', 'exec_time_s']].median().reset_index()


def resumo_modelos(by_series, ordenar_por='mape'):
    r = by_series.groupby('model')[['mape', 'mae', 'r2', 'exec_time_s']].median()
    r['familia'] = r.index.map(FAMILIA)
    r['n_series'] = by_series.groupby('model')['unique_id'].nunique()
    return r.sort_values(ordenar_por)


def taxas_status(df):
    """Proporção de janelas por status e modelo (C12, C14)."""
    t = pd.crosstab(df['model'], df['status'], normalize='index')
    for c in [OK, FB_SNAIVE, FB_MEAN, ERRO, SEM_JANELA]:
        if c not in t:
            t[c] = 0.0
    return t[[OK, FB_SNAIVE, FB_MEAN, ERRO, SEM_JANELA]]


def taxas_fallback_por_janela(df):
    """Proporção de janelas em fallback por modelo e janela (1ª, 2ª, 3ª)."""
    d = df[df['status'] != SEM_JANELA].copy()
    d['fallback'] = d['status'].isin([FB_SNAIVE, FB_MEAN])
    return d.pivot_table(index='model', columns='janela', values='fallback', aggfunc='mean')


def tabela_delta(by_series_base, by_series_adv, metrica='mape'):
    """ΔMAPE em % relativo e em p.p. absolutos, nas MESMAS séries (pareado)."""
    m = by_series_base.merge(by_series_adv, on=['model', 'unique_id'], suffixes=('_base', '_adv'))
    m = m.dropna(subset=[f'{metrica}_base', f'{metrica}_adv'])
    g = m.groupby('model')
    out = pd.DataFrame({
        f'{metrica}_base': g[f'{metrica}_base'].median(),
        f'{metrica}_adv' : g[f'{metrica}_adv'].median(),
    })
    out['delta_pp']  = out[f'{metrica}_adv'] - out[f'{metrica}_base']
    out['delta_pct'] = 100 * out['delta_pp'] / out[f'{metrica}_base']
    out['delta_pp_pareado_mediana'] = g.apply(lambda x: (x[f'{metrica}_adv'] - x[f'{metrica}_base']).median())
    out['n_series'] = g.size()
    return out


# ── Explicabilidade ──────────────────────────────────────────────────────────
MODELOS_XAI = ['RandomForest', 'GradientBoosting', 'SVR']


def treinar_modelos_xai(train, com_lag24=False):
    """Treina RF, GB e SVR (hiperparâmetros do protocolo) no treino dado."""
    X, y, nomes = build_features(train, com_lag24=com_lag24)
    if len(X) < MIN_LINHAS_XAI:
        return None
    out = {'nomes': nomes}
    for nome in ['RandomForest', 'GradientBoosting']:
        out[nome] = {'model': make_estimator(nome).fit(X, y), 'X': X, 'y': y, 'sx': None, 'sy': None}
    if np.std(y) > 0:
        sx, sy = StandardScaler().fit(X), StandardScaler().fit(y.reshape(-1, 1))
        svr = make_estimator('SVR').fit(sx.transform(X), sy.transform(y.reshape(-1, 1)).ravel())
        out['SVR'] = {'model': svr, 'X': X, 'y': y, 'sx': sx, 'sy': sy}
    return out


def _entrada_e_predict(entry):
    """Espaço de entrada do modelo e função de previsão na ESCALA ORIGINAL do alvo."""
    m, sx, sy = entry['model'], entry['sx'], entry['sy']
    X_in = sx.transform(entry['X']) if sx is not None else entry['X']
    if sy is not None:
        return X_in, lambda x: sy.inverse_transform(m.predict(x).reshape(-1, 1)).ravel()
    return X_in, m.predict


def shap_importancias(entry, nome, seed=SEED):
    """Média de |SHAP|. Árvores: TreeExplainer exato. SVR: KernelExplainer semeado (C8)."""
    import shap
    X_in, f = _entrada_e_predict(entry)
    if nome in ('RandomForest', 'GradientBoosting'):
        return np.abs(shap.TreeExplainer(entry['model']).shap_values(X_in)).mean(axis=0)
    rng = np.random.default_rng(seed)
    bg = X_in[rng.choice(len(X_in), size=min(SHAP_BACKGROUND, len(X_in)), replace=False)]
    pts = X_in[rng.choice(len(X_in), size=min(SHAP_N_PONTOS, len(X_in)), replace=False)]
    np.random.seed(seed)   # o KernelExplainer usa o gerador global do numpy
    vals = shap.KernelExplainer(f, bg).shap_values(pts, nsamples=2 * X_in.shape[1] + 2048, silent=True)
    return np.abs(vals).mean(axis=0)


def lime_importancias(entry, nomes, num_samples=LIME_NUM_SAMPLES, seed=SEED):
    """|coef| do LIME para o último ponto de treino, associados POR ÍNDICE (as_map()[1])."""
    from lime.lime_tabular import LimeTabularExplainer
    X_in, f = _entrada_e_predict(entry)
    exp = LimeTabularExplainer(X_in, feature_names=nomes, mode='regression', random_state=seed,
                               verbose=False).explain_instance(X_in[-1], f, num_features=len(nomes),
                                                               num_samples=num_samples)
    coefs = np.zeros(len(nomes))
    for j, w in exp.as_map()[1]:
        coefs[j] = abs(w)
    return coefs


N_PONTOS_XAI = 30


def treino_xai(series, h, condicao, serie_idx):
    """Treino da última janela (série[:-h]); com ruído σ=30% do desvio do próprio treino se pedido."""
    train = np.asarray(series, dtype=float)[:-h]
    if condicao == 'ruido30':
        rng = np.random.default_rng([SEED, int(serie_idx), len(train), CENARIO_ID['XAI_ruido_30pct']])
        train = perturbar(train, 'XAI_ruido_30pct', rng)
    elif condicao != 'limpo':
        raise ValueError(condicao)
    return train


def pontos_explicados(train, com_lag24=False, n_pontos=N_PONTOS_XAI):
    """Os MESMOS pontos para SHAP e LIME: as n_pontos-1 linhas de treino mais recentes + o ponto de previsão."""
    X, _, _ = build_features(train, com_lag24=com_lag24)
    recentes = X[-(n_pontos - 1):]
    pts = np.vstack([recentes, next_step_features(train, com_lag24=com_lag24)])
    return pts, ['treino_recente'] * len(recentes) + ['previsao']


def explicar_pontos(fit, nome, pts, metodo, lime_samples=LIME_NUM_SAMPLES, seed=SEED):
    """|atribuição| por ponto e feature (matriz n_pontos x p), na escala original do alvo."""
    entry = fit[nome]
    X_in, f = _entrada_e_predict(entry)
    P = entry['sx'].transform(pts) if entry['sx'] is not None else pts
    p = X_in.shape[1]
    if metodo == 'shap':
        import shap
        if nome in ('RandomForest', 'GradientBoosting'):
            return np.abs(np.asarray(shap.TreeExplainer(entry['model']).shap_values(P)))
        rng = np.random.default_rng(seed)
        bg = X_in[rng.choice(len(X_in), size=min(SHAP_BACKGROUND, len(X_in)), replace=False)]
        np.random.seed(seed)   # o KernelExplainer usa o gerador global do numpy
        vals = shap.KernelExplainer(f, bg).shap_values(P, nsamples=2 * p + 2048, silent=True)
        return np.abs(np.asarray(vals))
    if metodo == 'lime':
        from lime.lime_tabular import LimeTabularExplainer
        exp_ = LimeTabularExplainer(X_in, feature_names=fit['nomes'], mode='regression',
                                    random_state=seed, verbose=False)
        out = np.zeros((len(P), p))
        for i, x in enumerate(P):
            for j, w in exp_.explain_instance(x, f, num_features=p, num_samples=lime_samples).as_map()[1]:
                out[i, j] = abs(w)
        return out
    raise ValueError(metodo)


def explicar_serie(series, unique_id, serie_idx, h, condicao, com_lag24=False, metodos=('shap', 'lime'),
                   lime_samples=LIME_NUM_SAMPLES):
    """Treina RF, GB e SVR na última janela e explica os mesmos pontos com cada método."""
    train = treino_xai(series, h, condicao, serie_idx)
    fit = treinar_modelos_xai(train, com_lag24)
    if fit is None:
        return []
    pts, tipos = pontos_explicados(train, com_lag24)
    regs = []
    for nome in MODELOS_XAI:
        if nome not in fit:
            continue
        for met in metodos:
            A = explicar_pontos(fit, nome, pts, met, lime_samples)
            for k, (a, tipo) in enumerate(zip(A, tipos)):
                regs.append({'serie': serie_idx, 'unique_id': unique_id, 'modelo': nome, 'metodo': met,
                             'condicao': condicao, 'ponto': k, 'tipo_ponto': tipo, 'n_train': len(train),
                             **dict(zip(fit['nomes'], a))})
    return regs


def importancia_por_serie(df_pontos, nomes, tipo_ponto=None):
    """Média de |atribuição| sobre os pontos explicados, por (série, modelo, método, condição)."""
    d = df_pontos if tipo_ponto is None else df_pontos[df_pontos['tipo_ponto'] == tipo_ponto]
    return d.groupby(['serie', 'modelo', 'metodo', 'condicao'])[nomes].mean().reset_index()


def bootstrap_ic(valores, n=2000, seed=SEED, estat=np.mean, alfa=0.05):
    """Estatística e IC percentil por bootstrap (reamostragem das unidades, ex.: séries)."""
    v = np.asarray(valores, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    idx = np.random.default_rng(seed).integers(0, len(v), size=(n, len(v)))
    boot = estat(v[idx], axis=1)
    return float(estat(v)), float(np.percentile(boot, 100 * alfa / 2)), float(np.percentile(boot, 100 * (1 - alfa / 2)))


def bootstrap_delta(by_series_base, by_series_adv, modelos, n=2000, seed=SEED, metrica='mape', alfa=0.05):
    """IC por bootstrap pareado por série (mesmas séries reamostradas para base, cenário e todos os
    modelos) para ΔMAPE das medianas, em % e p.p.; e IC da diferença de Δ% entre pares de modelos."""
    B = by_series_base.pivot(index='unique_id', columns='model', values=metrica)[modelos]
    A = by_series_adv.pivot(index='unique_id', columns='model', values=metrica)[modelos]
    comuns = B.dropna().index.intersection(A.dropna().index)
    B, A = B.loc[comuns].values, A.loc[comuns].values
    idx = np.random.default_rng(seed).integers(0, len(comuns), size=(n, len(comuns)))
    mb, ma = np.median(B[idx], axis=1), np.median(A[idx], axis=1)
    d_pp, d_pct = ma - mb, 100 * (ma - mb) / mb
    pb, pa = np.median(B, axis=0), np.median(A, axis=0)
    q = [100 * alfa / 2, 100 * (1 - alfa / 2)]
    tab = pd.DataFrame({
        'delta_pct': 100 * (pa - pb) / pb, 'ic_pct_inf': np.percentile(d_pct, q[0], axis=0),
        'ic_pct_sup': np.percentile(d_pct, q[1], axis=0),
        'delta_pp': pa - pb, 'ic_pp_inf': np.percentile(d_pp, q[0], axis=0), 'ic_pp_sup': np.percentile(d_pp, q[1], axis=0),
    }, index=modelos)
    tab['ic_inclui_zero'] = (tab['ic_pct_inf'] <= 0) & (tab['ic_pct_sup'] >= 0)
    tab['n_series'] = len(comuns)
    pares = []
    for i in range(len(modelos)):
        for j in range(i + 1, len(modelos)):
            dif = d_pct[:, i] - d_pct[:, j]
            lo, hi = np.percentile(dif, q)
            pares.append({'modelo_a': modelos[i], 'modelo_b': modelos[j],
                          'dif_delta_pct': tab['delta_pct'].iloc[i] - tab['delta_pct'].iloc[j],
                          'ic_inf': lo, 'ic_sup': hi, 'ic_inclui_zero': bool(lo <= 0 <= hi)})
    return tab, pd.DataFrame(pares)


def bootstrap_ranks(by_series, modelos, metrica='mape', n=2000, seed=SEED, alfa=0.05):
    """Ranking dos modelos pela mediana entre séries, com IC por bootstrap pareado por série
    (as mesmas séries reamostradas para todos os modelos). Retorna (tabela, medianas_bootstrap n x M)."""
    from scipy.stats import rankdata
    P = by_series.pivot(index='unique_id', columns='model', values=metrica)[modelos].dropna()
    V = P.values
    idx = np.random.default_rng(seed).integers(0, len(P), size=(n, len(P)))
    med = np.median(V[idx], axis=1)
    ranks = np.apply_along_axis(rankdata, 1, med, method='min')
    q = [100 * alfa / 2, 100 * (1 - alfa / 2)]
    tab = pd.DataFrame({
        'mediana': np.median(V, axis=0), 'mediana_ic_inf': np.percentile(med, q[0], axis=0),
        'mediana_ic_sup': np.percentile(med, q[1], axis=0),
        'rank': rankdata(np.median(V, axis=0), method='min').astype(int),
        'rank_ic_inf': np.percentile(ranks, q[0], axis=0), 'rank_ic_sup': np.percentile(ranks, q[1], axis=0),
        'n_series': len(P)}, index=modelos)
    return tab, pd.DataFrame(med, columns=modelos)


def normalizar_por_serie(df, nomes):
    """Divide cada linha (série, modelo) pela soma das importâncias (C10)."""
    d = df.copy()
    tot = d[nomes].sum(axis=1).replace(0, np.nan)
    d[nomes] = d[nomes].div(tot, axis=0)
    return d


def importancia_global(df, nomes, modelo, normalizar=True):
    d = normalizar_por_serie(df, nomes) if normalizar else df
    return d[d['modelo'] == modelo][nomes].mean()


def jaccard_topk(a, b, k):
    ta, tb = set(np.argsort(-np.asarray(a))[:k]), set(np.argsort(-np.asarray(b))[:k])
    return len(ta & tb) / len(ta | tb)


def estabilidade(df_limpo, df_pert, nomes, ks=(3, 5)):
    """Por (série, modelo): Spearman e Jaccard top-k entre limpo e perturbado."""
    from scipy.stats import spearmanr
    m = df_limpo.merge(df_pert, on=['serie', 'modelo'], suffixes=('_l', '_p'))
    linhas = []
    for _, r in m.iterrows():
        a = r[[f'{n}_l' for n in nomes]].values.astype(float)
        b = r[[f'{n}_p' for n in nomes]].values.astype(float)
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        lin = {'serie': r['serie'], 'modelo': r['modelo'], 'spearman': spearmanr(a, b)[0]}
        for k in ks:
            lin[f'jaccard_top{k}'] = jaccard_topk(a, b, k)
        linhas.append(lin)
    return pd.DataFrame(linhas)


def concordancia_shap_lime(df_shap, df_lime, nomes, modelo, normalizar=True):
    """Pearson (normalização pelo máximo, como na Fig. 4 do NB09) e Spearman entre importâncias globais."""
    from scipy.stats import spearmanr
    s = importancia_global(df_shap, nomes, modelo, normalizar)
    l = importancia_global(df_lime, nomes, modelo, normalizar)
    r = np.corrcoef((s / s.max()).values, (l / l.max()).values)[0, 1]
    return {'modelo': modelo, 'pearson': float(r), 'spearman': float(spearmanr(s.values, l.values)[0])}


# ── Metadados de cache ───────────────────────────────────────────────────────
def hash_pipeline():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def versoes():
    import scipy, sklearn, statsmodels
    v = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
         'scikit-learn': sklearn.__version__, 'statsmodels': statsmodels.__version__, 'scipy': scipy.__version__}
    for mod in ['shap', 'lime', 'joblib']:
        try:
            v[mod] = getattr(__import__(mod), '__version__', 'instalado')
        except ImportError:
            v[mod] = None
    return v


_HASH_CACHE = {}


def info_arquivos(paths):
    """Nome, tamanho e SHA-256 dos arquivos de dados (calculado uma vez por processo)."""
    out = []
    for p in paths or []:
        p = Path(p)
        chave = (str(p.resolve()), p.stat().st_mtime, p.stat().st_size)
        if chave not in _HASH_CACHE:
            h = hashlib.sha256()
            with open(p, 'rb') as f:
                for bloco in iter(lambda: f.read(1 << 20), b''):
                    h.update(bloco)
            _HASH_CACHE[chave] = h.hexdigest()
        out.append({'arquivo': p.name, 'bytes': p.stat().st_size, 'sha256': _HASH_CACHE[chave]})
    return out


def write_meta(path, extra=None, arquivos_dados=None):
    """Grava <arquivo>.meta.json ao lado do cache (inclui SHA-256 dos dados usados)."""
    path = Path(path)
    meta = {
        'arquivo'       : path.name,
        'gerado_em'     : datetime.now().isoformat(timespec='seconds'),
        'pipeline'      : 'src/pipeline_v2.py',
        'pipeline_sha256': hash_pipeline(),
        'versoes'       : versoes(),
        'sementes'      : {'SEED': SEED, 'sorteio_amostra': SEED, 'subconjunto_xai': SEED + 1,
                           'perturbacao': '[SEED, serie_idx, origin, CENARIO_ID]',
                           'modelos': SEED, 'lime': SEED, 'shap_svr': SEED},
        'protocolo'     : {'MIN_TRAIN': MIN_TRAIN, 'N_ORIGINS': N_ORIGINS, 'SEASON': SEASON,
                           'N_LAGS': N_LAGS, 'features': feature_names(),
                           'hiperparametros': {k: {kk: str(vv) for kk, vv in v.items()}
                                               for k, v in HIPERPARAMETROS.items()},
                           'outlier_k': OUTLIER_K, 'lime_num_samples': LIME_NUM_SAMPLES},
        'dados'         : info_arquivos(arquivos_dados),
        **(extra or {}),
    }
    meta_path = path.with_name(path.name + '.meta.json')
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    return meta_path


def cache_csv(path, gerar, extra_meta=None, force=False, arquivos_dados=None):
    """Lê o CSV se existir; senão chama gerar(), salva CSV e .meta.json."""
    path = Path(path)
    if path.exists() and not force:
        return pd.read_csv(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = gerar()
    df.to_csv(path, index=False)
    write_meta(path, {**(extra_meta or {}), 'linhas': len(df), 'duracao_s': round(time.time() - t0, 1)},
               arquivos_dados)
    return df
