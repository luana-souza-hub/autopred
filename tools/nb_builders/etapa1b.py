from nbtools import md, code, append

cells = []

cells.append(md(r'''
---
# Etapa 1b: LIME corrigido no M4 (limpo e com ruído σ=30%)

Verificação pedida antes das Etapas 2 a 6. O LIME limpo corrigido do M4 foi regenerado no próprio NB09 (`results/explicabilidade/lime_importancias.csv`; versão anterior em `results/explicabilidade_original_bugado/`). Aqui calculamos o LIME **com ruído σ=30%** nas mesmas 30 séries, reproduzindo exatamente a perturbação do NB09 (seção 9: `default_rng(SEED)` avançando série a série), para ter SHAP e LIME nas duas condições.

As funções de explicabilidade abaixo são cópias fiéis do NB09, com duas mudanças explícitas:
- `h` e `min_train` viram parâmetros de `train_models_for_series` (no NB09 eram as globais `H=18` e `MIN_TRAIN=24`; aqui o `H` global é 24, do Tourism);
- `compute_lime` usa a associação corrigida por índice (`exp.as_map()[1]`); `compute_lime_original` mantém a associação por substring, **apenas para comparação**.

Arquivos desta etapa usam o prefixo `xai_m4_` em `results/tourism/`.
'''))

cells.append(md(r'''
## 1b.1 Funções de explicabilidade (cópia do NB09)
'''))

cells.append(code(r'''
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr
import shap
from lime.lime_tabular import LimeTabularExplainer

N_LAGS     = 12
N_EXPL     = 30
MODELOS_XAI = ['RandomForest', 'GradientBoosting', 'SVR']
ALGO_COLORS_XAI = {'RandomForest': '#F44336', 'GradientBoosting': '#9C27B0', 'SVR': '#795548'}


def build_features_xai(series, n_lags=N_LAGS, season=SEASON):
    """Cópia de build_features do NB09 (features do protocolo NB08 e NB09)."""
    df = pd.DataFrame({'y': series})
    feat_cols = []
    for lag in range(1, n_lags + 1):
        col = f'lag_{lag}'
        df[col] = df['y'].shift(lag)
        feat_cols.append(col)
    for w, name in [(3, 'roll_mean_3'), (3, 'roll_std_3'), (6, 'roll_mean_6')]:
        if 'std' in name:
            df[name] = df['y'].shift(1).rolling(w).std()
        else:
            df[name] = df['y'].shift(1).rolling(w).mean()
        feat_cols.append(name)
    df['lag_season']  = df['y'].shift(season)
    df['lag_season2'] = df['y'].shift(season * 2)
    feat_cols += ['lag_season', 'lag_season2']
    df.dropna(inplace=True)
    return df[feat_cols].values, df['y'].values, feat_cols


FEATURE_NAMES = [f'lag_{i}' for i in range(1, N_LAGS + 1)] + \
                ['roll_mean_3', 'roll_std_3', 'roll_mean_6', 'lag_season', 'lag_season2']
N_FEATURES = len(FEATURE_NAMES)


def train_models_for_series(series, h, min_train=MIN_TRAIN, usar_serie_inteira=False):
    """
    Cópia de train_models_for_series do NB09, com h e min_train explícitos.
    usar_serie_inteira=True: exceção documentada para o Car Parts (ver nota D).
    """
    if usar_serie_inteira:
        train = series
    else:
        train = series[:-h] if len(series) > min_train + h else series
    X, y, feat_cols = build_features_xai(train)
    if len(X) < 10:
        return None

    fitted = {}
    rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(X, y)
    fitted['RandomForest'] = {'model': rf, 'X': X, 'y': y, 'scaler_X': None, 'scaler_y': None}

    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=SEED)
    gb.fit(X, y)
    fitted['GradientBoosting'] = {'model': gb, 'X': X, 'y': y, 'scaler_X': None, 'scaler_y': None}

    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    Xs = scaler_X.fit_transform(X)
    ys = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr.fit(Xs, ys)
    fitted['SVR'] = {'model': svr, 'X': X, 'y': y, 'scaler_X': scaler_X, 'scaler_y': scaler_y}
    return fitted


def compute_shap(fitted_dict, model_name, max_background=50):
    """Cópia de compute_shap do NB09 (TreeExplainer para árvores, KernelExplainer para SVR)."""
    m, X, sX = fitted_dict['model'], fitted_dict['X'], fitted_dict['scaler_X']
    X_input = sX.transform(X) if sX is not None else X
    bg_size = min(max_background, len(X_input))
    rng_bg  = np.random.default_rng(SEED)
    bg_idx  = rng_bg.choice(len(X_input), size=bg_size, replace=False)
    background = X_input[bg_idx]
    try:
        if model_name in ('RandomForest', 'GradientBoosting'):
            shap_values = shap.TreeExplainer(m).shap_values(X_input)
        else:
            explainer   = shap.KernelExplainer(m.predict, background)
            sample_idx  = rng_bg.choice(len(X_input), size=min(30, len(X_input)), replace=False)
            shap_values = explainer.shap_values(X_input[sample_idx], nsamples=50)
        return np.abs(shap_values).mean(axis=0)
    except Exception:
        return np.full(N_FEATURES, np.nan)


def _lime_explain(fitted_dict, feature_names, num_samples=200):
    """Parte comum do compute_lime do NB09: explica o último ponto de treino."""
    m, X = fitted_dict['model'], fitted_dict['X']
    sX, sy = fitted_dict['scaler_X'], fitted_dict['scaler_y']
    X_input = sX.transform(X) if sX is not None else X

    def predict_fn(x):
        if sy is not None:
            return sy.inverse_transform(m.predict(x).reshape(-1, 1)).ravel()
        return m.predict(x)

    explainer = LimeTabularExplainer(training_data=X_input, feature_names=feature_names,
                                     mode='regression', random_state=SEED, verbose=False)
    return explainer.explain_instance(X_input[-1], predict_fn,
                                      num_features=len(feature_names), num_samples=num_samples)


def compute_lime(fitted_dict, model_name, feature_names):
    """LIME CORRIGIDO (oficial): associação por índice via exp.as_map()[1]."""
    try:
        exp = _lime_explain(fitted_dict, feature_names)
        coefs = np.zeros(len(feature_names))
        for j, val in exp.as_map()[1]:
            coefs[j] = abs(val)
        return coefs
    except Exception:
        return np.full(len(feature_names), np.nan)


def compute_lime_original(fitted_dict, model_name, feature_names):
    """LIME ORIGINAL do NB09 (associação por substring), mantido só para comparação."""
    try:
        exp = _lime_explain(fitted_dict, feature_names)
        coefs = np.zeros(len(feature_names))
        for fname, val in dict(exp.as_list()).items():
            for j, fn in enumerate(feature_names):
                if fn in fname:
                    coefs[j] += abs(val)
                    break
        return coefs
    except Exception:
        return np.full(len(feature_names), np.nan)


def perturb_noise_xai(series, sigma_frac, rng):
    """Cópia de perturb_noise do NB09 (idêntica à do NB08)."""
    return series + rng.normal(0, np.std(series) * sigma_frac, size=len(series))


def importancias_por_serie(fitted_list, fn, nome_arquivo):
    """Aplica fn (compute_shap/compute_lime/...) a cada série e modelo, com cache CSV."""
    arq = RESULT_DIR / nome_arquivo
    if arq.exists() and not FORCE_RERUN:
        print(f'  cache: {arq.name}')
        return pd.read_csv(arq)
    t0, rows = time.time(), []
    for i, fd in enumerate(fitted_list):
        if fd is None:
            continue
        for mn in MODELOS_XAI:
            imp = fn(fd[mn], mn) if fn is compute_shap else fn(fd[mn], mn, FEATURE_NAMES)
            rows.append({'serie': i, 'modelo': mn, **dict(zip(FEATURE_NAMES, imp))})
    df = pd.DataFrame(rows)
    df.to_csv(arq, index=False)
    print(f'  {arq.name} calculado em {time.time() - t0:.0f}s')
    return df


def resumo_xai(df_s, df_l, modelo):
    """Pearson SHAP x LIME (normalização pelo máximo, como na Fig. 4 do NB09), top-10 e posições."""
    s = df_s[df_s['modelo'] == modelo][FEATURE_NAMES].mean()
    l = df_l[df_l['modelo'] == modelo][FEATURE_NAMES].mean()
    r = np.corrcoef((s / (s.max() + 1e-10)).values, (l / (l.max() + 1e-10)).values)[0, 1]
    rs = s.rank(ascending=False, method='min').astype(int)
    rl = l.rank(ascending=False, method='min').astype(int)
    return {'modelo': modelo, 'pearson': round(r, 3),
            'pos_lag_season_SHAP': rs['lag_season'], 'pos_lag_season_LIME': rl['lag_season'],
            'pos_lag_season2_SHAP': rs['lag_season2'], 'pos_lag_season2_LIME': rl['lag_season2'],
            'top10_SHAP': ', '.join(s.sort_values(ascending=False).index[:10]),
            'top10_LIME': ', '.join(l.sort_values(ascending=False).index[:10])}


print(f'Funções XAI OK | {N_FEATURES} features: {FEATURE_NAMES}')
'''))

cells.append(md(r'''
## 1b.2 Séries do M4 e perturbação idênticas às do NB09
'''))

cells.append(code(r'''
H_M4 = 18   # horizonte do M4 (NB09)

_m4_train_raw = pd.read_csv(M4_TRAIN_PATH, header=0, index_col=0)
_idx_m4 = np.random.default_rng(SEED).choice(len(_m4_train_raw), size=200, replace=False)
m4_series_list = [_m4_train_raw.iloc[i].dropna().values.astype(float) for i in _idx_m4]
del _m4_train_raw

# Subconjunto de 30 séries: mesma regra do NB09 (semente SEED + 1 sobre as posições)
m4_expl_idx    = np.random.default_rng(SEED + 1).choice(200, size=N_EXPL, replace=False)
m4_expl_series = [m4_series_list[i] for i in m4_expl_idx]

# Perturbação: mesmo gerador sequencial da seção 9 do NB09
rng_pert = np.random.default_rng(SEED)
m4_expl_pert = [perturb_noise_xai(s, sigma_frac=0.30, rng=rng_pert) for s in m4_expl_series]

t0 = time.time()
m4_fit_clean = [train_models_for_series(s, h=H_M4) for s in m4_expl_series]
m4_fit_pert  = [train_models_for_series(s, h=H_M4) for s in m4_expl_pert]
print(f'Modelos treinados em {time.time() - t0:.0f}s | válidas: '
      f'{sum(f is not None for f in m4_fit_clean)} limpas, {sum(f is not None for f in m4_fit_pert)} com ruído')
'''))

cells.append(code(r'''
# Verificação de reprodutibilidade: o SHAP das árvores (determinístico) recalculado
# sobre as séries com ruído deve coincidir com shap_pert_* do cache do NB09.
_stab = pd.read_csv(M4_RESULTS_DIR / 'explicabilidade' / 'shap_estabilidade_adversa.csv')
_dev = []
for i in range(3):
    for mn in ['RandomForest', 'GradientBoosting']:
        rec = compute_shap(m4_fit_pert[i][mn], mn)
        cac = _stab[(_stab['serie'] == i) & (_stab['modelo'] == mn)][[f'shap_pert_{f}' for f in FEATURE_NAMES]].values[0]
        _dev.append(np.max(np.abs(rec - cac) / (np.abs(cac) + 1e-9)))
print(f'Desvio relativo máximo SHAP recalculado x cache NB09 (3 séries, árvores): {max(_dev):.2e}')
print('Reprodução exata' if max(_dev) < 1e-6 else 'ATENÇÃO: perturbação ou modelos não coincidem com o NB09')
'''))

cells.append(md(r'''
## 1b.3 LIME com ruído σ=30% (corrigido e original)

SHAP limpo e com ruído vêm do cache do NB09 (`shap_importancias.csv` e `shap_estabilidade_adversa.csv`); o SHAP do SVR usa KernelExplainer, cuja amostragem interna não é semeada no NB09, então não o recalculamos. LIME limpo corrigido: NB09 regenerado. LIME com ruído: calculado aqui.
'''))

cells.append(code(r'''
print('LIME com ruído σ=30%, corrigido:')
xai_m4_lime_pert = importancias_por_serie(m4_fit_pert, compute_lime, 'xai_m4_lime_corrigido_ruido30.csv')
print('LIME com ruído σ=30%, original (comparação):')
xai_m4_lime_pert_orig = importancias_por_serie(m4_fit_pert, compute_lime_original, 'xai_m4_lime_original_ruido30.csv')

# Demais insumos (cache do NB09)
_expl = M4_RESULTS_DIR / 'explicabilidade'
xai_m4_shap_clean = pd.read_csv(_expl / 'shap_importancias.csv')
xai_m4_lime_clean = pd.read_csv(_expl / 'lime_importancias.csv')                       # corrigido
xai_m4_lime_clean_orig = pd.read_csv(M4_RESULTS_DIR / 'explicabilidade_original_bugado' / 'lime_importancias.csv')
xai_m4_shap_pert = _stab[['serie', 'modelo']].copy()
for f in FEATURE_NAMES:
    xai_m4_shap_pert[f] = _stab[f'shap_pert_{f}']
xai_m4_shap_pert.to_csv(RESULT_DIR / 'xai_m4_shap_ruido30.csv', index=False)

# Garantia: o LIME limpo carregado é o corrigido (lag_10 deixa de ser sempre zero)
assert xai_m4_lime_clean['lag_10'].abs().sum() > 0, 'lime_importancias.csv ainda é a versão antiga: rode o NB09 corrigido'
print('Insumos carregados.')
'''))

cells.append(md(r'''
## 1b.4 Resultado: Pearson SHAP x LIME, top-10 e posição dos lags sazonais
'''))

cells.append(code(r'''
_linhas = []
for cond, df_s, df_l, versao in [
    ('limpo',      xai_m4_shap_clean, xai_m4_lime_clean,      'corrigido'),
    ('limpo',      xai_m4_shap_clean, xai_m4_lime_clean_orig, 'original'),
    ('ruído 30%',  xai_m4_shap_pert,  xai_m4_lime_pert,       'corrigido'),
    ('ruído 30%',  xai_m4_shap_pert,  xai_m4_lime_pert_orig,  'original'),
]:
    for mn in MODELOS_XAI:
        _linhas.append({'base': 'M4', 'condicao': cond, 'lime': versao, **resumo_xai(df_s, df_l, mn)})

xai_m4_resumo = pd.DataFrame(_linhas)
xai_m4_resumo.to_csv(RESULT_DIR / 'xai_m4_resumo_etapa1b.csv', index=False)

_cols = ['condicao', 'lime', 'modelo', 'pearson', 'pos_lag_season_SHAP', 'pos_lag_season_LIME',
         'pos_lag_season2_SHAP', 'pos_lag_season2_LIME']
print('Pearson SHAP x LIME e posições (1 = mais importante, entre 17 features):\n')
print(xai_m4_resumo[_cols].to_string(index=False))
print('\nSalvo: xai_m4_resumo_etapa1b.csv')
'''))

cells.append(code(r'''
# Top-10 (versão oficial: LIME corrigido)
for cond in ['limpo', 'ruído 30%']:
    print(f'=== M4, {cond} ===')
    sub = xai_m4_resumo[(xai_m4_resumo['condicao'] == cond) & (xai_m4_resumo['lime'] == 'corrigido')]
    for _, r in sub.iterrows():
        print(f"{r['modelo']:<17} SHAP: {r['top10_SHAP']}")
        print(f"{'':<17} LIME: {r['top10_LIME']}")
    print()
'''))

cells.append(code(r'''
# ── Figura: SHAP x LIME corrigido (normalizados), limpo e com ruído ───────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
for lin, (cond, df_s, df_l) in enumerate([('limpo', xai_m4_shap_clean, xai_m4_lime_clean),
                                           ('ruído σ=30%', xai_m4_shap_pert, xai_m4_lime_pert)]):
    for ax, mn in zip(axes[lin], MODELOS_XAI):
        s = df_s[df_s['modelo'] == mn][FEATURE_NAMES].mean(); s = s / (s.max() + 1e-10)
        l = df_l[df_l['modelo'] == mn][FEATURE_NAMES].mean(); l = l / (l.max() + 1e-10)
        r = np.corrcoef(s.values, l.values)[0, 1]
        ax.scatter(s.values, l.values, color=ALGO_COLORS_XAI[mn], alpha=0.7, s=55)
        for fn in set(s.sort_values(ascending=False).index[:4]) | {'lag_season', 'lag_season2'}:
            ax.annotate(fn, (s[fn], l[fn]), fontsize=7, xytext=(4, 2), textcoords='offset points',
                        fontweight='bold' if 'season' in fn else 'normal')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'{mn} ({cond})\nr = {r:.3f}', fontsize=10)
        ax.set_xlabel('SHAP (normalizado)', fontsize=9); ax.set_ylabel('LIME corrigido (normalizado)', fontsize=9)
fig.suptitle('M4 Monthly: concordância SHAP x LIME corrigido, dados limpos e com ruído σ=30%\n'
             '30 séries do NB09; lags sazonais em negrito', fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(RESULT_DIR / 'fig_xai_m4_shap_vs_lime_corrigido.png', dpi=150, bbox_inches='tight')
plt.show()
print('Figura salva: fig_xai_m4_shap_vs_lime_corrigido.png')
'''))

append(cells)
print('Etapa 1b adicionada:', len(cells), 'células')
