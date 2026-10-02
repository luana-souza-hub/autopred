import json
from pathlib import Path
from nbtools import md, code, check

P = Path(r'C:\Users\Usuario\Desktop\Mestrado\Mestrado - aula\dissertação\autopred\notebooks\09_explicabilidade.ipynb')
nb = json.loads(P.read_text(encoding='utf-8'))
cells = nb['cells']


def src(i):
    return ''.join(cells[i]['source'])


def set_src(i, s):
    lines = s.split('\n')
    cells[i]['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]


def replace_in(i, old, new):
    s = src(i)
    assert s.count(old) == 1, (i, old[:60])
    set_src(i, s.replace(old, new))


# ── 1. Correção do compute_lime (célula 12) ─────────────────────────────────
replace_in(12,
'''        # Monta vetor de coeficientes na ordem das features
        coef_dict = dict(exp.as_list())
        coefs = np.zeros(len(feature_names))
        for fname, val in coef_dict.items():
            # LIME inclui condições no nome (e.g. 'lag_1 > 0.5'), extrai nome base
            for j, fn in enumerate(feature_names):
                if fn in fname:
                    coefs[j] += abs(val)
                    break
        return coefs''',
'''        # Monta vetor de coeficientes na ordem das features.
        # CORREÇÃO (out/2026): associação por ÍNDICE via exp.as_map()[1]
        # (chave 1 = coeficientes do modo regressão, idênticos a exp.as_list()).
        # A versão anterior associava por substring do nome descritivo
        # ('lag_1' casava com 'lag_10 <= ...' e 'lag_season' com 'lag_season2 > ...'),
        # somando lag_10..lag_12 em lag_1 e lag_season2 em lag_season.
        # Resultado anterior preservado em results/explicabilidade_original_bugado/.
        coefs = np.zeros(len(feature_names))
        for j, val in exp.as_map()[1]:
            coefs[j] = abs(val)
        return coefs''')

s11 = src(11)
set_src(11, s11 + '''

> **Correção (out/2026):** a associação entre termos do LIME e features passou a ser feita por índice (`exp.as_map()[1]`), e não mais por substring do nome. Semente, `num_samples=200` e demais configurações foram mantidos. O resultado anterior está em `results/explicabilidade_original_bugado/`.''')

# ── 2. Posições de lag_season e lag_season2 na tabela-resumo (célula 25) ────
replace_in(25,
'''    # Estabilidade
    r_med = df_stab[df_stab['modelo'] == model_name]['spearman_r'].median()''',
'''    # Estabilidade
    r_med = df_stab[df_stab['modelo'] == model_name]['spearman_r'].median()

    # Posição no ranking global (1 = mais importante, entre as 17 features)
    rank_shap = shap_m.rank(ascending=False, method='min').astype(int)
    rank_lime = lime_m.rank(ascending=False, method='min').astype(int)''')
replace_in(25,
'''        'Estabilidade Spearman (mediana)': round(r_med, 3),
    })''',
'''        'Estabilidade Spearman (mediana)': round(r_med, 3),
        'Posição lag_season SHAP'  : rank_shap['lag_season'],
        'Posição lag_season LIME'  : rank_lime['lag_season'],
        'Posição lag_season2 SHAP' : rank_shap['lag_season2'],
        'Posição lag_season2 LIME' : rank_lime['lag_season2'],
    })''')
replace_in(25,
'''print('Tabela-resumo de explicabilidade:')
print(df_expl_resumo.to_string())''',
'''print('Tabela-resumo de explicabilidade:')
print(df_expl_resumo.T.to_string())''')

# ── 3. Registro de execução (célula 29) ─────────────────────────────────────
replace_in(29,
'''    'perturbacao_estabilidade': 'ruido_sigma_30pct',''',
'''    'perturbacao_estabilidade': 'ruido_sigma_30pct',
    'lime_associacao_features': 'por indice via exp.as_map()[1] (corrigido out/2026)',
    'lime_backup_versao_anterior': 'results/explicabilidade_original_bugado/',''')

# ── 4. Seção de sensibilidade (nova, inserida antes do registro) ────────────
sens = [
md(r'''
## 12. Análise de sensibilidade do LIME e da escala do SVR (separada do resultado oficial)

Esta seção **não altera** os arquivos oficiais acima. Resultados em `results/explicabilidade/sensibilidade/`.

1. **Número de amostras do LIME:** o oficial usa `num_samples=200`; aqui repetimos com 1000 e 5000 (mesma semente, mesmo ponto explicado, associação por índice).
2. **Escala do SVR:** no oficial, o SHAP do SVR explica `svr.predict` na escala **normalizada** do alvo (`scaler_y`), enquanto o LIME explica a previsão na escala **original**. Dentro de uma série a diferença é só um fator constante (`scaler_y.scale_`), mas na média entre séries ela muda o peso de cada série: no SHAP todas pesam igual, no LIME pesam as de maior nível. Comparamos:
   - SHAP original x LIME original: SHAP convertido por série multiplicando por `scaler_y.scale_` (exato, pois a transformação inversa é afim; não exige recalcular o KernelExplainer);
   - SHAP normalizado x LIME normalizado: LIME recalculado explicando `svr.predict` sem a inversão.

Para cada configuração: Pearson SHAP x LIME (mesma normalização pelo máximo da Fig. 4), top-10 do LIME e posição de lag_season e lag_season2.
'''),
code(r'''
SENS_DIR = EXPL_DIR / 'sensibilidade'
SENS_DIR.mkdir(parents=True, exist_ok=True)
MODELOS_XAI = ['RandomForest', 'GradientBoosting', 'SVR']


def compute_lime_sens(fitted_dict, feature_names, num_samples, escala='original'):
    """Igual a compute_lime (corrigido), com num_samples e escala do alvo configuráveis."""
    m, X = fitted_dict['model'], fitted_dict['X']
    sX, sy = fitted_dict['scaler_X'], fitted_dict['scaler_y']
    X_input = sX.transform(X) if sX is not None else X

    def predict_fn(x):
        if sy is not None and escala == 'original':
            return sy.inverse_transform(m.predict(x).reshape(-1, 1)).ravel()
        return m.predict(x)

    try:
        explainer = LimeTabularExplainer(training_data=X_input, feature_names=feature_names,
                                         mode='regression', random_state=SEED, verbose=False)
        exp = explainer.explain_instance(X_input[-1], predict_fn,
                                         num_features=len(feature_names), num_samples=num_samples)
        coefs = np.zeros(len(feature_names))
        for j, val in exp.as_map()[1]:
            coefs[j] = abs(val)
        return coefs
    except Exception:
        return np.full(len(feature_names), np.nan)


def lime_sens_cached(nome, modelos, num_samples, escala='original'):
    arq = SENS_DIR / f'lime_{nome}.csv'
    if arq.exists():
        print(f'  cache: {arq.name}')
        return pd.read_csv(arq)
    t0, rows = time.time(), []
    for i, fd in enumerate(fitted_all):
        if fd is None:
            continue
        for mn in modelos:
            imp = compute_lime_sens(fd[mn], FEATURE_NAMES, num_samples, escala)
            rows.append({'serie': i, 'modelo': mn, **dict(zip(FEATURE_NAMES, imp))})
    df = pd.DataFrame(rows)
    df.to_csv(arq, index=False)
    print(f'  {arq.name} calculado em {time.time() - t0:.0f}s')
    return df


print('LIME num_samples=1000:')
df_lime_1000 = lime_sens_cached('ns1000', MODELOS_XAI, 1000)
print('LIME num_samples=5000:')
df_lime_5000 = lime_sens_cached('ns5000', MODELOS_XAI, 5000)
print('LIME SVR escala normalizada (num_samples=200):')
df_lime_svr_norm = lime_sens_cached('svr_normalizada_ns200', ['SVR'], 200, escala='normalizada')

# SHAP do SVR na escala original: SHAP normalizado x scaler_y.scale_ da série
_escala_y = {i: fd['SVR']['scaler_y'].scale_[0] for i, fd in enumerate(fitted_all) if fd is not None}
df_shap_svr_orig = df_shap[df_shap['modelo'] == 'SVR'].copy()
df_shap_svr_orig[FEATURE_NAMES] = df_shap_svr_orig[FEATURE_NAMES].mul(
    df_shap_svr_orig['serie'].map(_escala_y), axis=0)
df_shap_svr_orig.to_csv(SENS_DIR / 'shap_svr_escala_original.csv', index=False)
print('shap_svr_escala_original.csv salvo')
'''),
code(r'''
def comparar(df_s, df_l, modelo, config):
    """Pearson (normalização pelo máximo, como na Fig. 4), top-10 e posições sazonais."""
    s = df_s[df_s['modelo'] == modelo][FEATURE_NAMES].mean()
    l = df_l[df_l['modelo'] == modelo][FEATURE_NAMES].mean()
    r = np.corrcoef((s / (s.max() + 1e-10)).values, (l / (l.max() + 1e-10)).values)[0, 1]
    rs = s.rank(ascending=False, method='min').astype(int)
    rl = l.rank(ascending=False, method='min').astype(int)
    return {
        'config': config, 'modelo': modelo, 'pearson_SHAP_LIME': round(r, 3),
        'pos_lag_season_SHAP': rs['lag_season'], 'pos_lag_season_LIME': rl['lag_season'],
        'pos_lag_season2_SHAP': rs['lag_season2'], 'pos_lag_season2_LIME': rl['lag_season2'],
        'top10_LIME': ', '.join(l.sort_values(ascending=False).index[:10]),
        'top10_SHAP': ', '.join(s.sort_values(ascending=False).index[:10]),
    }

linhas = []
for mn in MODELOS_XAI:
    linhas.append(comparar(df_shap, df_lime,      mn, 'oficial (ns=200)'))
    linhas.append(comparar(df_shap, df_lime_1000, mn, 'ns=1000'))
    linhas.append(comparar(df_shap, df_lime_5000, mn, 'ns=5000'))
linhas.append(comparar(df_shap_svr_orig, df_lime, 'SVR', 'SVR escala original (SHAP x scale_y), ns=200'))
linhas.append(comparar(df_shap, df_lime_svr_norm, 'SVR', 'SVR escala normalizada (LIME sem inversão), ns=200'))

df_sens = pd.DataFrame(linhas)
df_sens.to_csv(SENS_DIR / 'sensibilidade_resumo.csv', index=False)

pd.set_option('display.max_colwidth', None)
print(df_sens.drop(columns=['top10_LIME', 'top10_SHAP']).to_string(index=False))
print()
for _, r in df_sens.iterrows():
    print(f"{r['modelo']:<17} | {r['config']}\n   LIME top-10: {r['top10_LIME']}\n   SHAP top-10: {r['top10_SHAP']}")
print(f'\nSalvo em {SENS_DIR / "sensibilidade_resumo.csv"}')
'''),
]
check(sens)
assert src(28).startswith('---') and 'Registro' in src(28)
nb['cells'] = cells[:28] + sens + cells[28:]
P.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('NB09 editado. Total de células:', len(nb['cells']))
