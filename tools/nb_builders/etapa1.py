from nbtools import md, code, write_new

cells = []

cells.append(md(r'''
# 11: Replicação em uma terceira base (Tourism Monthly)

**Dissertação:** Análise Comparativa de Algoritmos de Regressão para Previsão em Séries Temporais: Desempenho, Robustez e Explicabilidade em Cenários de Dados Adversos
**Autora:** Luana Gomes de Souza, PPGI/UTFPR
**Etapa:** "prova dos nove": o mesmo protocolo dos notebooks 01 a 10 aplicado a uma base adicional, para verificar quais achados se sustentam em outro regime de dados.

### Base
Tourism Monthly Dataset, Monash Time Series Forecasting Repository (Zenodo, versão 3, DOI 10.5281/zenodo.4656096). São 366 séries mensais reais da competição de turismo (Athanasopoulos et al., 2011).

### Estrutura do notebook
| Etapa | Conteúdo | Protocolo de origem |
|---|---|---|
| 1 | Configuração e caracterização da base | NB10 (loader) |
| 1b | LIME corrigido no M4 (verificação antes de seguir) | NB09 |
| 2 | Desempenho em dados limpos | NB01 a 07 e NB10 |
| 3 | Cenários adversos | NB08 |
| 4 | Explicabilidade (SHAP e LIME) | NB09 |
| 5 | Estabilidade das explicações (Spearman e Jaccard top-k) | NB09 |
| 6 | Tabela de replicação (H1 a H4) | |

Resultados em `results/tourism/`. Cada etapa cara tem cache em CSV: com `FORCE_RERUN = False`, nada é recalculado se o arquivo já existe.
'''))

cells.append(md(r'''
## Nota de limitações (diferenças de protocolo entre os notebooks existentes)

Esta nota registra diferenças encontradas nos notebooks 01 a 10 ao planejar esta replicação. **Os notebooks anteriores não foram alterados**; cada etapa deste notebook replica fielmente o protocolo do notebook de origem, para que os números sejam comparáveis com os já reportados.

**(B) Duas amostras piloto do M4.** Os notebooks 01 a 04b sorteiam as 200 séries com `rng.choice` sobre os IDs em ordem alfabética (`train_df['unique_id'].unique()` após `sort_values`), enquanto os notebooks 05 a 09 sorteiam sobre a ordem do arquivo (`src/m4_io.sample_pilot_ids` e `idx_sel` nos NB08 e NB09). As duas amostras têm apenas 3 séries em comum. Consequência: as tabelas do M4 que comparam modelos estatísticos (NB02 a 04b) com modelos de ML (NB05 a 07) usam conjuntos de séries diferentes. Para a hipótese H4 no M4, usamos `baseline_limpo_200series.csv` (NB08), em que todos os modelos foram avaliados nas mesmas 200 séries. Neste notebook, o Tourism usa uma única amostra para todas as etapas (regra dos NB05 a 09, ordem do arquivo).

**(C) Dois protocolos de avaliação.**

| Item | NB01 a 07 e NB10 (desempenho) | NB08 e NB09 (adversos e XAI) |
|---|---|---|
| Origens do rolling origin | `linspace(MIN_TRAIN, n - H, N_ORIGINS)` | os N_ORIGINS últimos blocos consecutivos de H |
| MAPE | máscara sobre valores reais não nulos | fallback para sMAPE quando mais de 50% dos valores reais são ~0 |
| Features de ML | lag_1 a lag_12, roll_mean e roll_std (janela 12), lag_season, lag_season2 | lag_1 a lag_12, roll_mean_3, roll_std_3, roll_mean_6, lag_season, lag_season2 |
| Random Forest | n_estimators=100, min_samples_leaf=2 | n_estimators=100, demais padrões do sklearn |
| Gradient Boosting | subsample=0,8 | subsample=1,0 (padrão) |
| Holt-Winters | aditivo ou multiplicativo por AIC | apenas aditivo, `remove_bias=True` |
| SARIMA automático | busca por AIC com limite de 300 s por série (NB04b) | grade completa sem limite (NB08, só na linha de base) |
| Agregação | mediana por série, depois mediana entre séries | mediana por série, depois mediana entre séries |

Decisão: a Etapa 2 usa o protocolo NB01 a 07 e NB10; as Etapas 3 a 5 usam o protocolo NB08 e NB09. Os números de desempenho das Etapas 2 e 3 não devem ser comparados diretamente entre si.

**(A) Correspondência de nomes no LIME do NB09.** A função `compute_lime` do NB09 associa cada termo do LIME a uma feature por substring (`if fn in fname`). Como `'lag_1'` está contido em `'lag_10 <= ...'`, a importância de lag_10, lag_11 e lag_12 é somada a lag_1, e a de lag_season2 é somada a lag_season (no cache `lime_importancias.csv` essas quatro colunas são sempre zero). Neste notebook o LIME é calculado em duas versões: **corrigida (oficial)**, com associação por índice via `exp.as_map()`, e **original**, mantida apenas para comparação.

**(D) Car Parts na explicabilidade.** Com 51 observações e o protocolo do NB09 (treino = série sem os últimos H), restam 9 linhas supervisionadas, abaixo do mínimo de 10 do NB09. Para o Car Parts, SHAP e LIME são calculados com a série inteira (27 linhas de treino) e os resultados são rotulados como **exploratórios**.
'''))

cells.append(md(r'''
---
# Etapa 1: configuração e caracterização da base
## 1.1 Imports e configuração
'''))

cells.append(code(r'''
import warnings, time, json
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from statsmodels.tsa.seasonal import STL

# ── Resolução de caminhos (mesma lógica do NB10: cwd em autopred/ ou notebooks/) ──
_root = Path.cwd().resolve()
for _candidate in (_root, *_root.parents):
    if (_candidate / 'notebooks').is_dir() and (_candidate / 'results').is_dir():
        PROJECT_ROOT = _candidate
        break
else:
    PROJECT_ROOT = _root

DATA_DIR = PROJECT_ROOT / 'data' / 'raw'

# Tourism: aceita o .tsf em data/raw/ ou em data/raw/tourism/
_tm_candidates = [
    DATA_DIR / 'tourism_monthly_dataset.tsf',
    DATA_DIR / 'tourism' / 'tourism_monthly_dataset.tsf',
]
TM_TSF_PATH = next((p for p in _tm_candidates if p.exists()), None)
if TM_TSF_PATH is None:
    raise FileNotFoundError('tourism_monthly_dataset.tsf não encontrado em:\n'
                            + '\n'.join(f'  {p}' for p in _tm_candidates))

# Car Parts (mesmos candidatos do NB10)
_cp_candidates = [
    DATA_DIR / 'car_parts' / 'car_parts_dataset_without_missing_values.tsf',
    DATA_DIR / 'car_parts_dataset_without_missing_values.tsf',
]
CP_TSF_PATH = next((p for p in _cp_candidates if p.exists()), None)
M4_TRAIN_PATH = DATA_DIR / 'Monthly-train.csv'

# ── Protocolo (idêntico aos notebooks 01 a 10, exceto H) ─────────────────────
SEED      = 42
# H = 24: horizonte oficial do Tourism Monthly no benchmark (Athanasopoulos et al., 2011;
# cabeçalho do .tsf: "@horizon 24"). No M4 Monthly e no Car Parts usamos H = 18.
# É a ÚNICA diferença de protocolo em relação às bases anteriores.
H         = 24
MIN_TRAIN = 24
N_ORIGINS = 3
SEASON    = 12

# ── Amostragem de séries (mesma lógica do piloto M4: 200 séries, SEED=42) ────
# Viabilidade das 366 séries completas: o Tourism tem séries de 91 a 333 obs
# (mediana 330). Pelos tempos medianos do M4 (RF ~0,34 s por janela, SARIMA
# heurístico ~0,1 s), rodar as 366 custa ~1,8x o piloto: estimativa de 15 a 30 min
# na Etapa 2 sem SARIMA auto, 2 a 4 h com SARIMA auto e ~1 h na Etapa 3.
# É viável; fica como análise de sensibilidade (RODAR_TODAS = True).
N_SERIES     = 200
RODAR_TODAS  = False          # True: usa as 366 séries (sensibilidade)
if RODAR_TODAS:
    N_SERIES = None

# ── Saídas: a sensibilidade com 366 séries vai para uma subpasta própria,
#    para não sobrescrever o cache do piloto de 200 séries ─────────────────────
RESULT_DIR = PROJECT_ROOT / 'results' / 'tourism'
if N_SERIES is None:
    RESULT_DIR = RESULT_DIR / 'sensibilidade_366'
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Caches de bases anteriores (somente leitura)
M4_RESULTS_DIR = PROJECT_ROOT / 'results'
CP_RESULTS_DIR = PROJECT_ROOT / 'results' / 'car_parts'

# ── Cache: True força recálculo mesmo com CSV salvo ──────────────────────────
FORCE_RERUN = False

# Paleta por base (tons já usados no NB10)
BASE_COLORS = {'M4 Monthly': '#5B8DB8', 'Car Parts': '#E8834E', 'Tourism Monthly': '#6BAE75'}

print(f'PROJECT_ROOT = {PROJECT_ROOT}')
print(f'TM_TSF_PATH  = {TM_TSF_PATH}')
print(f'CP_TSF_PATH  = {CP_TSF_PATH}')
print(f'RESULT_DIR   = {RESULT_DIR}')
print(f'SEED={SEED} | H={H} | MIN_TRAIN={MIN_TRAIN} | N_ORIGINS={N_ORIGINS} | SEASON={SEASON}')
print(f'N_SERIES = {N_SERIES if N_SERIES else "TODAS"} | FORCE_RERUN = {FORCE_RERUN}')
'''))

cells.append(md(r'''
## 1.2 Carregador do formato TSF

Cópia fiel de `load_car_parts_tsf` (NB10), com uma única mudança: o parâmetro `encoding`. O cabeçalho do arquivo do Tourism contém o byte `0x96` (travessão em cp1252) num comentário, o que faz a leitura em UTF-8 falhar. As linhas de dados são ASCII, então ler em `latin-1` não altera nenhum valor.

Também adicionamos duas funções auxiliares: `read_tsf_header` (metadados `@...`) e `list_tsf_ids` (IDs na ordem do arquivo, equivalente a `src/m4_io.list_series_ids`, usada na amostragem).
'''))

cells.append(code(r'''
def load_tsf(tsf_path, encoding='utf-8'):
    """
    Cópia de load_car_parts_tsf (NB10), com encoding parametrizável.

    Formato TSF: comentários '#', metadados '@', início dos dados em '@data',
    cada linha de dado: nome:timestamp:val1,val2,...
    Retorna DataFrame [unique_id, step, y] ordenado por série e passo.
    """
    records  = []
    in_data  = False

    with open(tsf_path, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.lower() == '@data':
                in_data = True
                continue
            if line.startswith('@'):
                continue
            if not in_data:
                continue

            parts = line.split(':')
            if len(parts) < 3:
                continue

            series_name = parts[0].strip()
            values_str  = parts[-1].strip()   # último campo = valores CSV

            try:
                values = [float(v) for v in values_str.split(',') if v.strip()]
            except ValueError:
                continue

            for step, val in enumerate(values, start=1):
                records.append({
                    'unique_id': series_name,
                    'step'     : step,
                    'y'        : val,
                })

    df = pd.DataFrame(records, columns=['unique_id', 'step', 'y'])
    return df.sort_values(['unique_id', 'step']).reset_index(drop=True)


def read_tsf_header(tsf_path, encoding='latin-1'):
    """Lê os metadados '@chave valor' antes de '@data' (ex.: @horizon, @missing)."""
    meta = {}
    with open(tsf_path, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if line.lower() == '@data':
                break
            if line.startswith('@') and not line.startswith('@attribute'):
                key, _, val = line[1:].partition(' ')
                meta[key] = val.strip()
    return meta


def list_tsf_ids(tsf_path, encoding='latin-1'):
    """IDs das séries na ordem do arquivo (equivalente a m4_io.list_series_ids)."""
    ids, in_data = [], False
    with open(tsf_path, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if line.lower() == '@data':
                in_data = True
                continue
            if in_data and line and not line.startswith('#'):
                ids.append(line.split(':')[0].strip())
    return ids


print('Funções de leitura TSF OK')
'''))

cells.append(md(r'''
## 1.3 Carga do Tourism Monthly e metadados do cabeçalho
'''))

cells.append(code(r'''
tm_meta = read_tsf_header(TM_TSF_PATH)
print('Metadados do cabeçalho (.tsf):')
for k, v in tm_meta.items():
    print(f'  @{k:<12} {v}')

TM_MISSING     = tm_meta.get('missing')
TM_EQUALLENGTH = tm_meta.get('equallength')
TM_HORIZON     = int(tm_meta['horizon']) if 'horizon' in tm_meta else None
assert TM_HORIZON == H, f'H={H} difere do @horizon do arquivo ({TM_HORIZON})'

t0 = time.time()
df_tm_raw = load_tsf(TM_TSF_PATH, encoding='latin-1')
print(f'\nCarregado em {time.time() - t0:.2f}s')
print(f'Shape  : {df_tm_raw.shape}')
print(f'Séries : {df_tm_raw["unique_id"].nunique():,}')
print(f'NaN em y: {df_tm_raw["y"].isna().sum()}  (esperado 0, pois @missing = {TM_MISSING})')
print(f'Valores negativos: {(df_tm_raw["y"] < 0).sum()}')
print(df_tm_raw.head(5).to_string(index=False))
'''))

cells.append(md(r'''
## 1.4 Seleção da amostra (mesma regra dos NB05 a 09)

`np.random.default_rng(SEED).choice(n, size=N_SERIES, replace=False)` sobre os IDs **na ordem do arquivo**. A ordem do sorteio é preservada (`posicao`), porque o subconjunto de 30 séries da explicabilidade (Etapa 4) é sorteado sobre essas posições, como no NB09.

Filtro de comprimento: `MIN_TRAIN + H = 48`. Também verificamos quantas séries têm as 3 janelas completas no protocolo do NB08 (`MIN_TRAIN + N_ORIGINS * H = 96`).
'''))

cells.append(code(r'''
ids_arquivo = list_tsf_ids(TM_TSF_PATH)
assert len(ids_arquivo) == df_tm_raw['unique_id'].nunique()

if N_SERIES is None or N_SERIES >= len(ids_arquivo):
    idx_sel = np.arange(len(ids_arquivo))
else:
    rng_sel = np.random.default_rng(SEED)
    idx_sel = rng_sel.choice(len(ids_arquivo), size=N_SERIES, replace=False)

sample_ids = [ids_arquivo[i] for i in idx_sel]

# Séries em lista, na ordem do sorteio (como series_list nos NB08 e NB09)
_grp = {uid: g['y'].values.astype(float) for uid, g in df_tm_raw.groupby('unique_id')}
series_list = [_grp[uid] for uid in sample_ids]
lengths     = np.array([len(s) for s in series_list])

min_len    = MIN_TRAIN + H
min_len_08 = MIN_TRAIN + N_ORIGINS * H
print(f'Séries no arquivo  : {len(ids_arquivo)}')
print(f'Séries na amostra  : {len(sample_ids)}')
print(f'Comprimento >= MIN_TRAIN + H = {min_len}: {(lengths >= min_len).sum()} de {len(lengths)}')
print(f'3 janelas completas no protocolo NB08 (n >= {min_len_08}): {(lengths >= min_len_08).sum()} de {len(lengths)}')

df_amostra = pd.DataFrame({'posicao': range(len(sample_ids)), 'unique_id': sample_ids, 'n_obs': lengths})
df_amostra.to_csv(RESULT_DIR / 'amostra_series_tm.csv', index=False)
print(f'\nAmostra salva: amostra_series_tm.csv | primeiras: {sample_ids[:5]}')
'''))

cells.append(md(r'''
## 1.5 Caracterização: comprimento, zeros, força de tendência e de sazonalidade

Para cada série, decomposição STL (período 12) e as medidas de Wang, Smith e Hyndman (2006), na forma de Hyndman e Athanasopoulos (*Forecasting: Principles and Practice*, 3a ed., seção 4.3):

$$F_T = \max\left(0,\ 1 - \frac{\mathrm{Var}(R_t)}{\mathrm{Var}(T_t + R_t)}\right) \qquad F_S = \max\left(0,\ 1 - \frac{\mathrm{Var}(R_t)}{\mathrm{Var}(S_t + R_t)}\right)$$

Valores próximos de 1 indicam tendência (ou sazonalidade) forte. Séries constantes (variância nula, comuns no Car Parts) recebem `NaN`.

Conjuntos caracterizados:
- **M4 Monthly**: as 200 séries do piloto dos NB05 a 09 (ordem do arquivo), parte de treino (`Monthly-train.csv`), como no NB09;
- **Car Parts**: as 2.674 séries (todas usadas no NB10);
- **Tourism Monthly**: a amostra de 200 séries e, para referência, as 366.

Cache por série em `caracterizacao_series_{base}.csv`.
'''))

cells.append(code(r'''
def caracterizar_serie(y, season=SEASON):
    """Comprimento, proporção de zeros, CV e forças F_T e F_S via STL (Wang et al., 2006)."""
    y = np.asarray(y, dtype=float)
    out = {
        'n_obs'    : len(y),
        'zero_prop': float((y == 0).mean()),
        'cv'       : float(np.std(y) / np.mean(y)) if np.mean(y) != 0 else np.nan,
        'F_T'      : np.nan,
        'F_S'      : np.nan,
    }
    if len(y) < 2 * season or np.std(y) == 0:
        return out
    try:
        res = STL(y, period=season).fit()
        T, S, R = res.trend, res.seasonal, res.resid
        var_r = np.var(R)
        if np.var(T + R) > 0:
            out['F_T'] = float(max(0.0, 1 - var_r / np.var(T + R)))
        if np.var(S + R) > 0:
            out['F_S'] = float(max(0.0, 1 - var_r / np.var(S + R)))
    except Exception:
        pass
    return out


def caracterizar_base(series_dict, cache_file):
    """Aplica caracterizar_serie a {id: array}. Usa cache CSV se existir."""
    if cache_file.exists() and not FORCE_RERUN:
        df = pd.read_csv(cache_file)
        print(f'  cache: {cache_file.name} ({len(df)} séries)')
        return df
    t0 = time.time()
    rows = [{'unique_id': uid, **caracterizar_serie(y)} for uid, y in series_dict.items()]
    df = pd.DataFrame(rows)
    df.to_csv(cache_file, index=False)
    print(f'  calculado em {time.time() - t0:.1f}s, salvo: {cache_file.name} ({len(df)} séries)')
    return df


# Verificação com série sintética: sazonal pura + ruído pequeno => F_S alto, F_T baixo
_ex = 100 + 10 * np.sin(2 * np.pi * np.arange(120) / 12) + np.random.default_rng(SEED).normal(0, 1, 120)
print('Série sintética sazonal:', {k: round(v, 3) for k, v in caracterizar_serie(_ex).items()})
'''))

cells.append(code(r'''
print('Tourism Monthly (amostra):')
car_tm = caracterizar_base(dict(zip(sample_ids, series_list)),
                           RESULT_DIR / 'caracterizacao_series_tm.csv')

print('Tourism Monthly (366 séries):')
car_tm_all = caracterizar_base(_grp, PROJECT_ROOT / 'results' / 'tourism' / 'caracterizacao_series_tm_366.csv')

print('Car Parts (2674 séries):')
_cp_cache = PROJECT_ROOT / 'results' / 'tourism' / 'caracterizacao_series_cp.csv'
if _cp_cache.exists() and not FORCE_RERUN:
    car_cp = caracterizar_base({}, _cp_cache)
else:
    df_cp_raw = load_tsf(CP_TSF_PATH, encoding='utf-8')
    car_cp = caracterizar_base(
        {uid: g['y'].values.astype(float) for uid, g in df_cp_raw.groupby('unique_id')}, _cp_cache)

print('M4 Monthly (200 séries do piloto NB05 a 09):')
_m4_cache = PROJECT_ROOT / 'results' / 'tourism' / 'caracterizacao_series_m4.csv'
if _m4_cache.exists() and not FORCE_RERUN:
    car_m4 = caracterizar_base({}, _m4_cache)
else:
    # Mesma seleção dos NB08 e NB09: índices de linha sobre a ordem do arquivo
    _m4_train_raw = pd.read_csv(M4_TRAIN_PATH, header=0, index_col=0)
    _m4_idx = np.random.default_rng(SEED).choice(len(_m4_train_raw), size=200, replace=False)
    _m4_dict = {_m4_train_raw.index[i]: _m4_train_raw.iloc[i].dropna().values.astype(float)
                for i in _m4_idx}
    car_m4 = caracterizar_base(_m4_dict, _m4_cache)
    del _m4_train_raw
'''))

cells.append(md(r'''
## 1.6 Tabela comparativa entre as três bases

Medianas entre séries. O limiar de 0,5 em $F_S$ é apenas descritivo (sazonalidade moderada a forte).
'''))

cells.append(code(r'''
def resumo_base(df, nome, n_total, h, missing, equallength):
    return {
        'Base'                   : nome,
        'Séries (total)'         : n_total,
        'Séries analisadas'      : len(df),
        'H'                      : h,
        '@missing'               : missing,
        '@equallength'           : equallength,
        'Comprimento mín.'       : int(df['n_obs'].min()),
        'Comprimento mediano'    : float(df['n_obs'].median()),
        'Comprimento máx.'       : int(df['n_obs'].max()),
        'Zeros (mediana, %)'     : round(100 * df['zero_prop'].median(), 1),
        'Séries com zeros (%)'   : round(100 * (df['zero_prop'] > 0).mean(), 1),
        'CV mediano'             : round(df['cv'].median(), 3),
        'F_T mediana'            : round(df['F_T'].median(), 3),
        'F_S mediana'            : round(df['F_S'].median(), 3),
        'F_S > 0,5 (%)'          : round(100 * (df['F_S'] > 0.5).mean(), 1),
        'STL indefinido (%)'     : round(100 * df['F_S'].isna().mean(), 1),
    }

cp_meta = read_tsf_header(CP_TSF_PATH) if CP_TSF_PATH else {}
tabela_bases = pd.DataFrame([
    resumo_base(car_m4, 'M4 Monthly', 48000, 18, 'n/a (CSV)', 'false'),
    resumo_base(car_cp, 'Car Parts', 2674, 18, cp_meta.get('missing'), cp_meta.get('equallength')),
    resumo_base(car_tm, 'Tourism Monthly', len(ids_arquivo), H, TM_MISSING, TM_EQUALLENGTH),
    resumo_base(car_tm_all, 'Tourism Monthly (366)', len(ids_arquivo), H, TM_MISSING, TM_EQUALLENGTH),
]).set_index('Base')

tabela_bases.to_csv(RESULT_DIR / 'caracterizacao_bases.csv')
print(tabela_bases.T.to_string())
print('\nSalvo: caracterizacao_bases.csv')
'''))

cells.append(md(r'''
### 1.6.1 Piso de ruído do $F_S$ (controle)

O STL sempre extrai algum componente sazonal, mesmo de ruído branco, e esse efeito é maior em séries curtas: com poucos ciclos (o Car Parts tem 51 obs, ou seja, ~4 anos) o suavizador sazonal absorve parte do ruído. Por isso $F_S$ não é comparável entre bases de comprimentos muito diferentes sem uma referência. Para cada base, simulamos 300 séries de ruído branco com o comprimento mediano da base e calculamos $F_S$ (piso). Reportamos o piso mediano, seu percentil 95 e a proporção de séries da base com $F_S$ acima do p95 do piso, que é a medida de sazonalidade **distinguível de ruído**.
'''))

cells.append(code(r'''
N_SIM_PISO = 300
_piso_cache = RESULT_DIR / 'piso_ruido_fs.csv'

if _piso_cache.exists() and not FORCE_RERUN:
    df_piso = pd.read_csv(_piso_cache)
    print(f'cache: {_piso_cache.name}')
else:
    rng_piso = np.random.default_rng(SEED)
    _rows = []
    for nome, df_c in [('M4 Monthly', car_m4), ('Car Parts', car_cp),
                       ('Tourism Monthly', car_tm), ('Tourism Monthly (366)', car_tm_all)]:
        n_med = int(df_c['n_obs'].median())
        fs_sim = [caracterizar_serie(rng_piso.normal(0, 1, n_med))['F_S'] for _ in range(N_SIM_PISO)]
        _rows.append({'Base': nome, 'n_mediano': n_med,
                      'F_S piso (mediana)': float(np.median(fs_sim)),
                      'F_S piso (p95)': float(np.percentile(fs_sim, 95))})
    df_piso = pd.DataFrame(_rows)
    df_piso.to_csv(_piso_cache, index=False)

df_piso = df_piso.set_index('Base')
_car_por_base = {'M4 Monthly': car_m4, 'Car Parts': car_cp,
                 'Tourism Monthly': car_tm, 'Tourism Monthly (366)': car_tm_all}
tabela_bases['F_S piso ruído (mediana)'] = df_piso['F_S piso (mediana)'].round(3)
tabela_bases['F_S piso ruído (p95)']     = df_piso['F_S piso (p95)'].round(3)
tabela_bases['F_S acima do piso p95 (%)'] = pd.Series({
    b: round(100 * (_car_por_base[b]['F_S'] > df_piso.loc[b, 'F_S piso (p95)']).mean(), 1)
    for b in tabela_bases.index})

tabela_bases.to_csv(RESULT_DIR / 'caracterizacao_bases.csv')
print(tabela_bases[['Comprimento mediano', 'F_S mediana', 'F_S piso ruído (mediana)',
                    'F_S piso ruído (p95)', 'F_S acima do piso p95 (%)']].to_string())
print('\ncaracterizacao_bases.csv atualizado com o piso de ruído')
'''))

cells.append(code(r'''
# ── Figura: distribuição de F_T e F_S por base + dispersão F_T x F_S ──────────
_car = pd.concat([
    car_m4.assign(base='M4 Monthly'),
    car_cp.assign(base='Car Parts'),
    car_tm.assign(base='Tourism Monthly'),
], ignore_index=True)
_ordem = list(BASE_COLORS.keys())

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
for ax, col, titulo in zip(axes[:2], ['F_T', 'F_S'], ['Força de tendência $F_T$', 'Força de sazonalidade $F_S$']):
    sns.boxplot(data=_car, x='base', y=col, order=_ordem, palette=BASE_COLORS,
                ax=ax, fliersize=2, linewidth=0.8)
    for i, b in enumerate(_ordem):
        med = _car.loc[_car['base'] == b, col].median()
        ax.text(i, med, f'{med:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        if col == 'F_S':
            ax.hlines(df_piso.loc[b, 'F_S piso (p95)'], i - 0.4, i + 0.4, colors='black',
                      linestyles=':', linewidth=1.5, label='p95 do piso de ruído' if i == 0 else None)
    if col == 'F_S':
        ax.legend(fontsize=8, loc='lower right')
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel('')
    ax.set_ylim(-0.02, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

for b in _ordem:
    sub = _car[_car['base'] == b]
    axes[2].scatter(sub['F_T'], sub['F_S'], s=10, alpha=0.45, color=BASE_COLORS[b],
                    label=f'{b} (n={sub["F_S"].notna().sum():,})')
axes[2].set_xlabel('$F_T$'); axes[2].set_ylabel('$F_S$')
axes[2].set_xlim(-0.02, 1.02); axes[2].set_ylim(-0.02, 1.02)
axes[2].set_title('Regime de dados: tendência x sazonalidade', fontsize=10)
axes[2].legend(fontsize=8, loc='lower left')
axes[2].grid(linestyle='--', alpha=0.3)

plt.suptitle('Caracterização das três bases (STL, período 12; Wang et al., 2006)', fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(RESULT_DIR / 'fig_caracterizacao_bases.png', dpi=150, bbox_inches='tight')
plt.show()
print('Figura salva: fig_caracterizacao_bases.png')
'''))

cells.append(code(r'''
# ── Figura: EDA do Tourism (comprimentos, zeros, exemplos com F_S alto e baixo) ──
fig, axes = plt.subplots(1, 3, figsize=(17, 4))

axes[0].hist(car_tm['n_obs'], bins=20, color=BASE_COLORS['Tourism Monthly'], edgecolor='white')
axes[0].axvline(car_tm['n_obs'].median(), color='tomato', linestyle='--',
                label=f'mediana={car_tm["n_obs"].median():.0f}')
axes[0].axvline(MIN_TRAIN + N_ORIGINS * H, color='gray', linestyle=':',
                label=f'3 janelas NB08 ({MIN_TRAIN + N_ORIGINS * H})')
axes[0].set_title('Comprimento das séries (amostra)'); axes[0].set_xlabel('Observações'); axes[0].legend(fontsize=8)

axes[1].hist(car_tm['zero_prop'], bins=20, color='#E8834E', edgecolor='white')
axes[1].set_title('Proporção de zeros por série'); axes[1].set_xlabel('Proporção de zeros')

_hi = car_tm.sort_values('F_S', ascending=False).iloc[0]['unique_id']
_lo = car_tm.sort_values('F_S').iloc[0]['unique_id']
for uid, cor in [(_hi, '#2E8B57'), (_lo, '#C05050')]:
    fs = car_tm.set_index('unique_id').loc[uid, 'F_S']
    axes[2].plot(_grp[uid], color=cor, linewidth=1.2, label=f'{uid} ($F_S$={fs:.2f})')
axes[2].set_title('Séries com maior e menor $F_S$'); axes[2].set_xlabel('Passo (mês)'); axes[2].legend(fontsize=8)

plt.suptitle(f'Tourism Monthly: EDA | {len(sample_ids)} séries da amostra', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(RESULT_DIR / 'fig_eda_tourism.png', dpi=150, bbox_inches='tight')
plt.show()
print('Figura salva: fig_eda_tourism.png')
'''))

cells.append(md(r'''
## 1.7 Registro da Etapa 1
'''))

cells.append(code(r'''
log_etapa1 = {
    'timestamp'        : datetime.now().isoformat(),
    'notebook'         : '11_replicacao_tourism',
    'etapa'            : 1,
    'dataset'          : 'Tourism Monthly (Monash, Zenodo v3, DOI 10.5281/zenodo.4656096)',
    'tsf_meta'         : tm_meta,
    'n_series_arquivo' : len(ids_arquivo),
    'n_series_amostra' : len(sample_ids),
    'seed'             : SEED, 'H': H, 'min_train': MIN_TRAIN, 'n_origins': N_ORIGINS, 'season': SEASON,
    'nota_H'           : 'H=24 (benchmark Tourism); M4 e Car Parts usam H=18',
    'caracterizacao'   : tabela_bases.reset_index().to_dict(orient='records'),
}
with open(RESULT_DIR / 'run_log_tm_etapa1.json', 'w', encoding='utf-8') as f:
    json.dump(log_etapa1, f, indent=2, ensure_ascii=False, default=str)
print('Salvo: run_log_tm_etapa1.json')
'''))

write_new(cells)
print('notebook criado com', len(cells), 'células')
