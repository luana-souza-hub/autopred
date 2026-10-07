#!/usr/bin/env python
"""Refaz só a APRESENTAÇÃO das Figs. 15, 16, 17, 19, 20, 21 e 22 (M4) e dos boxplots de MAE e MAPE do Car Parts.

Nenhum dado é recalculado de outra forma: os valores vêm dos CSVs de dados congelados em
results/v2/fase5/figuras/*_dados.csv (Fase 5). As duas figuras de boxplot por série (Fig. 15 e Car Parts)
precisam dos valores por série, que esses CSVs não guardam; eles são lidos dos caches by_window com a mesma
agregação da Fase 5 (pv.agregar_por_serie), e as medianas são conferidas contra os CSVs congelados.

Estilo igual ao das Figs. 8 e 9 (tools/figuras_5_2_m4.py): Arial, cores por família, vírgula decimal, ponto
de milhar, sinal de menos tipográfico, sem título interno, 300 dpi, 15 cm de largura.

Saída em results/v2/fase5/figuras_dissertacao/ (mesmos nomes de arquivo de fase5/figuras/; nada existente
é alterado): os PNGs, figuras_dissertacao.docx (cada figura seguida da legenda atual da dissertação),
conferencia_valores_anotados.csv (valor anotado antes x agora) e MUDANCAS.md.
Uso (na raiz do projeto):  .venv\\Scripts\\python.exe tools\\figuras_apresentacao_v2.py
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figuras_5_2_m4 as f52  # noqa: E402  (estilo, cores, br(), boxplot_h(), legenda() e montagem do .docx)
from figuras_5_2_m4 import COR, NOMES, br, boxplot_h, familia, fmt_eixo, legenda, plt, pv  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

RAIZ = f52.RAIZ
V2 = RAIZ / 'results' / 'v2'
FIG = V2 / 'fase5' / 'figuras'
SAIDA = V2 / 'fase5' / 'figuras_dissertacao'
LARG = f52.LARGURA_CM / 2.54
DPI = 300
VERDE, VERDE_ESC = COR['Aprendizado de máquina'], '#00553E'
XAI = {'RandomForest': 'Random Forest', 'GradientBoosting': 'Gradient Boosting', 'SVR': 'SVR'}
CEN_NOME = {'A_ruido_30pct': 'Ruído σ = 30%', 'B_ausencias_30pct': 'Ausências 30%', 'C_outliers_5pct': 'Outliers 5%',
            'D_trunca_25pct': 'Truncamento 25%'}
ORDEM_RAP = [m for m in f52.ORDEM_T9 if m != 'sarima_auto']   # ordem da Tabela 9, sem o SARIMA automático

# Legendas atuais da dissertação (lidas do .docx em 2026-10-07, sem alteração)
LEGENDAS = {
    15: 'Figura 15 – MAPE por série no nível mais forte de cada tipo de cenário (M4 mensal, 200 séries; MAPE truncado em 150% para visualização; teste limpo)',
    16: 'Figura 16 – Perfil de robustez: Δ% do MAPE por intensidade da perturbação (M4 mensal, 200 séries)',
    17: 'Figura 17 – Top-10 da importância global por SHAP, por modelo (M4 mensal, 30 séries)',
    18: 'Figura 18 – Variabilidade da importância SHAP entre as 30 séries (M4 mensal)',
    19: 'Figura 19 – CV médio da importância SHAP entre as 30 séries (M4 mensal)',
    20: 'Figura 20 – Coerência entre importâncias SHAP e LIME normalizadas por característica, para Random Forest (r=0,959), Gradient Boosting (r=0,952) e SVR (r=0,908)',
    21: 'Figura 21 – Spearman por série entre importâncias limpas e com ruído, por modelo e método (M4 mensal, 30 séries)',
    22: 'Figura 22 – Importância SHAP normalizada média, sem e com ruído de 30%, por modelo (M4 mensal, 30 séries)',
    'cp_mae': 'Figura 23 – Car Parts, distribuição do MAE por série e por modelo, ordenada pelo MAE mediano (eixo truncado no percentil 95)',
    'cp_mape': '(boxplot do MAPE no Car Parts: sem legenda no .docx atual)',
    25: '(Figura 25, Tourism: sem legenda própria no .docx atual; a legenda numerada 25 no .docx é a dos exemplos do Car Parts)',
}
CONF = []   # (figura, item, antes, agora)


def salvar(fig, nome):
    arq = SAIDA / nome
    fig.savefig(arq, dpi=DPI, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    return arq


def ler(nome):
    return pd.read_csv(FIG / nome)


def caches(base, cond, modelos):
    arqs = [a for a in sorted((V2 / base / cond).glob(f'{base}_{cond}_*_by_window.csv')) if '.parcial' not in a.name]
    d = pd.concat([pd.read_csv(a) for a in arqs], ignore_index=True)
    return pv.agregar_por_serie(d[d.model.isin(modelos)])


def legenda_familias(fig, modelos, y=0.0):
    legenda(fig, modelos, y)


# ── Fig. 15 ─────────────────────────────────────────────────────────────────
def fig15():
    ant = ler('fig15_v2_m4_mape_cenarios_dados.csv').set_index('modelo')
    fig, axs = plt.subplots(2, 2, figsize=(LARG, 6.4), sharey=True)
    for ax, c in zip(axs.flat, CEN_NOME):
        bs = caches('m4', c, ORDEM_RAP)
        v = {m: bs[bs.model == m]['mape'].dropna().values for m in ORDEM_RAP}
        boxplot_h(ax, v, ORDEM_RAP, 2, limite=150)
        ax.set_xlim(0, 150)
        ax.xaxis.set_major_formatter(fmt_eixo())
        ax.set_xlabel('MAPE por série (%)')
        ax.set_title(CEN_NOME[c], pad=4)
        for m in ORDEM_RAP:
            CONF.append(('15', f'mediana {NOMES[m]}, {CEN_NOME[c]}', round(ant.loc[NOMES[m], c], 2), round(float(np.median(v[m])), 2)))
    for ax in axs[:, 0]:
        ax.set_yticklabels([NOMES[m] for m in ORDEM_RAP])
    for ax in axs[:, 1]:
        ax.tick_params(axis='y', length=0)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=1.6, w_pad=1.0)
    legenda_familias(fig, ORDEM_RAP)
    return salvar(fig, 'fig15_v2_m4_mape_cenarios.png')


# ── Fig. 16 ─────────────────────────────────────────────────────────────────
TIPO_CEN = {'Ruído': ['A_ruido_10pct', 'A_ruido_30pct'], 'Ausências': ['B_ausencias_10pct', 'B_ausencias_20pct', 'B_ausencias_30pct'],
            'Outliers': ['C_outliers_2pct', 'C_outliers_5pct'], 'Truncamento (fração mantida do treino)': ['D_trunca_50pct', 'D_trunca_25pct']}
MARC = {'ses': 'o', 'holt_winters': 's', 'sarima_heuristic': 'D', 'random_forest': 'o', 'gbt': 's', 'svr': '^',
        'persistence': 'o', 'seasonal_naive': 's', 'mean': '^'}


def fig16():
    d = ler('fig16_v2_m4_perfil_robustez_dados.csv')
    fig, axs = plt.subplots(2, 2, figsize=(LARG, 5.6))
    for ax, (t, cs) in zip(axs.flat, TIPO_CEN.items()):
        xs = np.arange(len(cs) + 1)
        for m in ORDEM_RAP:
            sub = d[d.model == m].set_index('cenario').loc[cs]
            cor = COR[familia(m)]
            ls = '--' if pv.FAMILIA[m] == 'Baseline' else '-'
            ax.plot(xs, [0] + list(sub.delta_pct), marker=MARC[m], ms=3.2, lw=1.0, color=cor, ls=ls)
            ax.fill_between(xs, [0] + list(sub.ic_pct_inf), [0] + list(sub.ic_pct_sup), color=cor, alpha=0.06, lw=0)
        ax.set_xticks(xs)
        ax.set_xticklabels(['limpo'] + [br(100 * pv.CENARIOS[c][1], 0) + '%' for c in cs])
        ax.axhline(0, color='black', lw=0.6)
        ax.yaxis.set_major_formatter(fmt_eixo())
        ax.set_title(t, pad=4)
        ax.set_ylabel('Δ% do MAPE')
        ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout(rect=(0, 0.13, 1, 1), h_pad=1.4, w_pad=1.2)
    h = [Line2D([], [], color=COR[familia(m)], marker=MARC[m], ms=3.5, lw=1.0, ls='--' if pv.FAMILIA[m] == 'Baseline' else '-',
                label=NOMES[m]) for m in ORDEM_RAP]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False, fontsize=7,
               title='Modelo (cor = família; tracejado = baseline; faixa = IC 95%)', title_fontsize=7.5)
    return salvar(fig, 'fig16_v2_m4_perfil_robustez.png')


# ── Figs. 17 e 19 ───────────────────────────────────────────────────────────
def top10(arq_dados, nome, metodo):
    d = ler(arq_dados)
    fig, axs = plt.subplots(1, 3, figsize=(LARG, 3.0))
    for ax, m in zip(axs, XAI):
        t = d[(d.modelo == m) & (d['rank'] <= 10)].sort_values('rank', ascending=False)
        cores = [VERDE_ESC if f == 'lag_12' else VERDE for f in t.feature]
        ax.barh(t.feature, t.importancia, color=cores, alpha=0.85, edgecolor='black', linewidth=0.4)
        ax.set_title(XAI[m], pad=4)
        ax.xaxis.set_major_formatter(fmt_eixo())
        ax.set_xlabel(f'{metodo} normalizado')
        ax.tick_params(axis='y', labelsize=7, pad=2)
        ax.grid(axis='x', alpha=0.3, lw=0.4)
        ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=2.2)
    fig.legend(handles=[Patch(facecolor=VERDE_ESC, edgecolor='black', linewidth=0.4, label='lag_12 (lag sazonal)'),
                        Patch(facecolor=VERDE, alpha=0.85, edgecolor='black', linewidth=0.4, label='demais características')],
               loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    return salvar(fig, nome)


# ── Fig. 18 ─────────────────────────────────────────────────────────────────
def fig18():
    """Importância SHAP normalizada por série (soma 1), condição limpa, 30 séries: mesma agregação da Fase 5
    (pv.importancia_por_serie + pv.normalizar_por_serie sobre m4/xai/xai_m4_limpo_pontos.csv). Top-8 pela média
    entre séries; CV médio normalizado (15 características) lido de fase5/reanalise_cv_q1_m4.csv."""
    fn = pv.feature_names()
    pts = pd.read_csv(V2 / 'm4' / 'xai' / 'xai_m4_limpo_pontos.csv')
    imp = pv.normalizar_por_serie(pv.importancia_por_serie(pts, fn), fn)
    imp = imp[imp.metodo == 'shap']
    cv = pd.read_csv(V2 / 'fase5' / 'reanalise_cv_q1_m4.csv').query("metodo == 'shap' and escala == 'normalizada por série'") \
        .set_index('modelo')['CV médio (15 features)']
    top17 = ler('fig17_v2_m4_top10_shap_dados.csv').set_index(['modelo', 'feature'])['importancia']
    fig, axs = plt.subplots(1, 3, figsize=(LARG, 3.4))
    for ax, m in zip(axs, XAI):
        X = imp[imp.modelo == m][fn]
        assert len(X) == 30
        mu = X.mean()
        top8 = list(mu.sort_values(ascending=False).index[:8])
        cv_calc = float(np.nanmean(X.std(ddof=1) / mu))   # conferência do CV congelado
        CONF.append(('18', f'CV médio {XAI[m]}', round(float(cv[m]), 3), round(cv_calc, 3)))
        for f in top8[:3]:
            CONF.append(('18', f'média {f} {XAI[m]} (= Fig. 17)', round(float(top17[(m, f)]), 4), round(float(mu[f]), 4)))
        bp = ax.boxplot([X[f].values for f in top8], widths=0.55, patch_artist=True,
                        medianprops={'color': 'black', 'linewidth': 1.2}, whiskerprops={'linewidth': 0.8},
                        capprops={'linewidth': 0.8},
                        flierprops={'marker': 'o', 'markersize': 2.2, 'markerfacecolor': VERDE, 'markeredgecolor': VERDE})
        for caixa, f in zip(bp['boxes'], top8):
            caixa.set(facecolor=VERDE_ESC if f == 'lag_12' else VERDE, alpha=0.55 if f != 'lag_12' else 0.8,
                      edgecolor='black', linewidth=0.7)
        ax.set_xticks(range(1, 9))
        ax.set_xticklabels(top8, rotation=45, ha='right', fontsize=7)
        ax.set_title(f'{XAI[m]}\nCV médio = {br(cv_calc, 2)}', pad=4)   # valor sem arredondar (o CSV guarda 3 casas)
        ax.yaxis.set_major_formatter(fmt_eixo())
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=0.3, lw=0.4)
        ax.set_axisbelow(True)
    axs[0].set_ylabel('SHAP normalizado por série')
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.4)
    fig.legend(handles=[Patch(facecolor=VERDE_ESC, alpha=0.8, edgecolor='black', linewidth=0.7, label='lag_12 (lag sazonal)'),
                        Patch(facecolor=VERDE, alpha=0.55, edgecolor='black', linewidth=0.7, label='demais características')],
               loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    return salvar(fig, 'fig18_v2_m4_variabilidade_shap.png')


# ── Fig. 20 ─────────────────────────────────────────────────────────────────
def fig20():
    d = ler('fig20_v2_m4_shap_x_lime_dados.csv')
    cc = pd.read_csv(V2 / 'm4' / 'xai' / 'xai_m4_concordancia_shap_lime.csv').query("condicao == 'limpo'").set_index('modelo')
    fig, axs = plt.subplots(1, 3, figsize=(LARG, 2.75))
    for ax, m in zip(axs, XAI):
        s = d[d.modelo == m]
        r = float(np.corrcoef(s.shap, s.lime)[0, 1])
        CONF.append(('20', f'r {XAI[m]}', round(float(cc.loc[m, 'pearson_global']), 3), round(r, 3)))
        ax.plot([0, 1.05], [0, 1.05], ls='--', color='grey', lw=0.7)
        ax.scatter(s.shap, s.lime, s=14, color=[VERDE_ESC if f == 'lag_12' else VERDE for f in s.feature],
                   edgecolor='black', linewidth=0.3, zorder=3)
        for f, off, va in [('lag_1', (-6, -2), 'top'), ('roll_mean_3', (-4, 4), 'bottom')]:   # lag_12 identificado pela cor
            row = s[s.feature == f].iloc[0]
            ax.annotate(f, (row.shap, row.lime), xytext=off, textcoords='offset points', fontsize=6.3, ha='right', va=va)
        ax.set_xlim(-0.03, 1.08)
        ax.set_ylim(-0.03, 1.08)
        ax.set_aspect('equal')
        ax.xaxis.set_major_formatter(fmt_eixo())
        ax.yaxis.set_major_formatter(fmt_eixo())
        ax.set_title(f'{XAI[m]}: r = {br(r, 3)}', pad=4)
        ax.text(0.97, 0.03, f'{len(s)} características', transform=ax.transAxes, ha='right', va='bottom', fontsize=6.5)
        ax.set_xlabel('SHAP (relativo ao máximo)')
        ax.grid(alpha=0.3, lw=0.4)
    axs[0].set_ylabel('LIME (relativo ao máximo)')
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.2)
    fig.legend(handles=[Line2D([], [], ls='', marker='o', ms=4, color=VERDE_ESC, markeredgecolor='black', label='lag_12 (lag sazonal)'),
                        Line2D([], [], ls='', marker='o', ms=4, color=VERDE, markeredgecolor='black', label='demais características'),
                        Line2D([], [], ls='--', color='grey', lw=0.7, label='SHAP = LIME')],
               loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
    return salvar(fig, 'fig20_v2_m4_shap_x_lime.png')


# ── Fig. 21 ─────────────────────────────────────────────────────────────────
def fig21():
    d = ler('fig21_v2_m4_estabilidade_dados.csv')
    est = pd.read_csv(V2 / 'm4' / 'xai' / 'xai_m4_estabilidade.csv').set_index(['modelo', 'metodo'])
    mods = list(XAI)
    fig, axs = plt.subplots(1, 2, figsize=(LARG, 2.8), sharey=True)
    for ax, met in zip(axs, ['shap', 'lime']):
        v = {m: d[(d.modelo == m) & (d.metodo == met)]['spearman'].values for m in mods}
        pos = np.arange(len(mods))[::-1]
        bp = ax.boxplot([v[m] for m in mods], positions=pos, vert=False, widths=0.5, patch_artist=True,
                        medianprops={'color': 'black', 'linewidth': 1.2}, whiskerprops={'linewidth': 0.8},
                        capprops={'linewidth': 0.8},
                        flierprops={'marker': 'o', 'markersize': 2.2, 'markerfacecolor': VERDE, 'markeredgecolor': VERDE})
        for caixa in bp['boxes']:
            caixa.set(facecolor=VERDE, alpha=0.55, edgecolor='black', linewidth=0.7)
        for p, m in zip(pos, mods):
            med = float(np.median(v[m]))
            CONF.append(('21', f'mediana {XAI[m]} {met.upper()}', round(float(est.loc[(m, met), 'spearman_mediana']), 3), round(med, 3)))
            ax.text(med, p + 0.29, br(med, 2), ha='center', va='bottom', fontsize=6.8,
                    bbox={'boxstyle': 'round,pad=0.12', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85})
        ax.axvline(0.8, ls=':', color='grey', lw=0.8)
        ax.set_yticks(pos)
        ax.set_ylim(-0.6, len(mods) - 0.25)
        ax.set_title(f'{met.upper()}: sem ruído × ruído σ = 30%', pad=4)
        ax.set_xlabel('Spearman por série')
        ax.xaxis.set_major_formatter(fmt_eixo())
        ax.grid(axis='x', alpha=0.3, lw=0.4)
        ax.set_axisbelow(True)
    axs[0].set_yticklabels([XAI[m] for m in mods])
    axs[1].tick_params(axis='y', length=0)
    fig.tight_layout(w_pad=1.0)
    return salvar(fig, 'fig21_v2_m4_estabilidade.png')


# ── Fig. 22 ─────────────────────────────────────────────────────────────────
def fig22():
    d = ler('fig22_v2_m4_shap_limpo_x_ruido_dados.csv')
    fig, axs = plt.subplots(3, 1, figsize=(LARG, 6.6), sharex=True)
    for ax, m in zip(axs, XAI):
        s = d[d.modelo == m]
        x = np.arange(len(s))
        ax.bar(x - 0.2, s.limpo, 0.4, color='#BBBBBB', edgecolor='black', linewidth=0.3, label='sem ruído')
        ax.bar(x + 0.2, s.ruido30, 0.4, color=VERDE, alpha=0.85, edgecolor='black', linewidth=0.3, label='ruído σ = 30%')
        ax.set_title(XAI[m], pad=3)
        ax.set_ylabel('SHAP normalizado')
        ax.yaxis.set_major_formatter(fmt_eixo())
        ax.grid(axis='y', alpha=0.3, lw=0.4)
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels(s.feature, rotation=45, ha='right', fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.0)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    return salvar(fig, 'fig22_v2_m4_shap_limpo_x_ruido.png')


# ── Fig. 25 (Tourism): redistribuição sob ruído ─────────────────────────────
FEATS25 = ['lag_1', 'roll_mean_3', 'roll_mean_6', 'lag_12']   # mesma ordem da figura anterior (nb09v2.py)


def fig25():
    """Mesmo conteúdo de tm/xai/fig_v2_tm_redistribuicao_ruido.png (nb09v2.py, Fase 4c): médias entre séries da
    importância normalizada, sem ruído e com ruído σ = 30%, lidas de tm/xai/xai_tm_redistribuicao_ruido.csv.
    As alturas desenhadas são conferidas contra o CSV e contra as importâncias recalculadas dos caches de pontos."""
    red = pd.read_csv(V2 / 'tm' / 'xai' / 'xai_tm_redistribuicao_ruido.csv')
    fn = pv.feature_names()
    imp = pd.concat([pv.normalizar_por_serie(pv.importancia_por_serie(
        pd.read_csv(V2 / 'tm' / 'xai' / f'xai_tm_{c}_pontos.csv'), fn), fn) for c in ['limpo', 'ruido30']])
    fig, axs = plt.subplots(2, 3, figsize=(LARG, 4.6), sharey=True)
    x = np.arange(len(FEATS25))
    for i, met in enumerate(['shap', 'lime']):
        for j, m in enumerate(XAI):
            ax = axs[i, j]
            s = red[(red.modelo == m) & (red.metodo == met)].set_index('feature').loc[FEATS25]
            b0 = ax.bar(x - 0.2, s.limpo, 0.4, color='#BBBBBB', edgecolor='black', linewidth=0.3, label='sem ruído')
            b1 = ax.bar(x + 0.2, s.ruido30, 0.4, color=VERDE, alpha=0.85, edgecolor='black', linewidth=0.3, label='ruído σ = 30%')
            for cond, barras in [('limpo', b0), ('ruido30', b1)]:
                rec = imp[(imp.modelo == m) & (imp.metodo == met) & (imp.condicao == cond)][FEATS25].mean()
                for f, br_ in zip(FEATS25, barras):
                    CONF.append(('25', f'{XAI[m]} {met.upper()} {f} {"sem ruído" if cond == "limpo" else "ruído"}',
                                 float(s.loc[f, cond]), round(float(br_.get_height()), 3)))
                    assert round(float(rec[f]), 3) == float(s.loc[f, cond]), (m, met, f, cond)
            ax.set_title(f'{XAI[m]}, {met.upper()}', pad=3)
            ax.set_xticks(x)
            ax.set_xticklabels(FEATS25, rotation=30, ha='right', fontsize=7)
            ax.yaxis.set_major_formatter(fmt_eixo())
            ax.grid(axis='y', alpha=0.3, lw=0.4)
            ax.set_axisbelow(True)
        axs[i, 0].set_ylabel(f'{met.upper()} normalizado')
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=1.0, w_pad=0.8)
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    arq = salvar(fig, 'fig_v2_tm_redistribuicao_ruido.png')
    copia = SAIDA / 'fig25_v2_tm_redistribuicao_ruido.png'
    copia.write_bytes(arq.read_bytes())
    return copia


# ── Car Parts: boxplots de MAE e MAPE ───────────────────────────────────────
def car_parts():
    mods = list(pv.MODELOS)
    bs = caches('cp', 'limpo', mods)
    ant_mae = ler('fig24_v2_cp_mae_dados.csv').set_index('model')['mae']
    ant_mape = ler('fig23_v2_cp_mape_dados.csv').set_index('model')['mape']
    ordem = [next(k for k, v in NOMES.items() if v == n) for n in ant_mae.index]   # ordem do MAE mediano (Fase 5)
    p95 = float(bs['mae'].quantile(0.95))
    p95_mape = float(bs['mape'].quantile(0.95))   # só para o eixo (o MAPE por série chega a > 100.000% no SARIMA automático)
    CONF.append(('CP MAE', 'p95 do MAE (eixo)', 1.94, round(p95, 2)))
    saidas = []
    for met, nome, nd in [('mae', 'fig24_v2_cp_mae.png', 3), ('mape', 'fig23_v2_cp_mape.png', 1)]:
        v = {m: bs[bs.model == m][met].dropna().values for m in ordem}
        ant = ant_mae if met == 'mae' else ant_mape
        for m in ordem:
            CONF.append((f'CP {met.upper()}', f'mediana {NOMES[m]}', round(float(ant[NOMES[m]]), nd), round(float(np.median(v[m])), nd)))
        fig, ax = plt.subplots(figsize=(LARG, 4.6))
        boxplot_h(ax, v, ordem, nd, limite=p95 if met == 'mae' else p95_mape)
        ax.set_yticklabels([NOMES[m] for m in ordem])
        if met == 'mae':
            ax.set_xlim(0, p95)
            ax.set_xlabel(f'MAE por série, truncado no p95 ({br(p95, 2)})')
        else:
            ax.set_xlim(0, p95_mape)
            ax.set_xlabel(f'MAPE por série (%), só passos com demanda não nula; truncado no p95 ({br(p95_mape, 1)})')
        ax.xaxis.set_major_formatter(fmt_eixo())
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        legenda_familias(fig, ordem)
        saidas.append(salvar(fig, nome))
    return saidas


# ── .docx: figura seguida da legenda atual ──────────────────────────────────
def montar_docx(itens, arq):
    corpo, rels = [], []
    for i, (png, leg) in enumerate(itens):
        rid = f'rIdImg{i + 1}'
        rels.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                    f'Target="media/{png.name}"/>')
        corpo.append(f52.par_imagem(rid, i + 1, png.name, *f52.png_tamanho(png)))
        corpo.append(f52.par(leg, 11))
        if i < len(itens) - 1:
            corpo.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
    doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
           'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
           'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
           'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
           'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:body>' + ''.join(corpo) +
           '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
           '<w:pgMar w:top="1701" w:right="1134" w:bottom="1134" w:left="1701" w:header="709" w:footer="709" w:gutter="0"/>'
           '</w:sectPr></w:body></w:document>')
    tipos = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
             '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
             '<Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>'
             '<Override PartName="/word/document.xml" '
             'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
    rel_raiz = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                'Target="word/document.xml"/></Relationships>')
    rel_doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
               '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + ''.join(rels) + '</Relationships>')
    with zipfile.ZipFile(arq, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', tipos)
        z.writestr('_rels/.rels', rel_raiz)
        z.writestr('word/document.xml', doc)
        z.writestr('word/_rels/document.xml.rels', rel_doc)
        for png, _ in itens:
            z.write(png, f'word/media/{png.name}')


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    f15, f16 = fig15(), fig16()
    f17 = top10('fig17_v2_m4_top10_shap_dados.csv', 'fig17_v2_m4_top10_shap.png', 'SHAP')
    f18 = fig18()
    f19 = top10('fig19_v2_m4_top10_lime_dados.csv', 'fig19_v2_m4_top10_lime.png', 'LIME')
    f20, f21, f22 = fig20(), fig21(), fig22()
    f25 = fig25()
    cp_mae, cp_mape = car_parts()
    itens = [(f15, LEGENDAS[15]), (f16, LEGENDAS[16]), (f17, LEGENDAS[17]), (f18, LEGENDAS[18]), (f19, LEGENDAS[19]), (f20, LEGENDAS[20]),
             (f21, LEGENDAS[21]), (f22, LEGENDAS[22]), (f25, LEGENDAS[25]), (cp_mae, LEGENDAS['cp_mae']), (cp_mape, LEGENDAS['cp_mape'])]
    montar_docx(itens, SAIDA / 'figuras_dissertacao.docx')
    conf = pd.DataFrame(CONF, columns=['figura', 'valor anotado', 'antes', 'agora'])
    conf['idêntico'] = conf['antes'] == conf['agora']
    conf.to_csv(SAIDA / 'conferencia_valores_anotados.csv', index=False)
    print(conf.to_string(index=False))
    print(f'\n{int(conf["idêntico"].sum())} de {len(conf)} valores anotados idênticos')
    for png, _ in itens:
        w, h = f52.png_tamanho(png)
        print(f'{png.name}: {w} x {h} px = {w / DPI * 2.54:.1f} x {h / DPI * 2.54:.1f} cm')
    assert conf['idêntico'].all()


if __name__ == '__main__':
    main()
