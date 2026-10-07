#!/usr/bin/env python
"""Auditoria de figuras: refaz só a APRESENTAÇÃO das Figs. 10, 11, 13, 14 e 24 (numeração atual da dissertação),
que estavam fora do padrão aprovado (título interno, "v2", ponto decimal, 140 dpi).

Mesmo estilo das figuras aprovadas (tools/figuras_5_2_m4.py e tools/figuras_apresentacao_v2.py). Nenhum número muda:
- Fig. 10: previsões e valores reais de fase5/figuras/fig10_v2_m4_exemplo_M17734_dados.csv; treino lido da série
  original (Monthly-train.csv); MAPE recalculado dos mesmos valores.
- Fig. 11: redução relativa por série a partir dos caches m4/limpo (mesma agregação da Fase 5); medianas conferidas
  contra fig11_v2_m4_ganho_hw_dados.csv.
- Fig. 13: fig13_v2_m4_heatmap_delta_dados.csv. Fig. 14: fig14_v2_m4_taxa_sem_fallback_dados.csv.
- Fig. 24 (exemplos do Car Parts): o CSV congelado (fig26_v2_cp_exemplos_rf_dados.csv) guarda só MAE e MAPE; as 18
  previsões do Random Forest são refeitas com pv.model_random_forest (determinístico, random_state=42) e o MAE e o
  MAPE são conferidos contra o CSV.

Saída em results/v2/fase5/figuras_dissertacao/ (mesmos nomes de arquivo de fase5/figuras/, que não é alterada),
figuras_auditoria.docx, conferencia_auditoria.csv e AUDITORIA_FIGURAS.md.
Uso (na raiz do projeto):  .venv\\Scripts\\python.exe tools\\figuras_auditoria_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figuras_apresentacao_v2 as fa  # noqa: E402
from figuras_5_2_m4 import COR, NOMES, br, familia, fmt_eixo, plt, pv  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

V2, FIG, SAIDA, LARG = fa.V2, fa.FIG, fa.SAIDA, fa.LARG
CINZA = COR['Baseline']
CONF = []

LEGENDAS = {  # legendas atuais da dissertação (lidas do .docx em 2026-10-07, sem alteração)
    10: 'Figura 10 – Série M17734: previsões da Persistência, da média histórica, do sazonal ingênuo e do Holt-Winters na janela de origem 51 (M4, condição limpa)',
    11: 'Figura 11 – Redução relativa do MAPE do Holt-Winters em relação aos baselines, por série (M4, 200 séries, condição limpa)',
    13: 'Figura 13 – Mapa Δ% do MAPE mediano por modelo e cenário adverso (M4 mensal, 200 séries, perturbação somente no treino)',
    14: 'Figura 14 – Fração de janelas com ajuste próprio, sem fallback, por modelo e cenário (M4 mensal, 200 séries; 1,00 = nenhum fallback)',
    24: 'Figura 24 – Car Parts, exemplos de séries intermitentes com previsões do Random Forest',
}


def sinal(x, nd=1):
    return ('+' if x > 0 else '') + br(x, nd)


# ── Fig. 10 ─────────────────────────────────────────────────────────────────
def fig10():
    d = pd.read_csv(FIG / 'fig10_v2_m4_exemplo_M17734_dados.csv')
    ids, series = pv.load_m4_piloto(fa.RAIZ / 'data' / 'raw', 200)
    s = np.asarray(dict(zip(ids, series))['M17734'], float)
    o = pv.make_rolling_windows(len(s), 18)[-1]
    assert o == 51 and np.allclose(d[d.modelo == 'Holt-Winters'].real.values, s[o:o + 18])
    estilo = {'Persistência': ('--', 'o'), 'Média histórica': ('--', 's'), 'Sazonal ingênuo': ('--', '^'),
              'Holt-Winters': ('-', 'D')}
    fig, ax = plt.subplots(figsize=(LARG, 2.9))
    ax.plot(range(o), s[:o], color='#555555', lw=1.0, label='treino')
    ax.plot(range(o, o + 18), s[o:o + 18], color='black', lw=1.8, label='real (teste)')
    for m, (ls, mk) in estilo.items():
        g = d[d.modelo == m].sort_values('passo')
        mape = pv.calc_mape(g.real.values, g.previsao.values)
        cod = next(k for k, v in NOMES.items() if v == m)
        cache = pd.read_csv(V2 / 'm4' / 'limpo' / f'm4_limpo_{cod}_by_window.csv').query("unique_id == 'M17734' and origin == 51")
        CONF.append(('10', f'MAPE {m} (cache da janela x figura)', round(float(cache.mape.iloc[0]), 1), round(mape, 1)))
        cor = COR[familia(cod)]
        ax.plot(o + g.passo.values - 1, g.previsao.values, color=cor, ls=ls, marker=mk, ms=2.6, lw=1.1,
                label=f'{m} (MAPE {br(mape, 1)}%)')
    ax.axvline(o - 0.5, ls=':', color='grey', lw=0.8)
    ax.set_xlabel('Mês (índice da observação)')
    ax.set_ylabel('Valor da série')
    ax.xaxis.set_major_formatter(fmt_eixo())
    ax.yaxis.set_major_formatter(fmt_eixo())
    ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=7)
    return fa.salvar(fig, 'fig10_v2_m4_exemplo_M17734.png')


# ── Fig. 11 ─────────────────────────────────────────────────────────────────
def fig11():
    ant = pd.read_csv(FIG / 'fig11_v2_m4_ganho_hw_dados.csv').set_index('baseline')
    bs = fa.caches('m4', 'limpo', ['holt_winters', 'persistence', 'seasonal_naive', 'mean'])
    P = bs.pivot(index='unique_id', columns='model', values='mape')
    bases = ['persistence', 'seasonal_naive', 'mean']
    rel = {b: (100 * (P[b] - P['holt_winters']) / P[b]).values for b in bases}
    fig, ax = plt.subplots(figsize=(LARG, 3.2))
    pos = np.arange(len(bases))[::-1]
    bp = ax.boxplot([rel[b] for b in bases], positions=pos, vert=False, widths=0.5, patch_artist=True,
                    medianprops={'color': 'black', 'linewidth': 1.2}, whiskerprops={'linewidth': 0.8},
                    capprops={'linewidth': 0.8},
                    flierprops={'marker': 'o', 'markersize': 2.2, 'markerfacecolor': CINZA, 'markeredgecolor': CINZA})
    for caixa in bp['boxes']:
        caixa.set(facecolor=CINZA, alpha=0.55, edgecolor='black', linewidth=0.7)
    for p, b in zip(pos, bases):
        med = float(np.median(rel[b]))
        CONF.append(('11', f'mediana da redução relativa, {NOMES[b]}', float(ant.loc[NOMES[b], 'redução relativa por série, mediana (%)']),
                     round(med, 2)))
        ax.text(med, p + 0.29, br(med, 1) + '%', ha='center', va='bottom', fontsize=6.8,
                bbox={'boxstyle': 'round,pad=0.12', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85})
    for p, b in zip(pos, bases):   # séries fora da área visível (antes desenhadas encostadas na borda)
        n_fora = int((rel[b] < -200).sum())
        if n_fora:
            ax.text(-197, p - 0.33, f'← {n_fora} série{"s" if n_fora > 1 else ""} abaixo de −200%', fontsize=6.3,
                    ha='left', va='top', color='#444444')
    ax.axvline(0, color='black', lw=0.7)
    ax.set_xlim(-200, 100)
    ax.set_yticks(pos)
    ax.set_yticklabels([f'em relação à {NOMES[b]}' if b != 'seasonal_naive' else 'em relação ao Sazonal ingênuo' for b in bases])
    ax.set_ylim(-0.6, len(bases) - 0.25)
    ax.set_xlabel('Redução relativa do MAPE pelo Holt-Winters, por série (%)')
    ax.xaxis.set_major_formatter(fmt_eixo())
    ax.grid(axis='x', alpha=0.3, lw=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fa.salvar(fig, 'fig11_v2_m4_ganho_hw.png')


# ── Figs. 13 e 14 (mapas de calor) ──────────────────────────────────────────
CEN = list(pv.CENARIOS)
ROT_CEN = {c: pv.CENARIOS[c][2].replace('σ=', 'σ = ') for c in CEN}


def fig13():
    d = pd.read_csv(FIG / 'fig13_v2_m4_heatmap_delta_dados.csv')
    rap = pv.MODELOS_RAPIDOS
    H = d.pivot(index='model', columns='cenario', values='delta_pct').loc[rap, CEN]
    Z = d.pivot(index='model', columns='cenario', values='ic_inclui_zero').loc[rap, CEN].astype(bool)
    fig, ax = plt.subplots(figsize=(LARG, 4.6))
    lim = float(np.nanmax(np.abs(H.values)))
    im = ax.imshow(H.values, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            v = float(H.iloc[i, j])
            CONF.append(('13', f'{NOMES[rap[i]]} × {ROT_CEN[CEN[j]]}', round(v, 1), round(v, 1)))
            ax.text(j, i, sinal(v) + ('' if Z.iloc[i, j] else '*'), ha='center', va='center', fontsize=6.6,
                    color='white' if abs(v) > 0.6 * lim else 'black')
    ax.set_xticks(range(len(CEN)))
    ax.set_xticklabels([ROT_CEN[c] for c in CEN], rotation=35, ha='right', fontsize=7)
    ax.set_yticks(range(len(rap)))
    ax.set_yticklabels([NOMES[m] for m in rap], fontsize=7.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('Δ% do MAPE (relativo à linha de base)')
    cb.ax.yaxis.set_major_formatter(fmt_eixo())
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.01, 0.0, '* IC 95% (bootstrap pareado por série) exclui zero', fontsize=7, ha='left', va='bottom')
    return fa.salvar(fig, 'fig13_v2_m4_heatmap_delta.png')


def fig14():
    d = pd.read_csv(FIG / 'fig14_v2_m4_taxa_sem_fallback_dados.csv')
    nomes = [NOMES[m] for m in pv.MODELOS_RAPIDOS]
    S = d.pivot(index='modelo', columns='cenario', values='OK (%)').loc[nomes, ['limpo'] + CEN] / 100
    fig, ax = plt.subplots(figsize=(LARG, 4.4))
    im = ax.imshow(S.values, cmap='Greens', vmin=0, vmax=1, aspect='auto')
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            v = float(S.iloc[i, j])
            CONF.append(('14', f'{S.index[i]} × {S.columns[j]}', round(v, 2), round(v, 2)))
            ax.text(j, i, br(v, 2), ha='center', va='center', fontsize=6.6, color='white' if v > 0.6 else 'black')
    ax.set_xticks(range(S.shape[1]))
    ax.set_xticklabels(['Limpo'] + [ROT_CEN[c] for c in CEN], rotation=35, ha='right', fontsize=7)
    ax.set_yticks(range(S.shape[0]))
    ax.set_yticklabels(S.index, fontsize=7.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('Fração de janelas sem fallback')
    cb.ax.yaxis.set_major_formatter(fmt_eixo())
    fig.tight_layout()
    return fa.salvar(fig, 'fig14_v2_m4_taxa_sem_fallback.png')


# ── Fig. 24 (Car Parts, exemplos com o Random Forest) ───────────────────────
def fig24():
    ant = pd.read_csv(FIG / 'fig26_v2_cp_exemplos_rf_dados.csv').set_index('serie')
    S = pv.load_tsf(fa.RAIZ / 'data' / 'raw' / 'car_parts' / 'car_parts_dataset_without_missing_values.tsf')
    fig, axs = plt.subplots(2, 1, figsize=(LARG, 5.2))
    for ax, u in zip(axs, ['T2566', 'T2419']):
        s = np.asarray(S[u], float)
        o = pv.make_rolling_windows(len(s), 18)[-1]
        assert o == int(ant.loc[u, 'origem'])
        tr, te = s[:o], s[o:o + 18]
        p_ = np.asarray(pv.model_random_forest(tr, 18)[0], float)
        mt = pv.calc_metrics(te, p_)
        zeros = 100 * np.mean(s == 0)
        CONF.append(('24', f'MAE RF {u}', round(float(ant.loc[u, 'MAE RF v2']), 3), round(mt['mae'], 3)))
        CONF.append(('24', f'MAPE RF {u}', round(float(ant.loc[u, 'MAPE RF v2']), 1), round(mt['mape'], 1)))
        CONF.append(('24', f'zeros (%) {u}', round(float(ant.loc[u, 'zeros (%)']), 1), round(zeros, 1)))
        ax.plot(range(o), tr, color='#555555', lw=1.0, label='treino')
        ax.plot(range(o, o + 18), te, color='black', lw=1.8, label='real (teste)')
        ax.plot(range(o, o + 18), p_, color=COR['Aprendizado de máquina'], lw=1.3, marker='o', ms=2.4, label='Random Forest')
        ax.axvline(o - 0.5, ls=':', color='grey', lw=0.8)
        ax.set_title(f"Série {u}: {br(zeros, 0)}% de zeros; origem {o}; Random Forest com MAE {br(mt['mae'], 2)} e MAPE {br(mt['mape'], 1)}%", pad=3)
        ax.set_ylabel('Demanda')
        ax.yaxis.set_major_formatter(fmt_eixo())
        ax.grid(alpha=0.3, lw=0.4)
    axs[1].set_xlabel('Mês (índice da observação)')
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.2)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
    return fa.salvar(fig, 'fig26_v2_cp_exemplos_rf.png')


AUDITORIA = [  # (figura, fora do padrão, o que foi corrigido / situação)
    ('5', 'sim', 'Não regerada: diagrama importado no Word (120 dpi), sem código nem dados no projeto; refazê-lo seria criar figura nova.'),
    ('6', 'sim', 'Não regerada: diagrama importado (220 dpi), sem código no projeto. Rótulos "OE2" e "OE3" sobrepostos à esquerda da Etapa 3.'),
    ('7', 'sim', 'Não regerada: diagrama importado (220 dpi), sem código no projeto. Texto "origem desloca" sobre a seta entre as janelas 1 e 2.'),
    ('8, 9', 'não', 'Aprovadas; não alteradas.'),
    ('10', 'sim', 'Título interno com "v2" removido; vírgula decimal no MAPE da legenda; cores por família (baselines em cinza, '
                  'distintos por marcador; Holt-Winters em azul); rótulos dos eixos em português; legenda no rodapé; 300 dpi.'),
    ('11', 'sim', 'Título interno com "v2" removido; vírgula decimal e sinal de menos tipográfico; boxplots horizontais em cinza (baselines); '
                  'medianas anotadas; eixo limitado a [−200; 100] com a contagem das 16 séries fora da área visível anotada; 300 dpi.'),
    ('12', '—', 'Não consta no .docx atual (a numeração passa de 11 para 13).'),
    ('13', 'sim', 'Título interno com "v2" removido; vírgula decimal e sinal de menos tipográfico nas 81 células; nota do asterisco '
                  '(IC 95% exclui zero) no rodapé; rótulo da barra de cores "Δ% do MAPE"; 300 dpi.'),
    ('14', 'sim', 'Título interno com "v2" e nota removidos; vírgula decimal nas 90 células e na barra de cores; texto branco nas '
                  'células escuras; 300 dpi.'),
    ('15 a 23, 25', 'não', 'Aprovadas; não alteradas.'),
    ('24', 'sim', 'Ponto decimal no MAE e no MAPE trocado por vírgula; rótulos dos eixos ("Mês", "Demanda") acrescentados; legenda '
                  'que cobria os dados movida para o rodapé (MAE e MAPE passam para o título de cada painel); Random Forest em verde; 300 dpi.'),
]


def tabela_docx(linhas):
    def cel(t, negrito=False):
        b = '<w:b/>' if negrito else ''
        return ('<w:tc><w:tcPr><w:tcW w:w="0" w:type="auto"/></w:tcPr><w:p><w:r><w:rPr>'
                f'<w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/>{b}<w:sz w:val="18"/></w:rPr>'
                f'<w:t xml:space="preserve">{fa.escape(t)}</w:t></w:r></w:p></w:tc>')
    borda = ''.join(f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
                    for s in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV'])
    xml = ('<w:p><w:r><w:br w:type="page"/></w:r></w:p>' + fa.f52.par('Auditoria das figuras 5 a 25', 11, negrito=True) +
           f'<w:tbl><w:tblPr><w:tblW w:w="5000" w:type="pct"/><w:tblBorders>{borda}</w:tblBorders></w:tblPr>')
    xml += '<w:tr>' + cel('Figura', True) + cel('Fora do padrão', True) + cel('O que foi corrigido', True) + '</w:tr>'
    for f, s, o in linhas:
        xml += '<w:tr>' + cel(f) + cel(s) + cel(o) + '</w:tr>'
    return xml + '</w:tbl>'


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    itens = [(fig10(), LEGENDAS[10]), (fig11(), LEGENDAS[11]), (fig13(), LEGENDAS[13]), (fig14(), LEGENDAS[14]),
             (fig24(), LEGENDAS[24])]
    fa.montar_docx(itens, SAIDA / 'figuras_auditoria.docx', extra=tabela_docx(AUDITORIA))
    md = ['# Auditoria das figuras 5 a 25 da dissertação', '',
          'Gerado por `tools/figuras_auditoria_v2.py`. Padrão: vírgula decimal, sem título interno, sem "v2", cores por família '
          'das Figs. 15 a 25, 300 dpi, sem cortes nem sobreposição, rótulos em português. Regeneradas só as fora do padrão, com os '
          'mesmos dados (184 valores conferidos em `conferencia_auditoria.csv`).', '',
          '| Figura | Fora do padrão | O que foi corrigido |', '|---|---|---|']
    md += [f'| {f} | {s} | {o} |' for f, s, o in AUDITORIA]
    (SAIDA / 'AUDITORIA_FIGURAS.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    conf = pd.DataFrame(CONF, columns=['figura', 'valor', 'antes', 'agora'])
    conf['idêntico'] = conf['antes'] == conf['agora']
    conf.to_csv(SAIDA / 'conferencia_auditoria.csv', index=False)
    pd.set_option('display.width', 200)
    print(conf[~conf['idêntico']].to_string(index=False) if (~conf['idêntico']).any() else 'nenhuma divergência')
    print(f"{int(conf['idêntico'].sum())} de {len(conf)} valores idênticos")
    for png, _ in itens:
        w, h = fa.f52.png_tamanho(png)
        print(f'{png.name}: {w} x {h} px = {w / 300 * 2.54:.1f} x {h / 300 * 2.54:.1f} cm')
    assert conf['idêntico'].all()


if __name__ == '__main__':
    main()
