#!/usr/bin/env python
"""Figuras 8 e 9 da seção 5.2 (M4, 200 séries, condição limpa) e o .docx para colar na dissertação.

Lê só os caches by_window de results/v2/m4/limpo/ (agregação da Tabela 9: mediana das janelas de cada
série, depois entre séries; janelas OK e em fallback). Não altera nenhum arquivo existente.

Saída em results/v2/fase5/figuras/:
  fig08_m4_mape_mae_r2_baselines_hw.png (+ _dados.csv)
  fig09_m4_mape_mase_10_modelos.png (+ _dados.csv)
  figuras_5_2_m4.docx
Uso (na raiz do projeto):  .venv\\Scripts\\python.exe tools\\figuras_5_2_m4.py
"""
from __future__ import annotations

import struct
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src import pipeline_v2 as pv  # noqa: E402

V2 = RAIZ / 'results' / 'v2'
SAIDA = V2 / 'fase5' / 'figuras'
LARGURA_CM = 15.0
DPI = 300

NOMES = {'holt_winters': 'Holt-Winters', 'sarima_auto': 'SARIMA automático', 'gbt': 'Gradient Boosting', 'ses': 'SES',
         'sarima_heuristic': 'SARIMA heurístico', 'persistence': 'Persistência', 'random_forest': 'Random Forest',
         'svr': 'SVR', 'seasonal_naive': 'Sazonal ingênuo', 'mean': 'Média histórica'}
ORDEM_T9 = list(NOMES)                       # ordem da Tabela 9 (MAPE mediano)
FIG8 = ['holt_winters', 'persistence', 'seasonal_naive', 'mean']
FAMILIA = {'Baseline': 'Baseline', 'Suavização exponencial': 'Suavização exponencial', 'ARIMA': 'ARIMA',
           'ML': 'Aprendizado de máquina'}
COR = {'Suavização exponencial': '#0072B2', 'ARIMA': '#E69F00', 'Aprendizado de máquina': '#009E73', 'Baseline': '#8C8C8C'}

plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8.5,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 8, 'legend.fontsize': 7.5, 'axes.unicode_minus': True})


def br(x, nd=2):
    """Número com vírgula decimal, ponto de milhar e sinal de menos tipográfico."""
    s = f'{abs(x):,.{nd}f}'.replace(',', '§').replace('.', ',').replace('§', '.')
    return ('−' if x < 0 else '') + s


def fmt_eixo(nd_max=2):
    def f(x, _pos):
        if x == 0:
            return '0'
        nd = 0 if float(x).is_integer() else (1 if abs(x * 10 - round(x * 10)) < 1e-9 else nd_max)
        return br(x, nd)
    return FuncFormatter(f)


def familia(m):
    return FAMILIA[pv.FAMILIA[m]]


# ── dados ───────────────────────────────────────────────────────────────────
def dados():
    arqs = [a for a in sorted((V2 / 'm4' / 'limpo').glob('m4_limpo_*_by_window.csv')) if '.parcial' not in a.name]
    d = pd.concat([pd.read_csv(a) for a in arqs], ignore_index=True)
    ids, series = pv.load_m4_piloto(RAIZ / 'data' / 'raw', 200)
    d = pv.adicionar_mase(d, dict(zip(ids, series)))
    assert d.unique_id.nunique() == 200 and d.model.nunique() == 10 and d.janela.max() == 3
    return d, pv.agregar_por_serie(d), pv.agregar_por_serie(d, excluir_explosivas=True)


# ── figuras ─────────────────────────────────────────────────────────────────
def boxplot_h(ax, valores, modelos, nd, anotar=True, limite=None):
    pos = np.arange(len(modelos))[::-1]          # 1º modelo no topo
    bp = ax.boxplot([valores[m] for m in modelos], positions=pos, vert=False, widths=0.5, patch_artist=True,
                    showfliers=True, medianprops={'color': 'black', 'linewidth': 1.2},
                    whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8},
                    flierprops={'marker': 'o', 'markersize': 2.2, 'markeredgewidth': 0.4, 'alpha': 0.7})
    for caixa, fl, m in zip(bp['boxes'], bp['fliers'], modelos):
        cor = COR[familia(m)]
        caixa.set(facecolor=cor, alpha=0.55, edgecolor='black', linewidth=0.7)
        fl.set(markerfacecolor=cor, markeredgecolor=cor)
    if anotar:
        for p, m in zip(pos, modelos):
            med = float(np.median(valores[m]))
            x = med if limite is None else min(med, limite)
            ax.text(x, p + 0.29, br(med, nd), ha='center', va='bottom', fontsize=6.8,
                    bbox={'boxstyle': 'round,pad=0.12', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85})
    ax.set_yticks(pos)
    ax.set_ylim(-0.6, len(modelos) - 0.25)
    ax.grid(axis='x', linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)
    return pos


def legenda(fig, modelos, y):
    fams = list(dict.fromkeys(familia(m) for m in modelos))
    fig.legend(handles=[Patch(facecolor=COR[f], alpha=0.55, edgecolor='black', linewidth=0.7, label=f) for f in fams],
               loc='lower center', bbox_to_anchor=(0.5, y), ncol=len(fams), frameon=False, title='Família',
               title_fontsize=7.5)


def figura8(bs):
    v = {met: {m: bs[bs.model == m][met].dropna().values for m in FIG8} for met in ['mape', 'mae', 'r2']}
    fig, axs = plt.subplots(3, 1, figsize=(LARGURA_CM / 2.54, 6.3))
    rot = [('mape', 'MAPE por série (%)', 2), ('mae', 'MAE por série (escala logarítmica)', 2),
           ('r2', 'R² por série (escala logarítmica simétrica)', 2)]
    for ax, (met, xlabel, nd) in zip(axs, rot):
        boxplot_h(ax, v[met], FIG8, nd)
        ax.set_yticklabels([NOMES[m] for m in FIG8])
        ax.set_xlabel(xlabel)
        if met == 'mape':
            ax.set_xlim(0, 360)
            ax.xaxis.set_major_formatter(fmt_eixo())
        elif met == 'mae':
            ax.set_xscale('log')
            ax.set_xlim(5, 2e4)
            ax.xaxis.set_major_locator(FixedLocator([10, 100, 1000, 10000]))
            ax.xaxis.set_major_formatter(fmt_eixo())
            ax.xaxis.set_minor_formatter(FuncFormatter(lambda x, p: ''))
        else:
            ax.set_xscale('symlog', linthresh=1)
            ax.set_xlim(-3000, 2)
            ax.xaxis.set_major_locator(FixedLocator([-1000, -100, -10, -1, 0, 1]))
            ax.xaxis.set_major_formatter(fmt_eixo())
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=1.2)
    legenda(fig, FIG8, 0.0)
    arq = SAIDA / 'fig08_m4_mape_mae_r2_baselines_hw.png'
    fig.savefig(arq, dpi=DPI)
    plt.close(fig)
    bs[bs.model.isin(FIG8)][['model', 'unique_id', 'mape', 'mae', 'r2']].assign(modelo=lambda x: x.model.map(NOMES)) \
        .to_csv(SAIDA / 'fig08_m4_mape_mae_r2_baselines_hw_dados.csv', index=False)
    return arq


def figura9(bs):
    v = {met: {m: bs[bs.model == m][met].dropna().values for m in ORDEM_T9} for met in ['mape', 'mase']}
    fig, axs = plt.subplots(1, 2, figsize=(LARGURA_CM / 2.54, 5.0), sharey=True)
    for ax, (met, xlabel, lim, nd) in zip(axs, [('mape', 'MAPE por série (%)', 100, 2), ('mase', 'MASE por série', 6, 3)]):
        boxplot_h(ax, v[met], ORDEM_T9, nd, limite=lim)
        ax.set_xlim(0, lim)
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_formatter(fmt_eixo())
    axs[0].set_yticklabels([NOMES[m] for m in ORDEM_T9])
    axs[1].tick_params(axis='y', length=0)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.0)
    legenda(fig, ORDEM_T9, 0.0)
    arq = SAIDA / 'fig09_m4_mape_mase_10_modelos.png'
    fig.savefig(arq, dpi=DPI)
    plt.close(fig)
    bs[['model', 'unique_id', 'mape', 'mase']].assign(modelo=lambda x: x.model.map(NOMES)) \
        .to_csv(SAIDA / 'fig09_m4_mape_mase_10_modelos_dados.csv', index=False)
    return arq


# ── conferência contra o texto ──────────────────────────────────────────────
def conferir(bs, bs_sem_exp):
    g = lambda m, met: bs[bs.model == m][met].dropna()  # noqa: E731
    texto = {  # valor citado no texto da dissertação (seção 5.2.1)
        ('mediana MAPE', 'persistence'): 9.75, ('mediana MAPE', 'mean'): 21.82, ('mediana MAPE', 'seasonal_naive'): 11.77,
        ('mediana MAPE', 'holt_winters'): 8.36, ('máximo MAPE', 'seasonal_naive'): 165.11, ('máximo MAPE', 'mean'): 335.47,
        ('Q1 MAPE', 'mean'): 11.80, ('Q3 MAPE', 'mean'): 43.08, ('mediana MAE', 'mean'): 831.59, ('mediana MAE', 'holt_winters'): 325.29,
        ('máximo MAPE', 'sarima_auto'): 184.31,
    }
    calc = {('mediana MAPE', m): g(m, 'mape').median() for m in FIG8}
    calc.update({('máximo MAPE', m): g(m, 'mape').max() for m in ['seasonal_naive', 'mean', 'sarima_auto']})
    calc.update({('Q1 MAPE', 'mean'): g('mean', 'mape').quantile(0.25), ('Q3 MAPE', 'mean'): g('mean', 'mape').quantile(0.75),
                 ('mediana MAE', 'mean'): g('mean', 'mae').median(), ('mediana MAE', 'holt_winters'): g('holt_winters', 'mae').median()})
    linhas = [(f'{k[0]} {NOMES[k[1]]}', br(texto[k]), br(calc[k]), round(calc[k], 2) == texto[k]) for k in texto]
    mx = {m: g(m, 'mape').max() for m in FIG8}
    linhas.append(('máximos de MAPE dos 4 modelos dentro de [165,11; 335,47]', '165,11 a 335,47',
                   '; '.join(f'{NOMES[m]} {br(v)}' for m, v in mx.items()), 165.11 <= min(mx.values()) and max(mx.values()) <= 335.47))
    s = bs_sem_exp[bs_sem_exp.model == 'sarima_auto']['mape'].max()
    linhas.append(('máximo MAPE SARIMA automático sem explosivas', '130,97', br(s), round(s, 2) == 130.97))
    n100_texto = {'persistence': 2, 'ses': 2, 'seasonal_naive': 2, 'holt_winters': 3, 'random_forest': 3, 'gbt': 3, 'svr': 3,
                  'sarima_auto': 5, 'sarima_heuristic': 6, 'mean': 7}
    for m, n in n100_texto.items():
        c = int((g(m, 'mape') > 100).sum())
        linhas.append((f'séries com MAPE > 100%: {NOMES[m]}', str(n), str(c), c == n))
    return pd.DataFrame(linhas, columns=['item', 'texto', 'calculado', 'bate'])


# ── .docx (OOXML mínimo, sem dependências) ──────────────────────────────────
def png_tamanho(arq):
    with open(arq, 'rb') as fh:
        cab = fh.read(24)
    return struct.unpack('>II', cab[16:24])


def par(texto, tamanho_pt, negrito=False):
    b = '<w:b/><w:bCs/>' if negrito else ''
    return ('<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="120" w:after="120"/></w:pPr>'
            f'<w:r><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/>{b}<w:sz w:val="{tamanho_pt * 2}"/>'
            f'<w:szCs w:val="{tamanho_pt * 2}"/></w:rPr><w:t xml:space="preserve">{escape(texto)}</w:t></w:r></w:p>')


def par_imagem(rid, n, nome, larg_px, alt_px):
    cx = int(LARGURA_CM * 360000)
    cy = int(cx * alt_px / larg_px)
    return ('<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing>'
            f'<wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
            f'<wp:docPr id="{n}" name="{escape(nome)}"/>'
            '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic>'
            f'<pic:nvPicPr><pic:cNvPr id="{n}" name="{escape(nome)}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
            '</wp:inline></w:drawing></w:r></w:p>')


def montar_docx(figuras, arq_docx):
    corpo, rels, n = [], [], 1
    for i, (legenda_, png, nota) in enumerate(figuras):
        rid = f'rIdImg{i + 1}'
        rels.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                    f'Target="media/{png.name}"/>')
        corpo.append(par(legenda_, 12, negrito=True))
        corpo.append(par_imagem(rid, n, png.name, *png_tamanho(png)))
        if nota:
            corpo.append(par(nota, 10))
        corpo.append(par('Fonte: Elaborada pela autora (2026).', 10))
        if i < len(figuras) - 1:
            corpo.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
        n += 1
    doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
           'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
           'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
           'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
           'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:body>'
           + ''.join(corpo) +
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
    with zipfile.ZipFile(arq_docx, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', tipos)
        z.writestr('_rels/.rels', rel_raiz)
        z.writestr('word/document.xml', doc)
        z.writestr('word/_rels/document.xml.rels', rel_doc)
        for _, png, _ in figuras:
            z.write(png, f'word/media/{png.name}')


def main():
    d, bs, bs_sem_exp = dados()
    f8, f9 = figura8(bs), figura9(bs)
    conf = conferir(bs, bs_sem_exp)
    conf.to_csv(SAIDA / 'figuras_5_2_m4_conferencia_texto.csv', index=False)
    figuras = [
        ('Figura 8 – Distribuição por série do MAPE, do MAE e do R² para o Holt-Winters e os três baselines (200 séries M4 mensal)',
         f8, 'Nota: valores por série são a mediana das três janelas de origem; medianas anotadas sobre as caixas. O MAE é '
             'mostrado em escala logarítmica e o R², em escala logarítmica simétrica (linear entre −1 e 1), porque os valores '
             'por série variam em várias ordens de grandeza.'),
        ('Figura 9 – Distribuição do MAPE e do MASE por série para os dez algoritmos nas 200 séries M4 mensal (boxplots; '
         'MAPE truncado em 100% e MASE truncado em 6; valores medianos anotados)',
         f9, 'Nota: o truncamento é só visual; caixas, bigodes e medianas são calculados com todos os valores, e as séries '
             'acima dos limites não aparecem na figura. Modelos na ordem da Tabela 9.'),
    ]
    montar_docx(figuras, SAIDA / 'figuras_5_2_m4.docx')
    pd.set_option('display.width', 200)
    print(conf.to_string(index=False))
    for f in (f8, f9):
        w, h = png_tamanho(f)
        print(f'{f.name}: {w} x {h} px = {w / DPI * 2.54:.1f} x {h / DPI * 2.54:.1f} cm a {DPI} dpi')


if __name__ == '__main__':
    main()
