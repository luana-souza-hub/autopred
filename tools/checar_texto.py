#!/usr/bin/env python
"""Confere um .docx ou .md da dissertação/artigo contra as regras de redação e os fatos congelados.

Acusa:
  1. termos proibidos: "inversão de hierarquia" fora do par Tourism x Car Parts e do Holt-Winters,
     "validação externa", "paradoxo" (e variações), "protocolo v2", "seis cenários"/"seis subcenários",
     "5σ"/"cinco desvios", "otimizado" (aviso quando a frase trata de suavização exponencial),
     marcadores "[N]"/"[N+1]" e campos com realce amarelo (só .docx);
  2. números que não constam em results/v2/pacote_escrita/00_controle/FATOS_CONGELADOS.csv.

Só lê os arquivos; não altera nada.

Uso (na raiz do projeto):
  .venv\\Scripts\\python.exe tools\\checar_texto.py caminho\\arquivo.docx
  .venv\\Scripts\\python.exe tools\\checar_texto.py secao.md --sem-numeros
  .venv\\Scripts\\python.exe tools\\checar_texto.py arquivo.docx --csv relatorio.csv

Regras dos números (heurística): considera números com vírgula decimal (8,36), com separador de milhar
(2.674) e inteiros a partir de 10; ignora anos (1900 a 2099), números colados a letras ou sublinhado
(lag_12, H1, commit 1a08fa1), números de seção (5.2.1) e os que seguem "Tabela", "Figura", "seção",
"Capítulo", "OE", "Fase" ou "item". Um número do texto com menos casas decimais é aceito quando é o
arredondamento de um número dos fatos (0,99 aceita 0,993). Pré-textuais, listas, sumário e referências
não são conferidos quanto a números.
Código de saída: 0 sem achados; 1 com achados (avisos não contam).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
FATOS_PADRAO = RAIZ / 'results' / 'v2' / 'pacote_escrita' / '00_controle' / 'FATOS_CONGELADOS.csv'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'

NUM = re.compile(r'(?<![\w.,\d\-−])[−+\-]?(?:\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?)(?![\w]|[.,]\d)')
CONTEXTO_IGNORA = re.compile(r'(Tabela|Tabelas|Figura|Figuras|seção|seções|Seção|Capítulo|capítulo|OE|Fase|item|itens|Nota|nota)\s*$')
TITULOS_SEM_NUMEROS = re.compile(r'^(AGRADECIMENTOS|LISTA DE|SUMÁRIO|REFERÊNCIAS|APÊNDICE|ANEXO)', re.I)

PROIBIDOS = [
    ('inversão de hierarquia', re.compile(r'invers[ãa]o\s+(de|da)\s+hierarquia|hierarquia[^.]{0,120}(invertid|revertid)|reversed hierarchy|hierarchy[^.]{0,120}revers', re.I)),
    ('validação externa', re.compile(r'valida[çc][ãa]o\s+externa|external\s+validation', re.I)),
    ('paradoxo', re.compile(r'parado(x|ks)\w*', re.I)),
    ('protocolo v2', re.compile(r'protocolo\s+v2', re.I)),
    ('seis cenários', re.compile(r'\bseis\s+(sub)?cen[áa]rios|\b6\s+(sub)?cen[áa]rios|six\s+(sub)?scenarios', re.I)),
    ('5σ', re.compile(r'\b5\s*σ|\b5\s*sigma|cinco\s+desvios', re.I)),
    ('otimizado', re.compile(r'\botimizad[oa]s?\b|\boptimi[sz]ed\b', re.I)),
    ('[N]', re.compile(r'\[N(\s*\+\s*\d+)?\]')),
]
PERMITE_INVERSAO = [re.compile(r'tourism', re.I), re.compile(r'car\s*parts', re.I)]
PERMITE_INVERSAO_HW = re.compile(r'holt-winters|\bHW\b', re.I)
SUAVIZACAO = re.compile(r'suaviza|\bSES\b|holt', re.I)


# ── leitura ─────────────────────────────────────────────────────────────────
def _texto_run(r):
    t = ''.join(x.text or '' for x in r.iter(W + 't'))
    rpr = r.find(W + 'rPr')
    amarelo = False
    if rpr is not None:
        h, sh = rpr.find(W + 'highlight'), rpr.find(W + 'shd')
        amarelo = (h is not None and h.get(W + 'val') == 'yellow') or \
                  (sh is not None and (sh.get(W + 'fill') or '').upper() == 'FFFF00')
    return t, amarelo


def ler_docx(path):
    """Lista de (seção, texto, trechos_amarelos) por parágrafo, incluindo células de tabela."""
    z = zipfile.ZipFile(path)
    estilos = {}
    try:
        for s in ET.fromstring(z.read('word/styles.xml')).iter(W + 'style'):
            n = s.find(W + 'name')
            estilos[s.get(W + 'styleId')] = n.get(W + 'val') if n is not None else ''
    except KeyError:
        pass
    body = ET.fromstring(z.read('word/document.xml')).find(W + 'body')
    out, secao = [], '(início)'
    for p in body.iter(W + 'p'):
        ppr = p.find(W + 'pPr')
        sid = ppr.find(W + 'pStyle').get(W + 'val') if ppr is not None and ppr.find(W + 'pStyle') is not None else ''
        partes, amarelos = [], []
        for r in p.iter(W + 'r'):
            t, am = _texto_run(r)
            partes.append(t)
            if am and t.strip():
                amarelos.append(t.strip())
        texto = ''.join(partes).strip()
        if not texto:
            continue
        if re.match(r'(?i)(heading|título|titulo)\s*\d', estilos.get(sid, sid) or ''):
            secao = texto[:80]
        out.append((secao, texto, amarelos))
    return out


def ler_md(path):
    out, secao, bloco = [], '(início)', []
    for linha in Path(path).read_text(encoding='utf-8').splitlines() + ['']:
        if linha.startswith('#'):
            if bloco:
                out.append((secao, ' '.join(bloco), []))
                bloco = []
            secao = linha.lstrip('#').strip()[:80]
        elif linha.strip():
            bloco.append(linha.strip())
        elif bloco:
            out.append((secao, ' '.join(bloco), []))
            bloco = []
    return out


# ── números ─────────────────────────────────────────────────────────────────
def normalizar(tok):
    t = tok.replace('−', '').replace('+', '').replace('-', '')
    if re.fullmatch(r'\d{1,3}(\.\d{3})+(,\d+)?', t):
        t = t.replace('.', '')
    return t


def numeros(texto):
    for m in NUM.finditer(texto):
        tok = m.group(0)
        n = normalizar(tok)
        if ',' not in n:
            v = int(n)
            if v < 10 or 1900 <= v <= 2099:
                continue
            if CONTEXTO_IGNORA.search(texto[max(0, m.start() - 12):m.start()]):
                continue
        yield tok, n, m.start()


def carregar_fatos(path):
    exatos = set()
    for r in csv.DictReader(open(path, encoding='utf-8')):
        for col in ('fato', 'valor', 'ic_95'):
            for m in NUM.finditer(r[col] or ''):
                exatos.add(normalizar(m.group(0)))
    decimais = [Decimal(x.replace(',', '.')) for x in exatos]
    return exatos, decimais


def consta(n, exatos, decimais):
    if n in exatos:
        return True
    if ',' not in n:
        return False
    casas = len(n.split(',')[1])
    alvo = Decimal(n.replace(',', '.'))
    q = Decimal(1).scaleb(-casas)
    return any(d.quantize(q, rounding=ROUND_HALF_UP) == alvo for d in decimais
               if d.as_tuple().exponent < -casas)


# ── conferência ─────────────────────────────────────────────────────────────
def frase(texto, pos):
    ini = max(texto.rfind('. ', 0, pos) + 2, 0)
    fim = texto.find('. ', pos)
    return texto[ini:fim if fim != -1 else len(texto)]


def conferir(paragrafos, fatos, com_numeros=True):
    achados = []
    exatos, decimais = fatos if fatos else (set(), [])
    # só há pré-textuais a pular quando o arquivo é a dissertação inteira (tem o título RESUMO)
    tem_pretextuais = any(s.upper().startswith('RESUMO') for s, _, _ in paragrafos)
    for i, (secao, texto, amarelos) in enumerate(paragrafos, 1):
        # pré-textuais (antes do primeiro título), listas, sumário e referências: sem conferência de números
        sem_numeros = (tem_pretextuais and secao == '(início)') or bool(TITULOS_SEM_NUMEROS.match(secao))
        if secao.upper().startswith('REFERÊNCIAS'):
            continue
        for nome, rx in PROIBIDOS:
            for m in rx.finditer(texto):
                fr = frase(texto, m.start())
                nivel = 'ACHADO'
                if nome == 'inversão de hierarquia':
                    if all(p.search(fr) for p in PERMITE_INVERSAO) or PERMITE_INVERSAO_HW.search(fr):
                        continue
                if nome == 'otimizado' and SUAVIZACAO.search(fr):
                    nivel = 'aviso'
                achados.append((nivel, i, secao, f'termo: {nome}', m.group(0), fr[:220]))
        for a in amarelos:
            achados.append(('ACHADO', i, secao, 'campo amarelo', a[:60], texto[:220]))
        if com_numeros and fatos and not sem_numeros:
            for tok, n, pos in numeros(texto):
                if not consta(n, exatos, decimais):
                    achados.append(('ACHADO', i, secao, 'número fora de FATOS', tok, frase(texto, pos)[:220]))
    return achados


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('arquivo')
    ap.add_argument('--fatos', default=str(FATOS_PADRAO))
    ap.add_argument('--sem-numeros', action='store_true', help='só termos proibidos e campos amarelos')
    ap.add_argument('--csv', help='grava os achados em CSV')
    a = ap.parse_args()
    p = Path(a.arquivo)
    paragrafos = ler_docx(p) if p.suffix.lower() == '.docx' else ler_md(p)
    fatos = None if a.sem_numeros else carregar_fatos(a.fatos)
    achados = conferir(paragrafos, fatos, not a.sem_numeros)
    for nivel, i, secao, tipo, tok, trecho in achados:
        print(f'[{nivel}] §{i} ({secao[:40]}) {tipo}: "{tok}" | {trecho}')
    n_achados = sum(x[0] == 'ACHADO' for x in achados)
    print(f'\n{len(paragrafos)} parágrafos; {n_achados} achados; {len(achados) - n_achados} avisos')
    if a.csv:
        with open(a.csv, 'w', encoding='utf-8', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['nivel', 'paragrafo', 'secao', 'tipo', 'trecho_encontrado', 'frase'])
            w.writerows(achados)
    sys.exit(1 if n_achados else 0)


if __name__ == '__main__':
    main()
