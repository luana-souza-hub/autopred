"""Utilitário para montar/estender o notebook 11 preservando saídas já executadas."""
import json
from pathlib import Path

NB_PATH = Path(r'C:\Users\Usuario\Desktop\Mestrado\Mestrado - aula\dissertação\autopred\notebooks\11_replicacao_tourism.ipynb')

META = {
    'kernelspec': {'display_name': '.venv', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11.0'},
}


def _cell(kind, src):
    src = src.strip('\n')
    lines = src.split('\n')
    source = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    c = {'cell_type': kind, 'metadata': {}, 'source': source}
    if kind == 'code':
        c['execution_count'] = None
        c['outputs'] = []
    return c


def md(s):
    return _cell('markdown', s)


def code(s):
    return _cell('code', s)


def check(cells):
    for c in cells:
        txt = ''.join(c['source'])
        assert '\u2014' not in txt, f'travessão encontrado: {txt[:80]}'


def write_new(cells):
    check(cells)
    nb = {'cells': cells, 'metadata': META, 'nbformat': 4, 'nbformat_minor': 5}
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')


def append(cells):
    check(cells)
    nb = json.loads(NB_PATH.read_text(encoding='utf-8'))
    nb['cells'].extend(cells)
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')


def extract_code(path=NB_PATH):
    nb = json.loads(Path(path).read_text(encoding='utf-8'))
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
