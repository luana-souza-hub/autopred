"""Executa um .ipynb in-place com jupyter_client (sem nbclient), gravando as saídas.

Uso: python run_nb.py caminho.ipynb [timeout_por_celula_s]
"""
import json
import sys
from pathlib import Path
from queue import Empty

from jupyter_client.manager import KernelManager

nb_path = Path(sys.argv[1]).resolve()
timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 3600

nb = json.loads(nb_path.read_text(encoding='utf-8'))
km = KernelManager()
km.kernel_cmd = None
km.kernel_spec.argv[0] = sys.executable   # garante o python da .venv
km.start_kernel(cwd=str(nb_path.parent))
kc = km.client()
kc.start_channels()
kc.wait_for_ready(timeout=120)

exec_count = 0
ok = True
try:
    kc.execute('import matplotlib; matplotlib.use("module://matplotlib_inline.backend_inline")', silent=True)
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        msg_id = kc.execute(src)
        outputs = []
        while True:
            try:
                msg = kc.get_iopub_msg(timeout=timeout)
            except Empty:
                raise TimeoutError(f'célula {idx} excedeu {timeout}s')
            if msg['parent_header'].get('msg_id') != msg_id:
                continue
            t, c = msg['msg_type'], msg['content']
            if t == 'status' and c['execution_state'] == 'idle':
                break
            if t == 'execute_input':
                exec_count = c['execution_count']
            elif t == 'stream':
                if outputs and outputs[-1]['output_type'] == 'stream' and outputs[-1]['name'] == c['name']:
                    outputs[-1]['text'] += c['text']
                else:
                    outputs.append({'output_type': 'stream', 'name': c['name'], 'text': c['text']})
            elif t in ('display_data', 'execute_result'):
                o = {'output_type': t, 'data': c['data'], 'metadata': c.get('metadata', {})}
                if t == 'execute_result':
                    o['execution_count'] = c['execution_count']
                outputs.append(o)
            elif t == 'error':
                outputs.append({'output_type': 'error', 'ename': c['ename'],
                                'evalue': c['evalue'], 'traceback': c['traceback']})
                ok = False
        for o in outputs:   # formato nbformat: texto como lista de linhas
            if o['output_type'] == 'stream':
                o['text'] = o['text'].splitlines(keepends=True)
        cell['outputs'] = outputs
        cell['execution_count'] = exec_count
        first = src.strip().splitlines()[0][:70] if src.strip() else ''
        print(f'[{idx}] {"ERRO" if not ok else "ok"}  {first}', flush=True)
        if not ok:
            for o in outputs:
                if o['output_type'] == 'error':
                    print(o['ename'], o['evalue'])
            break
finally:
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    kc.stop_channels()
    km.shutdown_kernel(now=True)

sys.exit(0 if ok else 1)
