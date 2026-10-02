# Geradores dos notebooks v2

Scripts que **escrevem** os notebooks `_v2` a partir de células definidas em Python, e o executor que os **roda** gravando as saídas. Os notebooks só leem caches de `results/v2/`; o cálculo pesado fica em `scripts/run_v2.py` e `scripts/run_xai_v2.py`.

| Arquivo | Gera | Uso |
|---|---|---|
| `nbtools.py` | (utilitário: `md`, `code`, verificação de travessão) | importado pelos demais |
| `run_nb.py` | executa um `.ipynb` no lugar, com o Python da `.venv` | `.venv\Scripts\python.exe tools\nb_builders\run_nb.py notebooks\X.ipynb 3600` |
| `nb02v2.py` | `notebooks/02_desempenho_m4_v2.ipynb` (Fase 2a) | `python tools\nb_builders\nb02v2.py` |
| `nb08v2.py` | `notebooks/08_cenarios_adversos_v2.ipynb` (Fase 2b) | idem |
| `nb09v2.py` | explicabilidade v2; argumento: `m4`, `cp` ou `tm` | `python tools\nb_builders\nb09v2.py tm` gera `11c_explicabilidade_tourism_v2.ipynb` |
| `nb10v2.py` | `notebooks/10_car_parts_pipeline_v2.ipynb` (Fase 3) | idem |
| `nb11a_v2.py` | `notebooks/11a_desempenho_tourism_v2.ipynb` (Fase 4a) | idem |
| `nb11b_v2.py` | `notebooks/11b_cenarios_adversos_tourism_v2.ipynb` (Fase 4b) | idem |

**Históricos, não rodar de novo:** `etapa1.py` e `etapa1b.py` (criaram as Etapas 1 e 1b do NB11; `etapa1.py` recria o NB11 do zero) e `edit_nb09.py` (aplicou a correção do LIME ao NB09; as verificações de texto falham se rodado de novo).

Os caminhos dos notebooks estão fixos (absolutos) nos geradores; ajuste se o projeto mudar de pasta.
