"""Caminhos do projeto autopred (independente do cwd do Jupyter)."""
from pathlib import Path


def find_project_root() -> Path:
    """Sobe a árvore de pastas até achar data/raw/."""
    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / 'data' / 'raw').is_dir():
            return candidate
    raise FileNotFoundError(
        'Não encontrei a pasta data/raw/. '
        'Execute o notebook com o projeto autopred aberto no workspace.'
    )


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
RESULT_DIR = PROJECT_ROOT / 'results'
