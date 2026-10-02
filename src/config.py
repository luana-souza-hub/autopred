"""Caminhos e parâmetros do protocolo (compartilhados pelos notebooks)."""
import sys
from pathlib import Path


def ensure_project_on_path() -> Path:
    """Garante que a raiz do projeto está em sys.path (cwd em autopred/ ou notebooks/)."""
    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / 'src' / 'paths.py').is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise FileNotFoundError(
        'Não encontrei src/paths.py. Abra o workspace na pasta autopred.'
    )


ensure_project_on_path()

from src.paths import DATA_DIR, PROJECT_ROOT, RESULT_DIR  # noqa: E402

# Protocolo experimental (padrão M4 mensal — notebooks 01–05)
SEED = 42
H = 18
MIN_TRAIN = 24
N_ORIGINS = 3
PILOT_N = 200
SEASON = 12


def verify_data_files() -> None:
    required = {
        'Monthly-train.csv': DATA_DIR / 'Monthly-train.csv',
        'Monthly-test.csv': DATA_DIR / 'Monthly-test.csv',
    }
    missing = [name for name, p in required.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f'Arquivos não encontrados em {DATA_DIR}: {missing}\n'
            f'PROJECT_ROOT = {PROJECT_ROOT}'
        )
