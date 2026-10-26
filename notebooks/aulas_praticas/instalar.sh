#!/usr/bin/env bash
# Executar com: bash notebooks/aulas_praticas/instalar.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf 'Python ausente. Instale Python 3.10+ ou defina PYTHON_BIN.\n' >&2
    exit 1
fi
"$PYTHON_BIN" -c 'import sys; sys.exit("Use Python 3.10 ou superior.") if sys.version_info < (3, 10) else None'

if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    printf 'Ambiente inválido em %s. Defina VENV_DIR para uma nova pasta.\n' "$VENV_DIR" >&2
    exit 1
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"
"$VENV_DIR/bin/python" -m pip check
"$VENV_DIR/bin/python" "$SCRIPT_DIR/verificar_ambiente.py"

printf '\nAmbiente pronto. Para ativar nesta sessão:\n  source %q\n' "$VENV_DIR/bin/activate"
printf 'Depois execute:\n  python -m jupyterlab\n'
