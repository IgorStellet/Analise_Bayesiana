#!/usr/bin/env bash
# Prepara a única .venv do repositório para o conjunto solicitado.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ENV_DIR="$REPO_ROOT/.venv"

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Uso: bash scripts/instalar_ambiente.sh <pasta do conjunto> [mcmc]" >&2
    exit 2
fi

CONJUNTO_DIR="$(cd -- "$1" && pwd)"  # Recebe a pasta do projeto, das aulas ou da raiz.
PERFIL="${2:-base}"
REQUISITOS="$CONJUNTO_DIR/requirements.txt"
VERIFICADOR="$CONJUNTO_DIR/verificar_ambiente.py"
case "$PERFIL" in
    base) ;;
    mcmc) REQUISITOS="$CONJUNTO_DIR/requirements_mcmc.txt" ;;
    *) echo "Perfil desconhecido: $PERFIL. Use base ou mcmc." >&2; exit 2 ;;
esac
if [[ ! -f "$REQUISITOS" || ! -f "$VERIFICADOR" ]]; then
    echo "Não encontrei os requisitos ou o verificador em: $CONJUNTO_DIR" >&2
    exit 2
fi

if [[ -e "$ENV_DIR" ]]; then
    # Reaproveita o ambiente sem apagá-lo nem recriá-lo.
    if [[ ! -f "$ENV_DIR/pyvenv.cfg" || ! -x "$ENV_DIR/bin/python" ]]; then
        echo "A pasta $ENV_DIR não contém uma venv utilizável neste sistema." >&2
        echo "No Windows sem WSL, use as instruções do README central." >&2
        exit 1
    fi
    echo "Reaproveitando: $ENV_DIR"
else
    ANALISE_PYTHON="${ANALISE_PYTHON:-python3}"  # Permite indicar outro executável Python.
    "$ANALISE_PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "Use Python 3.10 ou superior.")'
    "$ANALISE_PYTHON" -m venv --prompt analise-bayesiana "$ENV_DIR"
    echo "Ambiente criado: $ENV_DIR"
fi

ENV_PYTHON="$ENV_DIR/bin/python"  # Todas as operações abaixo usam o Python da mesma .venv.
"$ENV_PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "A .venv precisa de Python 3.10 ou superior.")'
"$ENV_PYTHON" -m pip install --disable-pip-version-check -r "$REQUISITOS"
"$ENV_PYTHON" -m pip check  # Confere se as dependências instaladas são compatíveis.
if [[ "$PERFIL" == "mcmc" ]]; then
    "$ENV_PYTHON" "$VERIFICADOR" mcmc
else
    "$ENV_PYTHON" "$VERIFICADOR"
fi
"$ENV_PYTHON" -m ipykernel install --sys-prefix --name analise-bayesiana --display-name "Análise Bayesiana (.venv)"

echo "Instalação e verificação concluídas."
echo "Na raiz do repositório, execute:"
echo "  source .venv/bin/activate"
echo "  python -m jupyterlab"
