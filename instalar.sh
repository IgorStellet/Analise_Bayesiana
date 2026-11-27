#!/usr/bin/env bash
# Instala os requisitos de todos os conjuntos na mesma .venv da raiz.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$REPO_ROOT/scripts/instalar_ambiente.sh" "$REPO_ROOT" "$@"
