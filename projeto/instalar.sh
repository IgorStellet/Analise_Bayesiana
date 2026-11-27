#!/usr/bin/env bash
# Executar da raiz: bash projeto/instalar.sh
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
exec bash "$REPO_ROOT/scripts/instalar_ambiente.sh" "$SCRIPT_DIR" "$@"
