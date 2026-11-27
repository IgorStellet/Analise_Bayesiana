"""Confere todos os conjuntos com o mesmo interpretador, sem executar os notebooks."""

from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent
CHECKS = (
    ("Slides e complementos MCMC", "notebooks/exercicios_slides/verificar_ambiente.py", "mcmc"),
    ("Listas", "notebooks/listas/verificar_ambiente.py"),
    ("Aulas práticas", "notebooks/aulas_praticas/verificar_ambiente.py"),
    ("Projeto DALI", "projeto/verificar_ambiente.py"),
)

failed = []
for name, relative, *arguments in CHECKS:
    print(f"\nVerificando: {name}", flush=True)
    # Os processos separados evitam que imports de uma pasta escondam problemas de outra.
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / relative), *arguments],
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode:
        failed.append(name)  # Continua para mostrar todos os conjuntos que precisam de atenção.

if failed:
    raise SystemExit("A verificação falhou em: " + ", ".join(failed))
print("\nTodos os conjuntos foram verificados com:", sys.executable)
