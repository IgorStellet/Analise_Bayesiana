"""Confere os imports do projeto DALI sem executar os experimentos longos."""

from importlib.metadata import version
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))  # Encontra src mesmo ao chamar o verificador de outra pasta.

import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image  # Exportação opcional das animações da oficina.
from IPython.display import HTML, Markdown, display
import jupyterlab
import ipykernel
from src import dali, lista3, supernovas, bao, amostragem, oficina, eficiencia

print("Python:", sys.version.split()[0])
print("Interpretador:", sys.executable)
for package in ("numpy", "scipy", "matplotlib", "pandas", "pillow", "ipython", "jupyterlab", "ipykernel"):
    print(f"{package}: {version(package)}")
print("Módulos locais: dali, lista3, supernovas, bao, amostragem, oficina e eficiencia.")
print("Imports do projeto concluídos. Os testes científicos são executados separadamente.")
