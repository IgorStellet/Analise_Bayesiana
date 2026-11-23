"""Confere os imports das três listas e a integração usada na Lista 2."""

import sys
from importlib.metadata import version

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import stats, special, integrate, optimize
from scipy.fft import rfft, irfft, next_fast_len
from scipy.signal import fftconvolve
from IPython.display import display, Markdown
from IPython.utils.capture import capture_output
import jupyterlab
import ipykernel

if not hasattr(np, "trapezoid"):
    raise SystemExit("A Lista 2 utiliza np.trapezoid: instale NumPy 2.0 ou superior.")

print("Python:", sys.version.split()[0])
print("Interpretador:", sys.executable)
for package in ("numpy", "scipy", "matplotlib", "ipython", "jupyterlab", "ipykernel"):
    print(f"{package}: {version(package)}")
print("Imports das listas concluídos; np.trapezoid disponível.")
