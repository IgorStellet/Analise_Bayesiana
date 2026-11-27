"""Confere os imports das práticas de MCMC e do método de grid."""

import sys
from pathlib import Path
from importlib.metadata import version

import cobaya
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import camb
import h5py
from scipy import stats, integrate, interpolate, optimize, special
import sympy as sp
import ipywidgets as widgets
from IPython.display import display, Markdown
from cobaya.run import run
from cobaya.samplers.mcmc import plot_progress
from astropy.cosmology import FlatLambdaCDM, LambdaCDM
from getdist.mcsamples import MCSamplesFromCobaya, MCSamples
import getdist.plots as gdplt
import emcee
import jupyterlab
import ipykernel

sys.path.insert(0, str(Path(__file__).resolve().parent))  # Encontra o módulo local sem depender da pasta atual.
import mymodule  # Confere também os imports da prática cosmológica, sem carregar os dados.

print("Python:", sys.version.split()[0])
print("Interpretador:", sys.executable)
for package in (
    "numpy", "scipy", "matplotlib", "pandas", "camb", "h5py", "sympy", "ipywidgets", "ipython",
    "cobaya", "getdist", "emcee", "astropy", "jupyterlab", "ipykernel",
):
    print(f"{package}: {version(package)}")
print("Imports das práticas de MCMC e grid concluídos.")
