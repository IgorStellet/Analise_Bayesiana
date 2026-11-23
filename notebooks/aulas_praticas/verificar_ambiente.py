"""Confere os imports das práticas de MCMC e do método de grid."""

import sys
from importlib.metadata import version

import cobaya
import numpy as np
import matplotlib.pyplot as plt
import scipy
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

print("Python:", sys.version.split()[0])
print("Interpretador:", sys.executable)
for package in (
    "numpy", "scipy", "matplotlib", "sympy", "ipywidgets", "ipython",
    "cobaya", "getdist", "emcee", "astropy", "jupyterlab", "ipykernel",
):
    print(f"{package}: {version(package)}")
print("Imports das práticas de MCMC e grid concluídos.")
