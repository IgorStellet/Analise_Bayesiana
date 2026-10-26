"""Verifica os imports solicitados para as aulas práticas."""

import cobaya
import numpy as np
import matplotlib.pyplot as plt
from cobaya.run import run
from cobaya.samplers.mcmc import plot_progress
from astropy.cosmology import FlatLambdaCDM, LambdaCDM
from getdist.mcsamples import MCSamplesFromCobaya, MCSamples
import getdist.plots as gdplt
import emcee

from importlib.metadata import version
import sys

print("Python:", sys.version.split()[0])
print("Interpretador:", sys.executable)
for package in ("numpy", "matplotlib", "cobaya", "getdist", "emcee", "astropy"):
    print(f"{package}: {version(package)}")
print("Todos os imports solicitados pelo professor foram concluídos.")
