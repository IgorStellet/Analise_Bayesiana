"""Confere os imports da base dos slides e, se solicitado, de MCMC."""

import argparse
import sys
from importlib.metadata import version

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("perfil", nargs="?", default="base", choices=("base", "mcmc"))
args = parser.parse_args()

import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats, special, integrate, optimize, signal, ndimage, linalg
from scipy.integrate import quad, simpson, cumulative_trapezoid
from IPython.display import display, Markdown
import jupyterlab
import ipykernel

packages = ["numpy", "scipy", "matplotlib", "pandas", "ipython", "jupyterlab", "ipykernel"]
if args.perfil == "mcmc":
    import cobaya
    from cobaya.run import run
    import emcee

    packages += ["cobaya", "emcee"]

print("Python:", sys.version.split()[0])
print("Interpretador:", sys.executable)
for package in packages:
    print(f"{package}: {version(package)}")
print("Imports dos slides concluídos; perfil:", args.perfil)
