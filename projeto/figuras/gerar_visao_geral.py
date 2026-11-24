"""Gera a figura didática de orientação usada no README.

Modelo: d = (0, 0), mu(x, y) = (x, y - b*x**2), com covariância
diagonal e priors uniformes no retângulo mostrado. A expansão do modelo
até segunda ordem é exata. Esta figura acompanha o planejamento.

Executar com NumPy e Matplotlib disponíveis:
    python projeto/figuras/gerar_visao_geral.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def normalizar(log_densidade: np.ndarray, area: float) -> np.ndarray:
    """Normaliza uma densidade tabulada em células de áreas iguais."""
    densidade = np.exp(log_densidade - np.max(log_densidade))
    return densidade / (densidade.sum() * area)


def limiares_hpd(densidade: np.ndarray, area: float) -> np.ndarray:
    """Encontra os níveis para regiões de 68,3% e 95,4% da massa."""
    ordenada = np.sort(densidade.ravel())[::-1]
    acumulada = np.cumsum(ordenada) * area
    indices = np.searchsorted(acumulada, [0.683, 0.954])
    return np.sort(ordenada[indices])


def main() -> None:
    destino = Path(__file__).resolve().parent
    n = 520
    bordas_x = np.linspace(-3.0, 3.0, n + 1)
    bordas_y = np.linspace(-1.8, 5.6, n + 1)
    x = (bordas_x[:-1] + bordas_x[1:]) / 2
    y = (bordas_y[:-1] + bordas_y[1:]) / 2
    area = (bordas_x[1] - bordas_x[0]) * (bordas_y[1] - bordas_y[0])
    xx, yy = np.meshgrid(x, y)
    b, sigma_x, sigma_y = 0.55, 1.0, 0.45

    # O prior é constante no domínio e desaparece na normalização.
    log_exata = -0.5 * ((xx / sigma_x) ** 2 + ((yy - b * xx**2) / sigma_y) ** 2)
    log_fisher = -0.5 * ((xx / sigma_x) ** 2 + (yy / sigma_y) ** 2)

    # Em theta_0 = (0, 0), mu_0 = d e os únicos termos não lineares
    # pertencem à segunda derivada do modelo.
    mu2_x = xx
    mu2_y = yy + 0.5 * (-2 * b) * xx**2
    log_dubleto = -0.5 * ((mu2_x / sigma_x) ** 2 + (mu2_y / sigma_y) ** 2)

    exata = normalizar(log_exata, area)
    fisher = normalizar(log_fisher, area)
    dubleto = normalizar(log_dubleto, area)
    if not np.allclose(exata, dubleto, rtol=2e-14, atol=2e-14):
        raise AssertionError("A expansão quadrática deveria reproduzir o modelo exato.")
    for densidade in (exata, fisher, dubleto):
        if not np.isclose(densidade.sum() * area, 1.0):
            raise AssertionError("A densidade precisa estar normalizada.")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    })
    fig, eixos = plt.subplots(1, 2, figsize=(11.6, 5.1), sharex=True, sharey=True)
    fig.patch.set_facecolor("#faf9f5")
    nivel_exato = limiares_hpd(exata, area)
    cores = ["#245f94", "#b35c26"]
    titulos = ["Fisher: geometria local", "DALI: modelo até segunda ordem"]
    for ax, aproximada, cor, titulo in zip(eixos, (fisher, dubleto), cores, titulos):
        ax.set_facecolor("#faf9f5")
        ax.contourf(xx, yy, exata, levels=[*nivel_exato, exata.max() * 1.001],
                    colors=["#ece9df", "#d7d2c4"])
        ax.contour(xx, yy, exata, levels=nivel_exato, colors="#77756e", linewidths=3.0)
        ax.contour(xx, yy, aproximada, levels=limiares_hpd(aproximada, area),
                   colors=cor, linewidths=1.8, linestyles="--")
        ax.plot(0, 0, "o", color="#383a3a", markersize=4)
        ax.set(xlabel="$x$", title=titulo, xlim=(-2.9, 2.9), ylim=(-1.5, 5.0))
        ax.set_aspect("equal", adjustable="box")
        ax.legend(handles=[
            Line2D([], [], color="#77756e", lw=3, label="Referência"),
            Line2D([], [], color=cor, lw=1.8, ls="--", label="Aproximação"),
        ], loc="upper right", frameon=False, fontsize=9)
    eixos[0].set_ylabel("$y$")
    fig.suptitle("A persistência da não gaussianidade | Fisher e DALI", fontsize=15,
                 fontweight="bold", color="#303b43", y=0.98)
    fig.text(0.5, 0.045,
             "Regiões conjuntas de 68,3% e 95,4% | Modelo quadrático com resíduo nulo no ponto de expansão",
             ha="center", fontsize=9, color="#535b60")
    fig.subplots_adjust(top=0.84, bottom=0.22, left=0.07, right=0.98, wspace=0.17)
    for extensao in ("png", "svg"):
        fig.savefig(destino / f"visao_geral_fisher_dali.{extensao}", dpi=180,
                    facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print("Figura gerada; erro máximo entre referência e dubleto:",
          float(np.max(np.abs(exata - dubleto))))


if __name__ == "__main__":
    main()
