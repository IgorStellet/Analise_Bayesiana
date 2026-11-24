"""Funções principais do projeto Fisher e DALI: Fisher e DALI.

Aqui ficarão as funções principais que possivelmente podem ser reutilizadas em outros projetos
"""

from __future__ import annotations


from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import simpson
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.stats import chi2, norm

FloatArray = NDArray[np.float64]


# Funções de sanidade, para verificar erros bobos que podem acontecer ao longo do código e que ajudam a identificar problemas específicos que podem acontecer para casos mais complexos.

def _vector(values: ArrayLike, name: str) -> FloatArray: 
    "Verifica se os vetores possuem de fato formato de vetores"
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} deve ser um vetor finito não vazio.")
    return result

def _covariance(values: ArrayLike, size: int, name: str) -> FloatArray:
    "Verifica se a matriz de variância está da forma adequada, isto é, se é quadrada, se é simétrica e se é definida positiva"
    result = np.asarray(values, dtype=float)
    if result.shape != (size, size) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} deve ter forma ({size}, {size}) e valores finitos.")
    if not np.allclose(result, result.T, rtol=1e-12, atol=1e-14):
        raise ValueError(f"{name} precisa ser simétrica.")
    try:
        cholesky(result, lower=True, check_finite=False)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} precisa ser definida positiva.") from error
    return result


# ------------------------------------------------------------------------------



@dataclass(frozen=True)
class LinearGaussianResult:
    """Resultado de um modelo Linear com todos os elementos dados
    
    ``fisher`` está relacionado a aproximação da likelihood, ``precision`` inclui
    a prior gaussiana, se fornecida. ``mean`` é a média/MAP da posterior
    gaussiana. Sem prior, registra o limite de prior constante; esse
    limite não define uma evidência para podermos comparar modelos. covarience é a matriz     de convariânca/inverso da matriz de fisher
    """
    mle: FloatArray
    fisher: FloatArray
    mean: FloatArray
    precision: FloatArray
    covariance: FloatArray


def gaussian_logpdf(points: ArrayLike, mean: ArrayLike, covariance: ArrayLike) -> FloatArray:
    """Log da densidade gaussiana multivariada, com normalização.

    ``points`` tem forma ``(..., dimension)``; a última dimensão contém
    as coordenadas de cada ponto. As unidades das entradas precisam ser
    consistentes entre si e com as da covariância. O retorno tem a forma
    das dimensões iniciais de ``points`` (escalar para um único ponto).
    """
    center = _vector(mean, "mean")
    cov = _covariance(covariance, center.size, "covariance")
    values = np.asarray(points, dtype=float)
    if values.ndim < 1 or values.shape[-1] != center.size:
        raise ValueError("A última dimensão de points deve coincidir com mean.")
    if not np.all(np.isfinite(values)):
        raise ValueError("points contém valores não finitos.")
    factor = cholesky(cov, lower=True, check_finite=False)
    delta = (values - center).reshape(-1, center.size)
    whitened = solve_triangular(factor, delta.T, lower=True, check_finite=False).T
    square = np.sum(whitened**2, axis=1)
    log_determinant = 2 * np.log(np.diag(factor)).sum()
    result = -0.5 * (square + center.size * np.log(2 * np.pi) + log_determinant)
    return result.reshape(values.shape[:-1])


def fisher_gaussian(jacobian: ArrayLike, data_covariance: ArrayLike) -> FloatArray:
    """Informação esperada J.T C^{-1} J para C fixa.

    O Jacobiano tem forma ``(n_data, n_parameters)``. A matriz retornada
    pode ser semidefinida positiva e singular; a função não a inverte.
    """
    jac = np.asarray(jacobian, dtype=float)
    if jac.ndim != 2 or not np.all(np.isfinite(jac)):
        raise ValueError("jacobian deve ser uma matriz finita.")
    cov = _covariance(data_covariance, jac.shape[0], "data_covariance")
    result = jac.T @ cho_solve(cho_factor(cov, lower=True), jac)
    return (result + result.T) / 2


def observed_gaussian_hessian(
    residual: ArrayLike,
    jacobian: ArrayLike,
    model_hessians: ArrayLike,
    data_covariance: ArrayLike,
) -> FloatArray:
    """Hessiana observada de -log L para ruído gaussiano com C fixa.

    Usa residual = dados - modelo, J com forma ``(n, p)`` e segundas
    derivadas do modelo com forma ``(n, p, p)``. Não inclui a prior.
    """
    res = _vector(residual, "residual")
    jac = np.asarray(jacobian, dtype=float)
    if jac.ndim != 2 or jac.shape[0] != res.size or not np.all(np.isfinite(jac)):
        raise ValueError("jacobian é incompatível com residual.")
    hessians = np.asarray(model_hessians, dtype=float)
    if hessians.shape != (res.size, jac.shape[1], jac.shape[1]):
        raise ValueError("model_hessians deve ter forma (n_data, n_parameters, n_parameters).")
    if not np.all(np.isfinite(hessians)):
        raise ValueError("model_hessians contém valores não finitos.")
    cov = _covariance(data_covariance, res.size, "data_covariance")
    weighted_residual = cho_solve(cho_factor(cov, lower=True), res)
    result = fisher_gaussian(jac, cov) - np.einsum("i,iab->ab", weighted_residual, hessians)
    return (result + result.T) / 2


def linear_gaussian_posterior(
    design: ArrayLike,
    data: ArrayLike,
    data_covariance: ArrayLike,
    prior_mean: ArrayLike | None = None,
    prior_covariance: ArrayLike | None = None,
) -> LinearGaussianResult:
    """Solução exata do modelo d = X theta + erro gaussiano.

    X tem forma (n, p) e posto completo nas colunas. A covariância dos
    dados é conhecida e fixa. Uma prior gaussiana própria é opcional;
    sua média e covariância precisam ser fornecidas juntas. O ajuste
    não reescala as incertezas por chi-quadrado reduzido.
    """
    observed = _vector(data, "data")
    matrix = np.asarray(design, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != observed.size or not np.all(np.isfinite(matrix)):
        raise ValueError("design deve ser uma matriz finita compatível com data.")
    cov = _covariance(data_covariance, observed.size, "data_covariance")
    factor = cho_factor(cov, lower=True)
    fisher = fisher_gaussian(matrix, cov)
    rhs = matrix.T @ cho_solve(factor, observed)
    try:
        mle = cho_solve(cho_factor(fisher, lower=True), rhs)
    except np.linalg.LinAlgError as error:
        raise ValueError("O modelo linear deve identificar todos os parâmetros: X precisa ter posto completo.") from error
    precision = fisher.copy()
    if (prior_mean is None) != (prior_covariance is None):
        raise ValueError("Forneça prior_mean e prior_covariance juntos.")
    if prior_mean is not None:
        center = _vector(prior_mean, "prior_mean")
        if center.size != matrix.shape[1]:
            raise ValueError("prior_mean tem dimensão incorreta.")
        prior_cov = _covariance(prior_covariance, center.size, "prior_covariance")
        prior_precision = cho_solve(cho_factor(prior_cov, lower=True), np.eye(center.size))
        precision += prior_precision
        rhs = rhs + prior_precision @ center
    posterior_factor = cho_factor(precision, lower=True)
    posterior_mean = cho_solve(posterior_factor, rhs)
    posterior_cov = cho_solve(posterior_factor, np.eye(precision.shape[0]))
    return LinearGaussianResult(mle, fisher, posterior_mean, precision, posterior_cov)


def integrate_grid_2d(values: ArrayLike, x: ArrayLike, y: ArrayLike) -> float:
    """Integra values[y_index, x_index] por Simpson em ambos os eixos."""
    xx = _vector(x, "x")
    yy = _vector(y, "y")
    array = np.asarray(values, dtype=float)
    if min(xx.size, yy.size) < 3 or not np.all(np.diff(xx) > 0) or not np.all(np.diff(yy) > 0):
        raise ValueError("Os eixos devem ser crescentes e conter pelo menos três pontos.")
    if array.shape != (yy.size, xx.size) or not np.all(np.isfinite(array)):
        raise ValueError("values deve ser finito e ter forma (len(y), len(x)).")
    return float(simpson(simpson(array, x=xx, axis=1), x=yy))


def normalize_log_grid(log_density: ArrayLike, x: ArrayLike, y: ArrayLike) -> tuple[FloatArray, float]:
    """Normaliza uma densidade 2D em log; retorna (pdf, log_integral).

    -inf é permitido para representar suporte nulo. O domínio finito
    dos eixos faz parte da integral e deve ter sua extensão validada.
    """
    logs = np.asarray(log_density, dtype=float)
    if np.any(np.isnan(logs)) or np.any(np.isposinf(logs)) or not np.any(np.isfinite(logs)):
        raise ValueError("A densidade em log deve conter massa finita e não pode conter NaN ou +inf.")
    shift = float(np.max(logs))
    unnormalized = np.exp(logs - shift)
    integral = integrate_grid_2d(unnormalized, x, y)
    if integral <= 0 or not np.isfinite(integral):
        raise ValueError("A integral da densidade deve ser finita e positiva.")
    return unnormalized / integral, float(shift + np.log(integral))


def grid_moments_2d(pdf: ArrayLike, x: ArrayLike, y: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Média e covariância na malha 2D; a massa integrada é conferida.

    A média e os momentos são divididos pela massa numericamente
    integrada. Isso não corrige uma escolha insuficiente de domínio.
    """
    xx, yy = np.meshgrid(x, y)
    density = np.asarray(pdf, dtype=float)
    if np.any(density < 0):
        raise ValueError("pdf não pode conter densidades negativas.")
    mass = integrate_grid_2d(density, x, y)
    if mass <= 0:
        raise ValueError("pdf precisa ter massa positiva.")
    mean = np.array([integrate_grid_2d(grid * density, x, y) / mass for grid in (xx, yy)])
    delta = (xx - mean[0], yy - mean[1])
    covariance = np.array([
        [integrate_grid_2d(a * b * density, x, y) / mass for b in delta]
        for a in delta
    ])
    return mean, covariance


def hpd_thresholds_2d(
    pdf: ArrayLike, x: ArrayLike, y: ArrayLike,
    probabilities: ArrayLike = (0.683, 0.954),
) -> FloatArray:
    """Limiar de densidade por conteúdo de probabilidade, em ordem de probabilities.

    Os eixos devem ser regulares, crescentes e ter número ímpar de nós.
    As massas usam os pesos positivos da mesma quadratura de Simpson.
    A fronteira e a interpolação dos contornos permanecem discretizadas.
    """
    axes = [_vector(axis, name) for axis, name in ((x, "x"), (y, "y"))]
    weights = []
    for axis in axes:
        steps = np.diff(axis)
        if axis.size < 3 or axis.size % 2 == 0 or np.any(steps <= 0) or not np.allclose(steps, steps[0]):
            raise ValueError("HPD requer eixos regulares com número ímpar de nós.")
        # Integrar cada vetor da base canônica fornece os pesos da quadratura.
        weights.append(simpson(np.eye(axis.size), x=axis, axis=1))
    density = np.asarray(pdf, dtype=float)
    if density.shape != (axes[1].size, axes[0].size) or not np.all(np.isfinite(density)) or np.any(density < 0):
        raise ValueError("pdf deve ser finito, não negativo e compatível com os eixos.")
    levels = _vector(probabilities, "probabilities")
    if np.any((levels <= 0) | (levels >= 1)):
        raise ValueError("As probabilidades devem estar entre zero e um.")
    flat = density.ravel()
    masses = (density * weights[1][:, None] * weights[0][None, :]).ravel()
    total = masses.sum()
    if total <= 0:
        raise ValueError("pdf precisa ter massa positiva.")
    order = np.argsort(flat)[::-1]
    cumulative = np.cumsum(masses[order]) / total
    indices = np.minimum(np.searchsorted(cumulative, levels), order.size - 1)
    return flat[order[indices]]


def gaussian_ellipse(
    mean: ArrayLike, covariance: ArrayLike, probability: float = 0.683, n_points: int = 400,
) -> FloatArray:
    """Pontos da elipse gaussiana 2D com conteúdo conjunto probability."""
    center = _vector(mean, "mean")
    if center.size != 2 or not 0 < probability < 1 or n_points < 4:
        raise ValueError("Use dois parâmetros, probabilidade em (0, 1) e ao menos quatro pontos.")
    cov = _covariance(covariance, 2, "covariance")
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.linspace(0, 2 * np.pi, n_points)
    circle = np.stack((np.cos(angle), np.sin(angle)))
    scales = np.sqrt(chi2.ppf(probability, df=2) * eigenvalues)
    return center + (eigenvectors @ (scales[:, None] * circle)).T


def lower_truncated_gaussian_moments(
    mean: ArrayLike, covariance: ArrayLike, index: int, lower: float = 0.0,
) -> tuple[FloatArray, FloatArray, float]:
    """Momentos de N(mean, covariance) condicionada a theta[index] >= lower.

    Retorna média, covariância e massa aceita pela distribuição original.
    A fórmula é analítica; sua precisão em caudas extremas ainda depende
    de aritmética finita. Os exemplos do bloco não usam caudas extremas.
    """
    center = _vector(mean, "mean")
    cov = _covariance(covariance, center.size, "covariance")
    if not isinstance(index, (int, np.integer)) or not 0 <= index < center.size or not np.isfinite(lower):
        raise ValueError("index ou lower inválido.")
    sigma = np.sqrt(cov[index, index])
    alpha = (lower - center[index]) / sigma
    log_mass = norm.logsf(alpha)
    ratio = np.exp(norm.logpdf(alpha) - log_mass)
    column = cov[:, index]
    shifted_mean = center + column / sigma * ratio
    delta = ratio * (ratio - alpha)
    shifted_cov = cov - np.outer(column, column) / sigma**2 * delta
    return shifted_mean, shifted_cov, float(np.exp(log_mass))
