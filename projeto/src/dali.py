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
    # Converte a entrada para números reais antes de conferir sua forma.
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} deve ser um vetor finito não vazio.")
    return result

def _covariance(values: ArrayLike, size: int, name: str) -> FloatArray:
    "Verifica se a matriz de covariância está da forma adequada, isto é, se é quadrada, se é simétrica e se é definida positiva"
    # Converte a entrada para números reais antes de conferir sua forma.
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



# Resultados do ajuste linear
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class LinearGaussianResult:
    """Resultado de um modelo Linear com todos os elementos dados
    
    ``fisher`` está relacionado a aproximação da likelihood, ``precision`` inclui
    a prior gaussiana, se fornecida. ``mean`` é a média/MAP da posterior
    gaussiana. Sem prior, registra o limite de prior constante; Covariance é a covariância posterior: a inversa de precision, que inclui a prior
    """
    mle: FloatArray
    fisher: FloatArray
    mean: FloatArray
    precision: FloatArray
    covariance: FloatArray


# Likelihood gaussiana e informação de Fisher
# ------------------------------------------------------------------------------

def gaussian_logpdf(points: ArrayLike, mean: ArrayLike, covariance: ArrayLike) -> FloatArray:
    """Log da densidade gaussiana multivariada, com normalização.

    ``points`` tem forma ``(..., dimension)``; a última dimensão contém
    as coordenadas de cada ponto. As unidades das entradas precisam ser
    consistentes entre si e com as da covariância. O retorno tem a forma
    das dimensões iniciais de ``points`` (escalar para um único ponto).
    """
    # Converte e confere o vetor central da distribuição.
    center = _vector(mean, "mean")
    # Confere dimensões, simetria e positividade da covariância.
    cov = _covariance(covariance, center.size, "covariance")
    # Converte os pontos recebidos para um array numérico.
    values = np.asarray(points, dtype=float)
    if values.ndim < 1 or values.shape[-1] != center.size:
        raise ValueError("A última dimensão de points deve coincidir com mean.")
    if not np.all(np.isfinite(values)):
        raise ValueError("points contém valores não finitos.")
    # Fatora a covariância para resolver sistemas sem formar sua inversa.
    factor = cholesky(cov, lower=True, check_finite=False)
    # Mede os deslocamentos em relação ao centro.
    delta = (values - center).reshape(-1, center.size)
    # Aplica L^{-1} para ponderar os resíduos e incorporar suas correlações.
    whitened = solve_triangular(factor, delta.T, lower=True, check_finite=False).T
    # Soma os quadrados no espaço dos resíduos já ponderados.
    square = np.sum(whitened**2, axis=1)
    # Calcula log|C| usando a diagonal da fatoração de Cholesky.
    log_determinant = 2 * np.log(np.diag(factor)).sum()
    # Junta a distância quadrática e os termos de normalização.
    result = -0.5 * (square + center.size * np.log(2 * np.pi) + log_determinant)
    return result.reshape(values.shape[:-1])


def fisher_gaussian(jacobian: ArrayLike, data_covariance: ArrayLike) -> FloatArray:
    """Informação esperada J.T C^{-1} J para C fixa.

    O Jacobiano tem forma ``(n_data, n_parameters)``. A matriz retornada
    pode ser semidefinida positiva e singular;
    """
    # Converte o Jacobiano: dados nas linhas, parâmetros nas colunas.
    jac = np.asarray(jacobian, dtype=float)
    if jac.ndim != 2 or not np.all(np.isfinite(jac)):
        raise ValueError("jacobian deve ser uma matriz finita.")
    # Confere dimensões, simetria e positividade da covariância.
    cov = _covariance(data_covariance, jac.shape[0], "data_covariance")
    # Calcula a informação J.T C^{-1} J sem inverter C explicitamente.
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
    # Confere o vetor de resíduos, definido como dados menos modelo.
    res = _vector(residual, "residual")
    # Converte o Jacobiano: dados nas linhas, parâmetros nas colunas.
    jac = np.asarray(jacobian, dtype=float)
    if jac.ndim != 2 or jac.shape[0] != res.size or not np.all(np.isfinite(jac)):
        raise ValueError("jacobian é incompatível com residual.")
    # Recebe as segundas derivadas de cada previsão em relação aos parâmetros.
    hessians = np.asarray(model_hessians, dtype=float)
    if hessians.shape != (res.size, jac.shape[1], jac.shape[1]):
        raise ValueError("model_hessians deve ter forma (n_data, n_parameters, n_parameters).")
    if not np.all(np.isfinite(hessians)):
        raise ValueError("model_hessians contém valores não finitos.")
    # Confere dimensões, simetria e positividade da covariância.
    cov = _covariance(data_covariance, res.size, "data_covariance")
    # Resolve C x = resíduo para calcular o termo observado da curvatura.
    weighted_residual = cho_solve(cho_factor(cov, lower=True), res)
    # Subtrai de Fisher o termo que depende do resíduo observado.
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
    # Confere e guarda o vetor de dados do ajuste.
    observed = _vector(data, "data")
    # Converte a matriz X que relaciona parâmetros e previsões.
    matrix = np.asarray(design, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != observed.size or not np.all(np.isfinite(matrix)):
        raise ValueError("design deve ser uma matriz finita compatível com data.")
    # Confere dimensões, simetria e positividade da covariância.
    cov = _covariance(data_covariance, observed.size, "data_covariance")
    # Fatora a covariância para resolver sistemas sem formar sua inversa.
    factor = cho_factor(cov, lower=True)
    # Calcula J.T C^{-1} J com o Jacobiano do modelo linear.
    fisher = fisher_gaussian(matrix, cov)
    # Monta o lado direito das equações normais ou acrescenta a informação da prior.
    rhs = matrix.T @ cho_solve(factor, observed)
    try:
        # Resolve as equações normais para obter o máximo da likelihood.
        mle = cho_solve(cho_factor(fisher, lower=True), rhs)
    except np.linalg.LinAlgError as error:
        raise ValueError("O modelo linear deve identificar todos os parâmetros: X precisa ter posto completo.") from error
    # Começa pela precisão da likelihood; a prior será somada abaixo.
    precision = fisher.copy()
    if (prior_mean is None) != (prior_covariance is None):
        raise ValueError("Forneça prior_mean e prior_covariance juntos.")
    if prior_mean is not None:
        # Converte e confere o vetor central da distribuição.
        center = _vector(prior_mean, "prior_mean")
        if center.size != matrix.shape[1]:
            raise ValueError("prior_mean tem dimensão incorreta.")
        # Confere a covariância da prior gaussiana.
        prior_cov = _covariance(prior_covariance, center.size, "prior_covariance")
        # Resolve o sistema que fornece a precisão da prior.
        prior_precision = cho_solve(cho_factor(prior_cov, lower=True), np.eye(center.size))
        precision += prior_precision
        # Monta o lado direito das equações normais ou acrescenta a informação da prior.
        rhs = rhs + prior_precision @ center
    # Fatora a precisão total, já incluindo a prior.
    posterior_factor = cho_factor(precision, lower=True)
    # Resolve o sistema para obter a média posterior.
    posterior_mean = cho_solve(posterior_factor, rhs)
    # Obtém a covariância posterior resolvendo contra a identidade.
    posterior_cov = cho_solve(posterior_factor, np.eye(precision.shape[0]))
    return LinearGaussianResult(mle, fisher, posterior_mean, precision, posterior_cov)


# Integração em malhas e momentos da posterior
# ------------------------------------------------------------------------------

def integrate_grid_2d(values: ArrayLike, x: ArrayLike, y: ArrayLike) -> float:
    """Integra values[y_index, x_index] por Simpson em ambos os eixos."""
    # Confere o eixo horizontal ou monta suas coordenadas na malha.
    xx = _vector(x, "x")
    # Confere o eixo vertical.
    yy = _vector(y, "y")
    # Converte os valores tabulados que serão integrados.
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
    # Recebe a densidade em log, permitindo -inf em regiões excluídas.
    logs = np.asarray(log_density, dtype=float)
    if np.any(np.isnan(logs)) or np.any(np.isposinf(logs)) or not np.any(np.isfinite(logs)):
        raise ValueError("A densidade em log deve conter massa finita e não pode conter NaN ou +inf.")
    # Escolhe um deslocamento que evita exponenciais numericamente muito pequenas.
    shift = float(np.max(logs))
    # Exponencia a densidade depois de retirar o deslocamento.
    unnormalized = np.exp(logs - shift)
    # Integra a densidade deslocada com as coordenadas dos eixos.
    integral = integrate_grid_2d(unnormalized, x, y)
    if integral <= 0 or not np.isfinite(integral):
        raise ValueError("A integral da densidade deve ser finita e positiva.")
    return unnormalized / integral, float(shift + np.log(integral))


def grid_moments_2d(pdf: ArrayLike, x: ArrayLike, y: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Média e covariância na malha 2D; a massa integrada é conferida.

    A média e os momentos são divididos pela massa numericamente
    integrada. Isso não corrige uma escolha insuficiente de domínio.
    """
    # Confere o eixo horizontal ou monta suas coordenadas na malha.
    xx, yy = np.meshgrid(x, y)
    # Recebe a densidade já tabulada na malha.
    density = np.asarray(pdf, dtype=float)
    if np.any(density < 0):
        raise ValueError("pdf não pode conter densidades negativas.")
    # Mede a massa total antes de calcular os momentos.
    mass = integrate_grid_2d(density, x, y)
    if mass <= 0:
        raise ValueError("pdf precisa ter massa positiva.")
    # Integra as coordenadas para obter a média.
    mean = np.array([integrate_grid_2d(grid * density, x, y) / mass for grid in (xx, yy)])
    # Mede os deslocamentos em relação ao centro.
    delta = (xx - mean[0], yy - mean[1])
    # Integra os produtos dos deslocamentos para obter a covariância.
    covariance = np.array([
        [integrate_grid_2d(a * b * density, x, y) / mass for b in delta]
        for a in delta
    ])
    return mean, covariance


# Regiões de credibilidade e elipses
# ------------------------------------------------------------------------------

def hpd_thresholds_2d(
    pdf: ArrayLike, x: ArrayLike, y: ArrayLike,
    probabilities: ArrayLike = (0.683, 0.954),
) -> FloatArray:
    """Limiar de densidade por conteúdo de probabilidade, em ordem de probabilities.

    Os eixos devem ser regulares, crescentes e ter número ímpar de nós.
    As massas usam os pesos positivos da mesma quadratura de Simpson.
    A fronteira e a interpolação dos contornos permanecem discretizadas.
    """
    # Confere os dois eixos usados para localizar os níveis HPD.
    axes = [_vector(axis, name) for axis, name in ((x, "x"), (y, "y"))]
    # Guarda os pesos de quadratura de cada eixo.
    weights = []
    for axis in axes:
        # Calcula os espaçamentos para conferir a regularidade da malha.
        steps = np.diff(axis)
        if axis.size < 3 or axis.size % 2 == 0 or np.any(steps <= 0) or not np.allclose(steps, steps[0]):
            raise ValueError("HPD requer eixos regulares com número ímpar de nós.")
        # Integrar cada vetor da base canônica fornece os pesos da quadratura.
        weights.append(simpson(np.eye(axis.size), x=axis, axis=1))
    # Recebe a densidade já tabulada na malha.
    density = np.asarray(pdf, dtype=float)
    if density.shape != (axes[1].size, axes[0].size) or not np.all(np.isfinite(density)) or np.any(density < 0):
        raise ValueError("pdf deve ser finito, não negativo e compatível com os eixos.")
    # Confere as probabilidades desejadas para as regiões HPD.
    levels = _vector(probabilities, "probabilities")
    if np.any((levels <= 0) | (levels >= 1)):
        raise ValueError("As probabilidades devem estar entre zero e um.")
    # Organiza os nós num único vetor para ordená-los por densidade.
    flat = density.ravel()
    # Multiplica densidades pelos pesos da quadratura para obter massas.
    masses = (density * weights[1][:, None] * weights[0][None, :]).ravel()
    # Soma as massas da malha.
    total = masses.sum()
    if total <= 0:
        raise ValueError("pdf precisa ter massa positiva.")
    # Ordena os nós da maior para a menor densidade.
    order = np.argsort(flat)[::-1]
    # Acumula a probabilidade nessa ordem para encontrar cada limiar.
    cumulative = np.cumsum(masses[order]) / total
    # Localiza o primeiro nó que atinge cada probabilidade solicitada.
    indices = np.minimum(np.searchsorted(cumulative, levels), order.size - 1)
    return flat[order[indices]]


def gaussian_ellipse(
    mean: ArrayLike, covariance: ArrayLike, probability: float = 0.683, n_points: int = 400,
) -> FloatArray:
    """Pontos da elipse gaussiana 2D com conteúdo conjunto probability."""
    # Converte e confere o vetor central da distribuição.
    center = _vector(mean, "mean")
    if center.size != 2 or not 0 < probability < 1 or n_points < 4:
        raise ValueError("Use dois parâmetros, probabilidade em (0, 1) e ao menos quatro pontos.")
    # Confere dimensões, simetria e positividade da covariância.
    cov = _covariance(covariance, 2, "covariance")
    # Obtém as variâncias e direções principais da covariância.
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Percorre um círculo unitário que será transformado numa elipse.
    angle = np.linspace(0, 2 * np.pi, n_points)
    # Escreve as coordenadas do círculo unitário.
    circle = np.stack((np.cos(angle), np.sin(angle)))
    # Ajusta os semieixos para a probabilidade conjunta escolhida.
    scales = np.sqrt(chi2.ppf(probability, df=2) * eigenvalues)
    return center + (eigenvectors @ (scales[:, None] * circle)).T


# Fronteiras: momentos de uma gaussiana truncada
# ------------------------------------------------------------------------------

def lower_truncated_gaussian_moments(
    mean: ArrayLike, covariance: ArrayLike, index: int, lower: float = 0.0,
) -> tuple[FloatArray, FloatArray, float]:
    """Momentos de N(mean, covariance) condicionada a theta[index] >= lower.

    Retorna média, covariância e massa aceita pela distribuição original.
    A fórmula é analítica; sua precisão em caudas extremas ainda depende
    de aritmética finita. Os exemplos do bloco não usam caudas extremas.
    """
    # Converte e confere o vetor central da distribuição.
    center = _vector(mean, "mean")
    # Confere dimensões, simetria e positividade da covariância.
    cov = _covariance(covariance, center.size, "covariance")
    if not isinstance(index, (int, np.integer)) or not 0 <= index < center.size or not np.isfinite(lower):
        raise ValueError("index ou lower inválido.")
    # Extrai o desvio marginal do parâmetro que receberá o corte.
    sigma = np.sqrt(cov[index, index])
    # Expressa a fronteira em desvios da gaussiana original.
    alpha = (lower - center[index]) / sigma
    # Calcula a massa da cauda permitida em log para maior estabilidade.
    log_mass = norm.logsf(alpha)
    # Calcula a razão de Mills usada nos momentos da gaussiana truncada.
    ratio = np.exp(norm.logpdf(alpha) - log_mass)
    # Seleciona as covariâncias com o parâmetro que recebeu o corte.
    column = cov[:, index]
    # Desloca a média de todos os parâmetros segundo suas correlações.
    shifted_mean = center + column / sigma * ratio
    # Calcula o fator que reduz a covariância ao remover a cauda.
    delta = ratio * (ratio - alpha)
    # Aplica a correção analítica da covariância depois do corte.
    shifted_cov = cov - np.outer(column, column) / sigma**2 * delta
    return shifted_mean, shifted_cov, float(np.exp(log_mass))


# ------------------------------------------------------------------------------
# Hierarquia de DALI: derivadas do modelo, quadrado completo e likelihood
# ------------------------------------------------------------------------------

def _model_derivatives(derivatives: tuple[ArrayLike, ...], n_parameters: int) -> tuple[FloatArray, ...]:
    """Confere se J, H, T e U têm os eixos dos dados e dos parâmetros na ordem certa.

    Cada tensor tem forma (n_dados, p, ..., p), com k eixos de parâmetros
    na derivada de ordem k. Derivadas mistas devem ser simétricas nesses eixos.
    Forneça derivadas brutas: os fatores 1/2!, 1/3! e 1/4! entram na avaliação.
    """
    if not 1 <= len(derivatives) <= 4:  # A quarta ordem será usada no contraexemplo do Bloco 3.
        raise ValueError("Forneça de uma a quatro derivadas consecutivas: J, H, T, U.")
    first = np.asarray(derivatives[0], dtype=float)  # A primeira derivada é o Jacobiano J.
    if first.ndim != 2 or first.shape[0] == 0:  # O primeiro eixo enumera os dados.
        raise ValueError("J deve ter forma (n_dados, n_parametros), com n_dados > 0.")
    n_data = first.shape[0]  # Todas as derivadas devem prever os mesmos dados.
    checked = []  # Guarda cópias dos tensores conferidos.
    for k, values in enumerate(derivatives, start=1):  # k é a ordem da derivada.
        tensor = np.array(values, dtype=float, copy=True)  # Evita alterar os arrays recebidos.
        expected_shape = (n_data,) + (n_parameters,) * k  # J: (N,p); H: (N,p,p); T: (N,p,p,p).
        if tensor.shape != expected_shape or not np.all(np.isfinite(tensor)):
            raise ValueError(f"A derivada de ordem {k} deve ser finita e ter forma {expected_shape}.")
        for axis in range(1, k):  # Trocas adjacentes conferem a simetria entre parâmetros.
            if not np.allclose(tensor, np.swapaxes(tensor, axis, axis+1), rtol=1e-10, atol=1e-12):
                raise ValueError(f"A derivada de ordem {k} não é simétrica nos eixos dos parâmetros.")
        checked.append(tensor)  # Mantém a ordem J, H, T, U.
    return tuple(checked)


def _taylor_shift(delta: FloatArray, derivatives: tuple[FloatArray, ...], order: int) -> FloatArray:
    """Soma as respostas do modelo até a ordem escolhida, com os fatores 1/k!.

    Esta função interna recebe formas já conferidas. O einsum soma os
    índices dos parâmetros e mantém o índice dos dados. As reticências
    permitem avaliar um ponto, uma lista de pontos ou uma malha inteira.
    """
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)):
        raise ValueError("order deve ser um inteiro: 1, 2, 3 ou 4.")
    if not 1 <= order <= len(derivatives):  # Não podemos usar derivadas que não foram fornecidas.
        raise ValueError(f"order deve estar entre 1 e {len(derivatives)}.")
    shift = np.einsum("ia,...a->...i", derivatives[0], delta)  # Resposta linear J.delta.
    if order >= 2:  # O dubleto acrescenta a curvatura do modelo.
        shift += 0.5*np.einsum("iab,...a,...b->...i", derivatives[1], delta, delta)
    if order >= 3:  # O tripleto acrescenta a terceira derivada do modelo.
        shift += np.einsum("iabc,...a,...b,...c->...i", derivatives[2], delta, delta, delta)/6.0
    if order >= 4:  # A quarta derivada acrescenta a resposta quártica do modelo.
        shift += np.einsum("iabcd,...a,...b,...c,...d->...i", derivatives[3],
                           delta, delta, delta, delta)/24.0
    return shift


@dataclass(frozen=True)
class DALIExpansion:
    """Guarda a expansão preparada para avaliar até quatro ordens sem refazer Cholesky.

    Construa este objeto com prepare_dali. Os tensores e o resíduo já
    receberam L^{-1}, onde C = L L.T. Isso incorpora as correlações dos
    dados e transforma a distância de Mahalanobis numa soma de quadrados.
    Os métodos recebem parâmetros absolutos; o centro é subtraído aqui.
    """

    center: FloatArray  # Ponto theta_0 em que as derivadas foram calculadas.
    whitened_derivatives: tuple[FloatArray, ...]  # J, H, T e U após aplicar L^{-1} no eixo dos dados.
    whitened_residual: FloatArray  # L^{-1}(dados - modelo no centro).
    log_data_normalization: float  # -N/2 log(2*pi) - log|C|/2; não normaliza em parâmetros.

    def exponent(self, points: ArrayLike, order: int = 2) -> FloatArray:
        """Retorna E_K = ||L^{-1}(r_0 - T_K)||²/2, que é não negativo.

        points tem forma (..., p). order=1 usa J; order=2 usa J,H;
        order=3 usa J,H,T; order=4 usa também U. A prior fica fora deste quadrado.
        """
        values = np.asarray(points, dtype=float)  # Aceita listas ou arrays de parâmetros.
        if values.ndim < 1 or values.shape[-1] != self.center.size or not np.all(np.isfinite(values)):
            raise ValueError("points deve ser finito e sua última dimensão deve coincidir com center.")
        delta = values-self.center  # Mede o deslocamento em relação ao ponto da expansão.
        shift = _taylor_shift(delta, self.whitened_derivatives, order)  # Aproxima a mudança da previsão.
        residual = self.whitened_residual-shift  # Mantém o resíduo dos dados, inclusive se não for zero.
        return 0.5*np.sum(residual**2, axis=-1)  # Soma sobre os dados e conserva os eixos da malha.

    def loglike(self, points: ArrayLike, order: int = 2) -> FloatArray:
        """Avalia log L_K com a normalização gaussiana no espaço dos dados.

        O retorno é uma likelihood, não uma posterior normalizada.
        Some a log-prior e integre/amostre para obter a posterior.
        """
        return self.log_data_normalization-self.exponent(points, order=order)


def prepare_dali(
    center: ArrayLike,
    derivatives: tuple[ArrayLike, ...],
    data_covariance: ArrayLike,
    residual: ArrayLike | None = None,
) -> DALIExpansion:
    """Prepara DALI com C fixa e derivadas avaliadas no mesmo centro.

    center é theta_0; derivatives contém (J,), (J,H), (J,H,T) ou (J,H,T,U), sem
    fatores de Taylor. residual é dados-mu(theta_0). Se omitido, usa
    resíduo zero: o forecast com dados iguais à previsão fiducial.

    C deve ser conhecida, simétrica, definida positiva e independente
    dos parâmetros. Não é necessário inverter Fisher, nem exigir que
    J tenha posto completo. Essa liberdade não garante normalização
    em parâmetros ou boa aproximação global; a prior é tratada à parte.
    """
    anchor = _vector(center, "center").copy()  # Guarda uma cópia do ponto escolhido.
    tensors = _model_derivatives(derivatives, anchor.size)  # Confere as formas e a simetria de J,H,T,U.
    n_data = tensors[0].shape[0]  # O primeiro eixo de cada derivada é o dos dados.
    cov = _covariance(data_covariance, n_data, "data_covariance")  # Confere a covariância dos dados.
    res = np.zeros(n_data) if residual is None else _vector(residual, "residual").copy()
    if res.size != n_data:  # O resíduo precisa ter uma entrada por dado.
        raise ValueError("residual deve ter o mesmo número de entradas que o eixo dos dados de J.")
    factor = cholesky(cov, lower=True, check_finite=False)  # Calcula C = L L.T uma única vez.
    whitened = []  # As avaliações posteriores usarão somente esses tensores já ponderados.
    for tensor in tensors:  # Aplica a mesma transformação a cada derivada fornecida.
        flat = tensor.reshape(n_data, -1)  # Junta temporariamente os eixos dos parâmetros.
        transformed = solve_triangular(factor, flat, lower=True, check_finite=False)
        whitened.append(transformed.reshape(tensor.shape))  # Recupera os eixos originais.
    whitened_residual = solve_triangular(factor, res, lower=True, check_finite=False)
    log_det = 2*np.log(np.diag(factor)).sum()  # log|C| a partir da diagonal de Cholesky.
    log_norm = -0.5*(n_data*np.log(2*np.pi)+log_det)  # Normalização da densidade dos dados.
    return DALIExpansion(anchor, tuple(whitened), whitened_residual, float(log_norm))


# ------------------------------------------------------------------------------
# Malhas em uma dimensão: normalização e momentos dos exemplos controlados
# ------------------------------------------------------------------------------

def normalize_log_grid_1d(log_density: ArrayLike, x: ArrayLike) -> tuple[FloatArray, float]:
    """Normaliza uma densidade em log numa malha 1D; devolve (pdf, log_integral).

    O domínio faz parte da integral. Pode representar um suporte físico
    declarado ou um recorte numérico que deve ser testado. Os pontos de x
    podem ter espaçamentos diferentes, como numa mudança não linear de variável.
    """
    axis = _vector(x, "x")  # Confere o eixo e o converte para números reais.
    if axis.size < 3 or not np.all(np.diff(axis) > 0):  # Simpson precisa de um eixo crescente.
        raise ValueError("x deve ser crescente e conter pelo menos três pontos.")
    logs = np.asarray(log_density, dtype=float)  # Recebe a densidade antes da normalização.
    if logs.shape != axis.shape:  # Deve haver uma densidade para cada ponto do eixo.
        raise ValueError("log_density deve ter a mesma forma de x.")
    if np.any(np.isnan(logs)) or np.any(np.isposinf(logs)) or not np.any(np.isfinite(logs)):
        raise ValueError("log_density deve ter massa finita e não pode conter NaN ou +inf.")
    shift = float(np.max(logs))  # Retira o máximo para evitar subfluxo na exponencial.
    unnormalized = np.exp(logs-shift)  # -inf é permitido e representa densidade nula.
    integral = float(simpson(unnormalized, x=axis))  # Integra com as coordenadas reais do eixo.
    if not np.isfinite(integral) or integral <= 0:
        raise ValueError("A integral da densidade deve ser finita e positiva.")
    return unnormalized/integral, float(shift+np.log(integral))


def grid_moments_1d(pdf: ArrayLike, x: ArrayLike) -> tuple[float, float]:
    """Integra média e variância em 1D; não corrige um domínio insuficiente.

    A média e a variância usam a massa integrada, permitindo conferir
    resultados mesmo quando a normalização numérica não é exatamente um.
    """
    axis = _vector(x, "x")  # Confere o vetor de coordenadas.
    density = _vector(pdf, "pdf")  # Confere os valores da densidade.
    if density.shape != axis.shape or np.any(density < 0):
        raise ValueError("pdf deve ser não negativa e ter a mesma forma de x.")
    if axis.size < 3 or not np.all(np.diff(axis) > 0):
        raise ValueError("x deve ser crescente e conter pelo menos três pontos.")
    mass = float(simpson(density, x=axis))  # Verifica a massa representada pela malha.
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("pdf precisa ter massa finita e positiva.")
    mean = float(simpson(axis*density, x=axis)/mass)  # Integra o primeiro momento.
    variance = float(simpson((axis-mean)**2*density, x=axis)/mass)  # Usa o momento central para estabilidade.
    return mean, variance


def grid_tail_mass_1d(pdf: ArrayLike, x: ArrayLike, lower: float) -> float:
    """Integra a massa acima de lower, sem renormalizar a densidade recebida.

    Quando lower coincide com um nó a menos de arredondamento, reutiliza
    esse nó: inserir dois pontos quase iguais tornaria Simpson instável.
    Caso contrário, interpola a densidade apenas na nova fronteira.
    """
    axis = _vector(x, "x")  # Confere as coordenadas de integração.
    density = _vector(pdf, "pdf")  # Confere os valores da densidade.
    if axis.size < 3 or not np.all(np.diff(axis) > 0):
        raise ValueError("x deve ser crescente e conter pelo menos três pontos.")
    if density.shape != axis.shape or np.any(density < 0):
        raise ValueError("pdf deve ser não negativa e ter a mesma forma de x.")
    if not np.isfinite(lower) or not axis[0] <= lower <= axis[-1]:
        raise ValueError("lower deve pertencer ao domínio da malha.")
    index = int(np.searchsorted(axis, lower))  # Localiza o primeiro nó que alcança o limite.
    neighbors = (max(0,index-1), min(index,axis.size-1))  # Considera os vizinhos dos dois lados.
    nearest = min(neighbors, key=lambda i: abs(axis[i]-lower))  # Encontra o nó numericamente mais próximo.
    tolerance = 8*np.finfo(float).eps*max(1.0,abs(lower),abs(axis[nearest]))  # Tolera apenas arredondamento de ponto flutuante.
    if abs(axis[nearest]-lower) <= tolerance:  # Evita criar nós quase duplicados na fronteira.
        tail_x, tail_pdf = axis[nearest:], density[nearest:]
    else:  # Insere o limite quando ele realmente fica entre dois nós distintos.
        tail_x = np.r_[lower,axis[index:]]
        tail_pdf = np.r_[np.interp(lower,axis,density),density[index:]]
    if tail_x.size < 2:  # A massa acima da extremidade superior é zero.
        return 0.0
    return float(simpson(tail_pdf,x=tail_x))  # Integra somente o domínio suave da cauda.


# ------------------------------------------------------------------------------
# Marginalização em três dimensões e momentos da forma das densidades
# ------------------------------------------------------------------------------

def density_moments_1d(pdf: ArrayLike, x: ArrayLike) -> dict[str,float]:
    """Integra média, desvio, assimetria e excesso de curtose de uma densidade.

    São momentos da distribuição tabulada, não estimadores de uma amostra.
    Excesso zero é o valor gaussiano; um valor negativo indica curtose
    menor que a gaussiana, sem implicar proximidade da forma uniforme.
    """
    axis = _vector(x,"x")  # Confere as coordenadas da malha.
    density = _vector(pdf,"pdf")  # Confere a densidade recebida.
    mean,variance = grid_moments_1d(density,axis)  # Reutiliza as verificações da quadratura 1D.
    if variance <= 0:
        raise ValueError("A variância deve ser positiva para padronizar os momentos.")
    mass = float(simpson(density,x=axis))  # Corrige apenas o arredondamento da massa tabulada.
    central3 = float(simpson((axis-mean)**3*density,x=axis)/mass)  # Integra o terceiro momento central.
    central4 = float(simpson((axis-mean)**4*density,x=axis)/mass)  # Integra o quarto momento central.
    return {"mean":mean,"std":float(np.sqrt(variance)),
            "skewness":central3/variance**1.5,"excess_kurtosis":central4/variance**2-3.0}


@dataclass(frozen=True)
class Grid3DResult:
    """Guarda as marginais e os três planos, sem armazenar um cubo 3D.

    pairs[(i,j)] tem eixo j nas linhas e eixo i nas colunas. Todos os
    planos e marginais usam o mesmo normalizador da posterior conjunta.
    log_normalizer é a integral do núcleo recebido, não de outra função.
    """
    axes: tuple[FloatArray,FloatArray,FloatArray]
    marginals: tuple[FloatArray,FloatArray,FloatArray]
    pairs: dict[tuple[int,int],FloatArray]
    log_normalizer: float


def _regular_simpson_weights(axis: FloatArray) -> FloatArray:
    """Monta os pesos positivos de Simpson num eixo regular com número ímpar de nós."""
    steps = np.diff(axis)  # Mede o espaçamento antes de aplicar a regra regular.
    if axis.size < 3 or axis.size%2 == 0 or np.any(steps<=0) or not np.allclose(steps,steps[0]):
        raise ValueError("Cada eixo deve ser crescente, regular e ter número ímpar de nós.")
    weights = np.ones(axis.size)  # As extremidades recebem peso um.
    weights[1:-1:2] = 4  # Nós interiores de índice ímpar recebem peso quatro.
    weights[2:-1:2] = 2  # Os demais nós interiores recebem peso dois.
    return weights*(steps[0]/3)  # Inclui a largura das células na quadratura.


def marginalize_grid_3d(log_density, axes: tuple[ArrayLike,ArrayLike,ArrayLike],
                        log_shift: float = 0.0) -> Grid3DResult:
    """Integra uma densidade 3D por planos do terceiro parâmetro.

    log_density recebe points de forma (n_y,n_x,3) e retorna log q na
    forma (n_y,n_x). O terceiro eixo é percorrido plano a plano. O uso
    de memória cresce com os três planos 2D, não com n_x*n_y*n_w.
    log_shift é uma constante de estabilização; escolha um valor próximo
    ao log do pico. Os eixos são regulares e têm número ímpar de nós.
    """
    if len(axes) != 3 or not np.isfinite(log_shift):
        raise ValueError("Forneça três eixos e um log_shift finito.")
    xx,yy,ww = tuple(_vector(axis,f"axis_{i}").copy() for i,axis in enumerate(axes))
    wx,wy,wwgt = tuple(_regular_simpson_weights(axis) for axis in (xx,yy,ww))  # Calcula os pesos nos três eixos.
    qxy = np.zeros((yy.size,xx.size))  # Acumula a integral sobre o terceiro parâmetro.
    qxw = np.zeros((ww.size,xx.size))  # Guarda a integral sobre o segundo parâmetro.
    qyw = np.zeros((ww.size,yy.size))  # Guarda a integral sobre o primeiro parâmetro.
    points = np.empty((yy.size,xx.size,3))  # Reserva apenas um plano de coordenadas.
    points[...,0],points[...,1] = np.meshgrid(xx,yy)  # Mantém fixas as coordenadas dos dois primeiros eixos.
    for k,value in enumerate(ww):  # Percorre o terceiro eixo sem construir um cubo inteiro.
        points[...,2] = value  # Define o valor do parâmetro do plano atual.
        logs = np.asarray(log_density(points),dtype=float)  # Avalia o log do núcleo nesse plano.
        if logs.shape != qxy.shape or np.any(np.isnan(logs)) or np.any(np.isposinf(logs)):
            raise ValueError("log_density deve retornar um plano sem NaN ou +inf.")
        with np.errstate(over='ignore'):
            plane = np.exp(logs-log_shift)  # Retira a mesma constante em todos os planos.
        if not np.all(np.isfinite(plane)):
            raise ValueError("A exponencial excedeu a precisão; escolha log_shift próximo ao pico.")
        qxy += wwgt[k]*plane  # Integra sobre w com seus pesos de Simpson.
        qxw[k] = np.einsum('i,ij->j',wy,plane)  # Integra y e mantém x no plano xw.
        qyw[k] = np.einsum('ij,j->i',plane,wx)  # Integra x e mantém y no plano yw.
    total = float(np.einsum('i,ij,j->',wy,qxy,wx))  # Obtém um único normalizador conjunto.
    if total <= 0 or not np.isfinite(total):
        raise ValueError("O núcleo precisa ter massa finita e positiva no domínio.")
    pairs = {(0,1):qxy/total,(0,2):qxw/total,(1,2):qyw/total}  # Normaliza todos os planos pela mesma integral.
    marginals = (np.einsum('i,ij->j',wy,qxy)/total,  # Integra y para obter p(x).
                 np.einsum('ij,j->i',qxy,wx)/total,  # Integra x para obter p(y).
                 np.einsum('ij,j->i',qxw,wx)/total)  # Integra x no plano xw para obter p(w).
    return Grid3DResult((xx,yy,ww),marginals,pairs,float(log_shift+np.log(total)))
