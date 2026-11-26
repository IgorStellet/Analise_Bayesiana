"""Supernovas em wCDM plano: distâncias, calibração e derivadas até ordem quatro.

O catálogo permanece separado deste módulo. A análise usa os módulos de
distância já padronizados pela equipe Union2.1, sem reajustar curvas de luz.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import factorial
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import least_squares


# Leitura do catálogo e organização das coordenadas
# -----------------------------------------------------------------------------

def load_union21(path: str | Path) -> dict:
    """Lê as cinco colunas oficiais, sem ordenar nem cortar as supernovas."""
    rows = np.genfromtxt(path, dtype=str)  # Conserva o identificador de cada supernova.
    if rows.shape != (580, 5):  # Confere a versão do catálogo usada nesta reprodução.
        raise ValueError('Esperava as 580 linhas e cinco colunas da tabela Union2.1.')
    numeric = rows[:, 1:].astype(float)  # Converte somente redshift, módulo, erro e probabilidade.
    if not np.all(np.isfinite(numeric)) or np.any(numeric[:, [0, 2]] <= 0):
        raise ValueError('Redshifts e erros devem ser positivos e finitos.')
    return dict(name=rows[:, 0], z=numeric[:, 0], modulus=numeric[:, 1],
                sigma=numeric[:, 2], p_low_mass=numeric[:, 3])  # Não interpreta a quinta coluna como peso da análise.


def coordinates(theta: ArrayLike, chart: str = 'physical') -> np.ndarray:
    """Converte (Omega_m,w) às coordenadas usadas para expandir o modelo.

    A densidade será sempre comparada na medida física dOmega_m dw.
    Aqui só mudamos os argumentos do modelo, não a prior física.
    """
    values = np.array(theta, dtype=float, copy=True)  # Não modifica os parâmetros recebidos.
    if values.shape[-1] != 2:
        raise ValueError('theta deve terminar nos dois parâmetros Omega_m,w.')
    if chart == 'physical':
        return values
    if chart not in ('logw', 'logboth') or np.any(values[..., 1] >= 0):
        raise ValueError('Use physical, logw ou logboth; os mapas logarítmicos exigem w<0.')
    values[..., 1] = np.log(-values[..., 1])  # Evita atravessar w=0 nas coordenadas da expansão.
    if chart == 'logboth':
        if np.any(values[..., 0] <= 0):
            raise ValueError('logboth exige Omega_m>0.')
        values[..., 0] = np.log(values[..., 0])  # Representa a densidade positiva por seu logaritmo.
    return values


def _jet_product(left: dict, right: dict, order: int) -> dict:
    """Multiplica séries bivariadas e conserva somente o grau total pedido."""
    result = {}  # Cada chave (a,b) representa o coeficiente de du**a dv**b.
    for (a, b), first in left.items():
        for (c, d), second in right.items():
            if a + b + c + d <= order:  # Descarta apenas termos incapazes de contribuir à ordem final.
                key = (a + c, b + d)
                result[key] = result.get(key, 0.) + first * second  # Convolui os coeficientes das duas séries.
    return result


@dataclass
class FlatWCDM:
    """Modelo plano com matéria sem pressão e energia escura de w constante.

    Não inclui radiação, curvatura nem evolução de w. A previsão é
    mu=5 log10(dhat_L), com dhat_L=H0 d_L/c; a constante de calibração
    é tratada separadamente. A quadratura é Gauss–Legendre em [0,z].
    """
    z: ArrayLike
    quadrature_order: int = 48

    def __post_init__(self):
        self.z = np.asarray(self.z, dtype=float)  # Guarda os redshifts na ordem do catálogo.
        if self.z.ndim != 1 or np.any(self.z <= 0) or not np.all(np.isfinite(self.z)):
            raise ValueError('z deve ser um vetor de redshifts positivos e finitos.')
        nodes, weights = np.polynomial.legendre.leggauss(self.quadrature_order)  # Obtém nós e pesos de quadratura.
        self.nodes = self.z[:, None] * (nodes + 1) / 2  # Mapeia cada intervalo [-1,1] para [0,z_i].
        self.weights = self.z[:, None] * weights / 2  # Inclui o jacobiano dessa mudança de variável.
        self.log1pz = np.log1p(self.nodes)  # Reutiliza o logaritmo nas avaliações do modelo.
        self.matter = (1 + self.nodes) ** 3  # Evolução da densidade de matéria sem pressão.

    def predict(self, theta: ArrayLike) -> np.ndarray:
        """Calcula mu(theta) para um ponto ou uma lista de pontos cosmológicos."""
        points = np.asarray(theta, dtype=float)  # Aceita (...,2), com o eixo dos dados no retorno.
        if points.ndim < 1 or points.shape[-1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError('theta deve ser finito e terminar nos parâmetros Omega_m,w.')
        omega = points[..., 0, None, None]  # Prepara a densidade de matéria para percorrer os redshifts.
        w = points[..., 1, None, None]  # Prepara o parâmetro da equação de estado.
        e2 = omega * self.matter + (1 - omega) * np.exp(3 * (1 + w) * self.log1pz)  # Calcula H(z)^2/H0^2.
        if np.any(e2 <= 0):
            raise ValueError('A expansão escolhida produz H² não positivo no intervalo integrado.')
        integral = np.sum(self.weights / np.sqrt(e2), axis=-1)  # Integra 1/E ao longo da linha de visada.
        distance = (1 + self.z) * integral  # Converte a distância comóvel à distância de luminosidade adimensional.
        return 5 * np.log10(distance)  # Mantém a constante de calibração fora da previsão.

    def derivatives(self, center: ArrayLike, order: int = 4, chart: str = 'physical') -> tuple:
        """Deriva a quadratura por séries truncadas, sem diferenças finitas.

        A aritmética de séries implementa a regra da cadeia. Os coeficientes
        internos contêm 1/(a! b!); o retorno J,H,T,U contém derivadas brutas,
        como exige prepare_dali. A expansão pode usar w ou log(-w).
        """
        omega, w = np.asarray(center, dtype=float)  # Fixa o mesmo ponto físico para todas as parametrizações.
        coordinates(center, chart)  # Confere se o mapa escolhido é válido nesse ponto.
        if order not in (1, 2, 3, 4):
            raise ValueError('order deve estar entre 1 e 4.')
        ell = 3 * self.log1pz  # Derivadas do fator de energia escura envolvem este logaritmo.
        dark0 = np.exp((1 + w) * ell)  # Calcula a evolução da energia escura no centro.
        w_coeff = [w] + [1. if k == 1 else 0. for k in range(1, order + 1)]  # Taylor de w em coordenadas físicas.
        if chart in ('logw', 'logboth'):
            w_coeff = [w / factorial(k) for k in range(order + 1)]  # Taylor de w=-exp(v) em v=log(-w).
        dark = [dark0]  # Constrói exp[ell*(1+w(v))] pela recorrência da série exponencial.
        for n in range(1, order + 1):
            dark.append(sum(k * ell * w_coeff[k] * dark[n-k] for k in range(1, n+1)) / n)
        matter_coeff = [omega] + [1. if k == 1 else 0. for k in range(1, order + 1)]  # Taylor de Omega_m.
        if chart == 'logboth':
            matter_coeff = [omega / factorial(k) for k in range(order + 1)]  # Taylor de Omega_m=exp(u).
        e2 = {}  # Organiza a série bivariada de E² antes de tomar a raiz inversa.
        for a in range(order + 1):
            for b in range(order + 1 - a):
                e2[a, b] = (dark[b] if a == 0 else 0.) + matter_coeff[a] * ((self.matter if b == 0 else 0.) - dark[b])
        base = e2[0, 0]  # Separa o valor positivo de E² no centro.
        perturbation = {key: value / base for key, value in e2.items() if key != (0, 0)}  # Prepara u=delta(E²)/E0².
        power, inverse_root = {(0, 0): np.ones_like(base)}, {}  # Começa pela potência u^0.
        binomial = 1.  # Coeficiente binomial (-1/2 choose k).
        for k in range(order + 1):
            for key, value in power.items():
                inverse_root[key] = inverse_root.get(key, 0.) + binomial * value / np.sqrt(base)  # Expande (E0²+delta E²)^(-1/2).
            power = _jet_product(power, perturbation, order)  # Prepara a potência seguinte sem perder termos mistos.
            binomial *= (-.5-k) / (k+1)  # Atualiza o coeficiente da série binomial.
        distance = {key: (1+self.z) * np.sum(value*self.weights, axis=-1) for key, value in inverse_root.items()}  # Integra cada coeficiente na mesma quadratura.
        distance0 = distance[0, 0]  # Separa a distância central antes de expandir o logaritmo.
        relative = {key: value/distance0 for key, value in distance.items() if key != (0, 0)}
        power, logarithm = {(0, 0): np.ones_like(distance0)}, {}  # Prepara log(1+u), sem diferenciar números quase iguais.
        for k in range(1, order + 1):
            power = _jet_product(power, relative, order)  # Calcula u^k até o grau necessário.
            for key, value in power.items():
                logarithm[key] = logarithm.get(key, 0.) + (-1)**(k+1) * value/k  # Soma a série do logaritmo.
        tensors = []  # Reconstrói os tensores simétricos esperados pelo módulo DALI.
        for k in range(1, order + 1):
            tensor = np.empty((self.z.size,) + (2,) * k)
            for indices in product(range(2), repeat=k):
                a, b = indices.count(0), indices.count(1)  # Conta as derivadas em cada coordenada.
                tensor[(slice(None),) + indices] = 5/np.log(10) * logarithm[a, b] * factorial(a) * factorial(b)  # Retira os fatoriais de Taylor.
            tensors.append(tensor)
        return tuple(tensors)


# Calibração: retirar a direção que um deslocamento comum não consegue medir
# -----------------------------------------------------------------------------

class CalibrationProjection:
    """Marginaliza um único offset com prior constante e C independente de theta.

    Após ponderar por C, uma reflexão ortogonal alinha a direção do offset
    com o primeiro eixo. Retirar esse eixo deixa N-1 resíduos com métrica
    identidade. Isso evita tentar fatorar a precisão projetada, que é singular.
    """
    def __init__(self, covariance: ArrayLike):
        cov = np.asarray(covariance, dtype=float)  # Aceita variâncias diagonais ou uma covariância completa.
        if cov.ndim == 1:
            if np.any(cov <= 0) or not np.all(np.isfinite(cov)):
                raise ValueError('As variâncias devem ser positivas e finitas.')
            self.sigma, self.factor = np.sqrt(cov), None  # Usa a rota diagonal sem formar uma matriz densa.
            self.size = cov.size
        elif cov.ndim == 2 and cov.shape[0] == cov.shape[1] and np.allclose(cov, cov.T):
            self.factor, self.sigma = cholesky(cov, lower=True), None  # Incorpora também as correlações do catálogo.
            self.size = cov.shape[0]
        else:
            raise ValueError('covariance deve ser um vetor de variâncias ou uma matriz simétrica positiva.')
        self.direction = self.whiten(np.ones(self.size))  # Pondera a direção de deslocamento comum.
        self.s0 = float(self.direction @ self.direction)  # Calcula 1.T C^-1 1.
        unit = self.direction / np.sqrt(self.s0)  # Normaliza a direção que será removida.
        unit[0] += 1.  # Escolhe o sinal numericamente estável da reflexão de Householder.
        self.reflection = unit / np.linalg.norm(unit)  # Guarda somente o vetor, sem construir uma matriz N por N.

    def whiten(self, values: ArrayLike) -> np.ndarray:
        """Pondera o último eixo por L^-1, onde C=L L.T."""
        values = np.asarray(values, dtype=float)
        if values.shape[-1] != self.size:
            raise ValueError('O último eixo deve enumerar as supernovas na ordem original.')
        if self.sigma is not None:
            return values / self.sigma  # Pondera cada dado pelo seu desvio, quando são independentes.
        flat = values.reshape(-1, self.size)  # Junta temporariamente os eixos que enumeram pontos cosmológicos.
        return solve_triangular(self.factor, flat.T, lower=True, check_finite=False).T.reshape(values.shape)

    def project(self, values: ArrayLike) -> np.ndarray:
        """Pondera e retira a direção do offset; devolve N-1 componentes."""
        whitened = self.whiten(values)  # Primeiro incorpora os erros e suas possíveis correlações.
        reflected = whitened - 2*np.sum(whitened*self.reflection, axis=-1, keepdims=True)*self.reflection  # Aplica a reflexão ortogonal.
        return reflected[..., 1:]  # Descarta apenas a direção que a calibração absorve.

    def offset(self, residual: ArrayLike) -> np.ndarray:
        """Obtém Ahat(theta)=S1/S0, com residual=d-mu(theta)."""
        return np.sum(self.whiten(residual)*self.direction, axis=-1)/self.s0  # Usa o mesmo peso da marginalização.

    def derivative_tensors(self, derivatives: tuple) -> tuple:
        """Aplica a mesma projeção aos eixos dos dados de J,H,T,U."""
        return tuple(np.moveaxis(self.project(np.moveaxis(t, 0, -1)), -1, 0) for t in derivatives)  # Conserva todos os eixos dos parâmetros.


def fit_supernovae(model: FlatWCDM, projection: CalibrationProjection, data: ArrayLike,
                   bounds: ArrayLike, start: ArrayLike = (.28, -1.)):
    """Ajusta os dois parâmetros após marginalizar a calibração analiticamente."""
    target = projection.project(data)  # Retira a calibração dos dados uma única vez.
    limits = np.asarray(bounds, dtype=float)  # Recebe uma linha [mínimo,máximo] por parâmetro.
    fit = least_squares(lambda theta: target-projection.project(model.predict(theta)), start,
                        bounds=(limits[:, 0], limits[:, 1]), jac='3-point',
                        xtol=1e-12, ftol=1e-12, gtol=1e-10)  # Minimiza o mesmo chi² usado na posterior.
    if not fit.success:
        raise RuntimeError(f'O ajuste não convergiu: {fit.message}')
    return fit


def loglike_grid(model: FlatWCDM, projection: CalibrationProjection, data: ArrayLike,
                 points: ArrayLike, batch_size: int = 128) -> np.ndarray:
    """Avalia a likelihood marginalizada em lotes, sem guardar todos os modelos."""
    points = np.asarray(points, dtype=float)  # Aceita uma malha ou uma lista de pontos cosmológicos.
    flat = points.reshape(-1, 2)  # Reúne os pontos para limitar a memória durante a quadratura.
    target = projection.project(data)  # Faz a transformação linear dos dados somente uma vez por chamada.
    output = np.empty(flat.shape[0])  # Reserva apenas um escalar por ponto.
    for begin in range(0, flat.shape[0], batch_size):
        end = min(begin+batch_size, flat.shape[0])  # Seleciona o lote que cabe em memória.
        residual = target-projection.project(model.predict(flat[begin:end]))  # Calcula o resíduo depois da calibração.
        output[begin:end] = -.5*np.sum(residual**2, axis=-1)  # Mantém somente a parte dependente de theta.
    return output.reshape(points.shape[:-1])
