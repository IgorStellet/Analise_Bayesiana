"""BAO de BOSS DR12 e sua combinação com distâncias relativas de supernovas.

O vetor publicado intercala D_M*r_fid/r_d e H*r_d/r_fid nos três redshifts.
Usamos B=H0*r_d/r_fid como escala livre; não identificamos B com H0.
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
from . import dali
from .supernovas import FlatWCDM, _jet_product

C_KMS = 299792.458  # Velocidade da luz em km/s, para combinar Mpc com km/s/Mpc.
RD_FID_MPC = 147.78  # Convenção do lançamento BOSS DR12, não uma medida de r_d imposta ao modelo.


def load_boss_dr12(folder: str | Path) -> dict:
    """Lê somente o consenso BAO pós-reconstrução, sem combinar novamente FS."""
    folder = Path(folder)
    rows = np.genfromtxt(folder/'BAO_consensus_results_dM_Hz.txt', dtype=str)  # Preserva a ordem nativa das seis observáveis.
    covariance = np.loadtxt(folder/'BAO_consensus_covtot_dM_Hz.txt')  # Já inclui a incerteza total publicada.
    expected = np.array(['dM(rsfid/rs)', 'Hz(rs/rsfid)']*3)  # Fixa a convenção que também ordena linhas e colunas da matriz.
    if rows.shape != (6, 3) or not np.array_equal(rows[:, 1], expected):
        raise ValueError('A ordem esperada é DM,H em z=0.38,0.51,0.61.')
    if not np.allclose(rows[:, 0].astype(float), np.repeat([.38,.51,.61], 2)):
        raise ValueError('A tabela não tem os três redshifts efetivos esperados.')
    if covariance.shape != (6, 6) or not np.allclose(covariance, covariance.T):
        raise ValueError('A covariância deve ser simétrica e ter forma (6,6).')
    cholesky(covariance, lower=True)  # Confere positividade antes de usar a métrica gaussiana.
    return dict(z=rows[::2, 0].astype(float), labels=rows[:, 1], data=rows[:, 2].astype(float), covariance=covariance)


@dataclass
class FlatBAO:
    """Prevê as seis observáveis nativas em theta=(Omega_m,w,B)."""
    z: ArrayLike
    quadrature_order: int = 48

    def __post_init__(self):
        self.z = np.asarray(self.z, dtype=float)  # Mantém os três redshifts efetivos em sua ordem.
        self.cosmology = FlatWCDM(self.z, self.quadrature_order)  # Reutiliza a integral cosmológica já validada no Bloco 5.

    def shapes(self, theta: ArrayLike) -> tuple:
        """Separa I(z)=int dz/E de E(z), ambos sem a escala de calibração."""
        theta = np.asarray(theta, dtype=float)
        omega, w = theta[..., 0, None], theta[..., 1, None]  # Usa apenas os dois parâmetros de forma da expansão.
        integral = 10**(self.cosmology.predict(theta[..., :2])/5)/(1+self.z)  # Retira o logaritmo e o fator de luminosidade, obtendo I(z).
        e = np.sqrt(omega*(1+self.z)**3+(1-omega)*(1+self.z)**(3*(1+w)))  # Calcula a taxa relativa nos mesmos redshifts.
        return integral, e

    @staticmethod
    def native(integral: ArrayLike, e: ArrayLike, calibration: ArrayLike) -> np.ndarray:
        """Monta o vetor nativo: distâncias c I/B e taxas B E, intercaladas."""
        integral, e = np.asarray(integral), np.asarray(e)
        scale = np.asarray(calibration)[..., None]  # Permite um ponto ou um plano de calibrações.
        first, second = C_KMS*integral/scale, scale*e  # Calcula DM*r_fid/r_d e H*r_d/r_fid nas unidades publicadas.
        result = np.empty(first.shape[:-1]+(2*first.shape[-1],))
        result[..., ::2], result[..., 1::2] = first, second  # Conserva exatamente a ordem da tabela e da covariância.
        return result

    def predict(self, theta: ArrayLike) -> np.ndarray:
        """Avalia as seis previsões para um ponto ou uma malha (...,3)."""
        theta = np.asarray(theta, dtype=float)
        if theta.shape[-1] != 3 or np.any(theta[..., 2] <= 0):
            raise ValueError('theta deve terminar em Omega_m,w,B, com B positivo.')
        integral, e = self.shapes(theta[..., :2])  # A integração não depende de B.
        return self.native(integral, e, theta[..., 2])

    def derivatives(self, center: ArrayLike, order: int = 4) -> tuple:
        """Calcula J,H,T,U, incluindo todas as derivadas mistas da calibração.

        A parte cosmológica usa séries bivariadas; a dependência em B é
        analítica: DM proporcional a B^-1 e H proporcional a B. O retorno
        contém derivadas brutas, sem os fatores de Taylor.
        """
        omega, w, scale = np.asarray(center, dtype=float)
        if order not in (1,2,3,4) or scale <= 0:
            raise ValueError('Use order entre 1 e 4 e B positivo.')
        tensors_mu = self.cosmology.derivatives([omega,w], order)  # Reaproveita a regra da cadeia da integral de distância.
        log_distance = {}  # Coeficientes de delta(log d), com seus fatoriais de Taylor.
        for a in range(order+1):
            for b in range(order+1-a):
                if a+b:
                    index = (0,)*a+(1,)*b
                    log_distance[a,b] = np.log(10)/5*tensors_mu[a+b-1][(slice(None),)+index]/(factorial(a)*factorial(b))
        integral0, _ = self.shapes([omega,w])  # Calcula a integral no centro.
        power, integral_coeff = {(0,0):np.ones_like(self.z)}, {}  # Expande exp(delta log d) até a ordem necessária.
        for k in range(order+1):
            for key,value in power.items():
                integral_coeff[key] = integral_coeff.get(key,0.)+integral0*value/factorial(k)
            power = _jet_product(power,log_distance,order)  # Mantém também os termos mistos em Omega_m e w.
        matter, ell = (1+self.z)**3, 3*np.log1p(self.z)
        dark = (1+self.z)**(3*(1+w))
        e20 = omega*matter+(1-omega)*dark  # Separa o valor central de E².
        relative = {}
        for a in range(order+1):
            for b in range(order+1-a):
                if a+b:
                    value = ((1-omega)*dark*ell**b/factorial(b) if a==0 else
                             ((matter-dark if b==0 else -dark*ell**b/factorial(b)) if a==1 else np.zeros_like(dark)))
                    relative[a,b] = value/e20  # Prepara u=delta(E²)/E0².
        power, e_coeff, binomial = {(0,0):np.ones_like(self.z)}, {}, 1.
        for k in range(order+1):
            for key,value in power.items():
                e_coeff[key] = e_coeff.get(key,0.)+np.sqrt(e20)*binomial*value  # Expande a raiz positiva de E².
            power = _jet_product(power,relative,order)
            binomial *= (.5-k)/(k+1)  # Atualiza (1/2 choose k).
        output = []
        for k in range(1,order+1):
            tensor = np.empty((2*len(self.z),)+(3,)*k)
            for indices in product(range(3),repeat=k):
                a,b,c = (indices.count(i) for i in range(3))  # Conta as derivadas em Omega_m, w e B.
                factor = factorial(a)*factorial(b)*factorial(c)  # Retira os fatoriais dos coeficientes.
                first = C_KMS*integral_coeff.get((a,b),np.zeros_like(self.z))*(-1)**c/scale**(c+1)
                second = e_coeff.get((a,b),np.zeros_like(self.z))*(scale if c==0 else (1. if c==1 else 0.))
                tensor[(slice(None,None,2),)+indices] = first*factor  # Preenche as três distâncias.
                tensor[(slice(1,None,2),)+indices] = second*factor  # Preenche as três taxas.
            output.append(tensor)
        return tuple(output)


class DistanceCombination:
    """Une SN com offset marginalizado e BAO com escala acústica livre.

    A independência entre os dois conjuntos é uma hipótese explícita.
    As covariâncias internas são preservadas e a prior fica fora desta classe.
    """
    def __init__(self, sn_model, sn_projection, sn_data, bao_model, bao_data, bao_covariance):
        self.sn_model, self.sn_projection = sn_model, sn_projection
        self.sn_data = np.asarray(sn_data, dtype=float)
        self.sn_target = sn_projection.project(sn_data)  # Retira a calibração de luminosidade uma única vez.
        self.bao_model, self.bao_data = bao_model, np.asarray(bao_data,dtype=float)
        self.bao_factor = cholesky(bao_covariance,lower=True)  # Pondera as seis observáveis correlacionadas.

    def bao_whiten(self, values):
        """Aplica L_BAO^-1 ao último eixo, respeitando a ordem dos dados."""
        values = np.asarray(values,dtype=float)
        return solve_triangular(self.bao_factor,values.reshape(-1,6).T,lower=True,check_finite=False).T.reshape(values.shape)

    def residual(self, theta, case='joint'):
        """Devolve o resíduo ponderado usado tanto no ajuste quanto em DALI."""
        theta = np.asarray(theta,dtype=float)
        bao_residual = self.bao_whiten(self.bao_data-self.bao_model.predict(theta))
        if case=='bao':
            return bao_residual
        if case!='joint':
            raise ValueError('Use case=bao ou joint.')
        sn_residual = self.sn_target-self.sn_projection.project(self.sn_model.predict(theta[..., :2]))
        return np.concatenate((sn_residual,bao_residual),axis=-1)  # A soma de chi² vira uma única norma com os dois blocos.

    def loglike(self, theta, case='joint', batch_size=128):
        """Avalia a likelihood completa por lotes, sem truncar a cosmologia."""
        points = np.asarray(theta,dtype=float)
        flat = points.reshape(-1,3)  # Impede que uma lista grande crie todas as quadraturas de SN de uma vez.
        if len(flat)<=batch_size:
            residual = self.residual(points,case)
            return -.5*np.sum(residual**2,axis=-1)  # Mantém a rota curta usada pelas cadeias e pelo ajuste.
        output = np.empty(len(flat))
        for begin in range(0,len(flat),batch_size):
            residual = self.residual(flat[begin:begin+batch_size],case)  # Limita os modelos transitórios em memória.
            output[begin:begin+batch_size] = -.5*np.sum(residual**2,axis=-1)
        return output.reshape(points.shape[:-1])  # Recupera a forma da lista ou malha recebida.

    def fit(self, bounds, case='joint', start=(.3,-1.,68.)):
        """Ajusta os três parâmetros com a mesma likelihood da referência."""
        limits = np.asarray(bounds)
        result = least_squares(lambda x:self.residual(x,case),start,bounds=(limits[:,0],limits[:,1]),
                               jac='3-point',xtol=1e-12,ftol=1e-12,gtol=1e-10)
        if not result.success:
            raise RuntimeError(result.message)
        return result

    def derivatives(self, center, case='joint', order=4):
        """Projeta as derivadas de SN e pondera as de BAO no mesmo centro."""
        bao_raw = self.bao_model.derivatives(center,order)
        bao_derivatives = tuple(solve_triangular(self.bao_factor,t.reshape(6,-1),lower=True).reshape(t.shape) for t in bao_raw)
        if case=='bao':
            return bao_derivatives
        sn_raw = self.sn_projection.derivative_tensors(self.sn_model.derivatives(np.asarray(center)[:2],order))
        output = []
        for k,(sn_tensor,bao_tensor) in enumerate(zip(sn_raw,bao_derivatives),start=1):
            embedded = np.zeros((sn_tensor.shape[0],)+(3,)*k)  # SN não dependem de B depois da calibração livre.
            for indices in product(range(2),repeat=k):
                embedded[(slice(None),)+indices] = sn_tensor[(slice(None),)+indices]  # Acrescenta eixos de B sem inventar informação sobre essa escala.
            output.append(np.concatenate((embedded,bao_tensor),axis=0))
        return tuple(output)

    def expansions(self, center, case='joint', order=4):
        """Prepara uma forma comprimida própria para cada ordem, sem nova aproximação."""
        derivatives, residual = self.derivatives(center,case,order), self.residual(center,case)
        return {k:dali.compress_dali(dali.prepare_dali(center,derivatives[:k],np.eye(residual.size),residual)) for k in range(1,order+1)}

    def grid_evaluator(self, axes, sn_log_grid=None):
        """Prepara planos exatos em B sem reintegrar distâncias a cada escala."""
        first,second = np.meshgrid(*axes[:2])  # Mantém w nas linhas e Omega_m nas colunas.
        integral,e = self.bao_model.shapes(np.stack((first,second),axis=-1))  # Integra a cosmologia somente uma vez por ponto (Omega_m,w).
        def evaluate(points):
            scale = points[...,2]  # A malha 3D fornece um plano de calibração por chamada.
            prediction = self.bao_model.native(integral,e,scale)
            residual = self.bao_whiten(self.bao_data-prediction)
            result = -.5*np.sum(residual**2,axis=-1)
            return result if sn_log_grid is None else result+sn_log_grid  # A independência de SN em B permite reutilizar sua malha 2D.
        return evaluate


def dali_plane_evaluator(expansion, axes):
    """Reutiliza os monômios em Omega_m,w entre planos da mesma expansão."""
    x,y = np.meshgrid(*axes[:2])
    delta = np.stack((x,y),axis=-1)-expansion.center[:2]  # Mantém o centro comum das derivadas.
    base = -np.prod(delta[...,None,:]**expansion.powers[:,:2],axis=-1)  # Calcula uma vez a parte do monômio que independe de B.
    base[...,0] = 1.  # O primeiro termo multiplica o resíduo observado, não o modelo.
    def evaluate(points):
        scale = points[...,2]-expansion.center[2]
        monomials = base*scale[...,None]**expansion.powers[:,2]  # Acrescenta apenas as potências da calibração corrente.
        residual = monomials@expansion.factor.T
        return -.5*np.sum(residual**2,axis=-1)  # Conserva o quadrado completo da ordem preparada.
    return evaluate
