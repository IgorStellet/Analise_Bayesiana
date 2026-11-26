"""Modelo da Lista 3: núcleo fornecido, derivadas e integral analítica em w.

As constantes independentes dos parâmetros foram retiradas do núcleo.
Sua integral normaliza a posterior; não é usada para comparar modelos.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Lista3Model:
    """Reúne os valores do enunciado para manter as contas consistentes.

    Altere aqui os desvios ou a média para explorar outro problema.
    Os coeficientes 0.5, 0.3 e o fator 2w²+3 são os da questão original.
    """
    mu_x: float = 5.0
    sigma_x: float = 3.0
    sigma_y: float = 6.0
    sigma_w: float = 4.0
    halfwidth: float = 13.0

    def __post_init__(self):
        values = (self.mu_x,self.sigma_x,self.sigma_y,self.sigma_w,self.halfwidth)
        if not np.all(np.isfinite(values)) or min(values[1:]) <= 0:
            raise ValueError("Os parâmetros devem ser finitos; desvios e halfwidth devem ser positivos.")

    @staticmethod
    def _points(points: ArrayLike) -> FloatArray:
        values = np.asarray(points,dtype=float)  # Aceita um ponto ou uma malha com três coordenadas no último eixo.
        if values.ndim < 1 or values.shape[-1] != 3 or not np.all(np.isfinite(values)):
            raise ValueError("points deve ser finito e ter forma (...,3), na ordem x,y,w.")
        return values

    def prediction(self, points: ArrayLike) -> FloatArray:
        """Organiza a parte gaussiana como mu=(x,r), com dado formal (mu_x,0)."""
        values = self._points(points)  # Confere a ordem dos parâmetros.
        x,y,w = np.moveaxis(values,-1,0)  # Separa as coordenadas sem perder os eixos da malha.
        r = y+0.5*y**2+0.3*x**2+w  # Calcula a combinação quadrática dada no enunciado.
        return np.stack((x,r),axis=-1)

    def log_extra(self, points: ArrayLike, include_factor: bool = True) -> FloatArray:
        """Conserva h(w), a prior gaussiana de w e o suporte de x,y.

        Omite somente as constantes 1/(26² sigma_w sqrt(2pi)), comuns a
        todos os métodos. include_factor=False define o controle sem h;
        isso muda o modelo, não é uma ordem de DALI.
        """
        values = self._points(points)
        x,y,w = np.moveaxis(values,-1,0)
        result = -0.5*(w/self.sigma_w)**2  # Mantém a prior gaussiana exata de w.
        if include_factor:
            result = result+np.log(2*w**2+3)  # Mantém o fator não gaussiano exatamente como foi fornecido.
        allowed = (abs(x)<=self.halfwidth)&(abs(y)<=self.halfwidth)  # Aplica as duas priors uniformes.
        return np.where(allowed,result,-np.inf)

    def log_kernel(self, points: ArrayLike, include_factor: bool = True) -> FloatArray:
        """Avalia log q da posterior, sem aproximação do modelo."""
        prediction = self.prediction(points)  # Obtém x e o resíduo quadrático r.
        gaussian = -0.5*((prediction[...,0]-self.mu_x)/self.sigma_x)**2  # Primeiro fator gaussiano.
        gaussian -= 0.5*(prediction[...,1]/self.sigma_y)**2  # Segundo fator, que acopla x,y,w.
        return gaussian+self.log_extra(points,include_factor=include_factor)

    def gradient(self, point: ArrayLike) -> FloatArray:
        """Gradiente analítico de log q num ponto interior ao suporte."""
        point = self._points(point)
        if point.shape != (3,):
            raise ValueError("gradient recebe um ponto de forma (3,).")
        x,y,w = point
        r = y+0.5*y*y+0.3*x*x+w  # Calcula o resíduo antes de derivar seus dois fatores.
        return np.array([-(x-self.mu_x)/self.sigma_x**2-r*0.6*x/self.sigma_y**2,
                         -r*(1+y)/self.sigma_y**2,
                         -r/self.sigma_y**2+4*w/(2*w*w+3)-w/self.sigma_w**2])

    def hessian(self, point: ArrayLike) -> FloatArray:
        """Retorna a Hessiana de -log q; inclui resíduo, fator h e prior de w."""
        point = self._points(point)
        if point.shape != (3,):
            raise ValueError("hessian recebe um ponto de forma (3,).")
        x,y,w = point
        r = y+0.5*y*y+0.3*x*x+w
        g = np.array([0.6*x,1+y,1.0])  # Calcula o gradiente de r.
        result = np.diag([1/self.sigma_x**2,0,1/self.sigma_w**2])  # Inclui o primeiro gaussiano e a prior de w.
        result += (np.outer(g,g)+r*np.diag([0.6,1,0]))/self.sigma_y**2  # Preserva o termo com resíduo na curvatura observada.
        result[2,2] += 4*(2*w*w-3)/(2*w*w+3)**2  # Acrescenta a curvatura de -log h.
        return result

    def derivatives(self, center: ArrayLike) -> tuple[FloatArray,...]:
        """Devolve J,H,T,U brutos do modelo gaussiano; T e U são zero."""
        center = self._points(center)
        if center.shape != (3,):
            raise ValueError("center deve ter forma (3,).")
        x,y,_ = center
        jacobian = np.array([[1,0,0],[0.6*x,1+y,1]])  # Deriva as duas previsões em relação a x,y,w.
        hessian = np.zeros((2,3,3))  # Prepara as segundas derivadas por observável.
        hessian[1] = np.diag([0.6,1,0])  # Somente a segunda previsão contém termos quadráticos.
        third = np.zeros((2,3,3,3))  # Um modelo quadrático não tem terceira derivada.
        fourth = np.zeros((2,3,3,3,3))  # Nem quarta derivada.
        return jacobian,hessian,third,fourth

    def log_integrated_w(self, x: ArrayLike, y: ArrayLike, include_factor: bool = True) -> FloatArray:
        """Integra q em w de -infinito a infinito por completar quadrados.

        Confere a marginal xy e o recorte numérico em w por uma rota
        independente do integrador 3D. Mantém as mesmas constantes de q.
        """
        x,y = np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(y,dtype=float))
        a = y+0.5*y*y+0.3*x*x  # Separa a parte de r que não contém w.
        variance = 1/(1/self.sigma_y**2+1/self.sigma_w**2)  # Variância após completar o quadrado em w.
        mean = -variance*a/self.sigma_y**2  # Média da gaussiana auxiliar em w, antes de multiplicar por h.
        result = -0.5*((x-self.mu_x)/self.sigma_x)**2-a*a/(2*(self.sigma_y**2+self.sigma_w**2))
        result += 0.5*np.log(2*np.pi*variance)  # Integra a gaussiana auxiliar sem normalização prévia.
        if include_factor:
            result += np.log(3+2*(variance+mean*mean))  # Usa E[2w²+3]=2(V+m²)+3.
        allowed = (abs(x)<=self.halfwidth)&(abs(y)<=self.halfwidth)
        return np.where(allowed,result,-np.inf)
