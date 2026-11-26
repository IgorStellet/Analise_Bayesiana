"""Verificações matemáticas da hierarquia de DALI.

Da raiz do repositório: python -m unittest discover -s projeto/tests -v
"""
from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import dali


class TestDALI(unittest.TestCase):
    def setUp(self):
        # Duas coordenadas e três observáveis permitem resíduo não nulo num ponto estacionário.
        self.J = np.array([[1., 0.], [0., 1.], [1., 1.]])
        self.H = np.zeros((3, 2, 2))
        self.H[0, 0, 0] = .8
        self.H[1, 0, 1] = self.H[1, 1, 0] = .3
        self.H[2, 0, 0] = -.4
        self.T = np.zeros((3, 2, 2, 2))
        self.T[0, 1, 1, 1] = .6
        self.T[2, 0, 0, 0] = 1.2
        self.derivatives = (self.J, self.H, self.T)
        self.C = np.array([[.7, .1, .05], [.1, 1.1, -.08], [.05, -.08, .8]])
        self.points = np.array([[.1, -.2], [-.5, .3], [.7, -.4], [0., 0.]])

    @staticmethod
    def polynomial(points):
        # Modelo escrito diretamente, sem chamar a rotina de contração de DALI.
        x, y = np.asarray(points)[..., 0], np.asarray(points)[..., 1]
        return np.stack((x+.4*x**2+.1*y**3, y+.3*x*y,
                         x+y-.2*x**2+.2*x**3), axis=-1)

    def test_triplet_reproduces_cubic_model_and_normalized_data_density(self):
        residual = np.array([.11, -.07, .2])
        expansion = dali.prepare_dali([0, 0], self.derivatives, self.C, residual=residual)
        expected = multivariate_normal.logpdf(residual-self.polynomial(self.points),
                                              mean=np.zeros(3), cov=self.C)
        assert_allclose(expansion.loglike(self.points, order=3), expected, rtol=0, atol=2e-14)
        # A terceira derivada não nula precisa alterar o resultado em pontos afastados.
        self.assertGreater(np.max(abs(expansion.loglike(self.points, 2)-expected)), 1e-3)

    def test_singlet_matches_linear_likelihood_at_nonzero_center(self):
        center = np.array([1.2, -.4])
        residual = np.array([.13, -.2, .06])
        expansion = dali.prepare_dali(center, (self.J,), self.C, residual=residual)
        points = center+self.points
        expected = multivariate_normal.logpdf(residual-self.points @ self.J.T,
                                              mean=np.zeros(3), cov=self.C)
        assert_allclose(expansion.loglike(points, 1), expected, atol=2e-14)
        # A forma (...,p) deve funcionar também em malhas e para um ponto isolado.
        grid = points.reshape(2, 2, 2)
        assert_allclose(expansion.loglike(grid, 1).ravel(), expected, atol=2e-14)
        self.assertEqual(expansion.exponent(center, 1).shape, ())

    def test_factorials_and_cross_terms_agree_with_independent_polynomial_product(self):
        j, h, t = 1.1, -.7, .9
        variance = .4
        expansion = dali.prepare_dali([0], (np.array([[j]]), np.array([[[h]]]),
                                               np.array([[[[t]]]])), [[variance]])
        # numpy.polynomial.polymul faz o produto dos polinômios sem usar einsum.
        x = np.array([-.8, -.3, .2, .9])
        for order, coefficients in [(1, [0, j]), (2, [0, j, h/2]), (3, [0, j, h/2, t/6])]:
            squared = np.polynomial.polynomial.polymul(coefficients, coefficients)/(2*variance)
            expected = np.polynomial.polynomial.polyval(x, squared)
            assert_allclose(expansion.exponent(x[:, None], order), expected, atol=3e-15)

    def test_observed_hessian_keeps_residual_even_at_stationary_point(self):
        weights = np.array([-.1, -.1, .1])
        residual = self.C @ weights
        assert_allclose(self.J.T @ weights, [0., 0.], atol=1e-15)
        expansion = dali.prepare_dali([0, 0], self.derivatives, self.C, residual=residual)
        expected = dali.observed_gaussian_hessian(residual, self.J, self.H, self.C)
        # Calcula a Hessiana observada por diferenças da função avaliada, não por seus tensores.
        step = 1e-4
        eye = np.eye(2)*step
        origin = np.zeros(2)
        numeric = np.zeros((2, 2))
        for i in range(2):
            numeric[i, i] = (expansion.exponent(eye[i], 3)-2*expansion.exponent(origin, 3)
                             +expansion.exponent(-eye[i], 3))/step**2
        numeric[0, 1] = numeric[1, 0] = (
            expansion.exponent(eye[0]+eye[1], 3)-expansion.exponent(eye[0]-eye[1], 3)
            -expansion.exponent(-eye[0]+eye[1], 3)+expansion.exponent(-eye[0]-eye[1], 3)
        )/(4*step**2)
        assert_allclose(numeric, expected, atol=2e-8)
        self.assertGreater(np.max(abs(expected-dali.fisher_gaussian(self.J, self.C))), .05)

    def test_affine_parameter_change_preserves_all_three_orders(self):
        # theta = S phi. Transforma todos os eixos de parâmetros dos tensores.
        scale = np.array([[1.7, .2], [-.3, .6]])
        transformed = (
            self.J @ scale,
            np.einsum('iab,ac,bd->icd', self.H, scale, scale),
            np.einsum('iabc,ad,be,cf->idef', self.T, scale, scale, scale),
        )
        physical = dali.prepare_dali([0, 0], self.derivatives, self.C)
        rescaled = dali.prepare_dali([0, 0], transformed, self.C)
        points_rescaled = np.linalg.solve(scale, self.points.T).T
        for order in (1, 2, 3):
            assert_allclose(physical.loglike(self.points, order),
                            rescaled.loglike(points_rescaled, order), atol=3e-14)

    def test_complete_square_stays_nonnegative_when_quartic_truncation_fails(self):
        # mu(x)=x-x³ dá E3=x²/2-x⁴+x⁶/2; cortar em grau quatro produz caudas ruins.
        expansion = dali.prepare_dali([0], (np.array([[1.]]), np.zeros((1,1,1)),
                                               np.array([[[[-6.]]]])), [[1.]])
        x = np.linspace(-2, 2, 41)
        full = expansion.exponent(x[:, None], 3)
        quartic = .5*x**2-x**4
        assert_allclose(full, .5*(x-x**3)**2, atol=2e-14)
        self.assertTrue(np.all(full >= 0))
        self.assertLess(quartic.min(), -1.)


if __name__ == '__main__':
    unittest.main()
