"""Verificações dos exemplos controlados, da quarta ordem e das integrações 1D.

Da raiz do repositório: python -m unittest discover -s projeto/tests -v
"""
from itertools import permutations
from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.polynomial import Polynomial
from numpy.testing import assert_allclose
from scipy.integrate import simpson
from scipy.stats import norm, multivariate_normal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import dali


class TestBloco3(unittest.TestCase):
    def test_quartic_with_mixed_fourth_derivatives_matches_exact_likelihood(self):
        # mu=x+0.3y+(x²+y²)² contém termos de quarta ordem mistos, não só diagonais.
        J = np.array([[1., .3]])
        H = np.zeros((1,2,2))
        T = np.zeros((1,2,2,2))
        U = np.zeros((1,2,2,2,2))
        U[0,0,0,0,0] = U[0,1,1,1,1] = 24.
        for indices in set(permutations((0,0,1,1))):
            U[(0,)+indices] = 8.
        points = np.array([[.2,-.3], [-.7,.4], [.8,.5], [0.,0.]])
        residual = .13
        sigma = .5
        exact_model = points[:,0]+.3*points[:,1]+np.sum(points**2,axis=1)**2
        expansion = dali.prepare_dali([0,0], (J,H,T,U), [[sigma**2]], residual=[residual])
        assert_allclose(expansion.loglike(points, order=4), norm.logpdf(residual, exact_model, sigma), atol=2e-14)
        self.assertGreater(np.max(abs(expansion.loglike(points,4)-expansion.loglike(points,3))), .1)

    def test_quadratic_is_exact_even_away_from_origin_with_observed_residual(self):
        kappa = .65
        center = np.array([.3,-.2])
        J = np.array([[1.,0.], [-2*kappa*center[0],1.]])
        H = np.zeros((2,2,2))
        H[1,0,0] = -2*kappa
        covariance = np.array([[.6,.08], [.08,.3]])
        data = np.array([.12,.05])
        def model(points):
            return np.stack((points[...,0], points[...,1]-kappa*points[...,0]**2),axis=-1)
        points = np.array([[-.7,.4], [.1,-.3], [.8,.2]])
        expansion = dali.prepare_dali(center, (J,H), covariance, residual=data-model(center))
        expected = multivariate_normal.logpdf(data-model(points), mean=[0,0], cov=covariance)
        assert_allclose(expansion.loglike(points,2), expected, atol=4e-15)

    def test_one_dimensional_moments_and_normalization_on_nonuniform_grid(self):
        # O espaçamento varia fortemente: somar densidades sem as coordenadas falharia.
        x = 4*np.linspace(-1,1,4001)**3
        mean, sigma = .2, .45
        pdf, log_integral = dali.normalize_log_grid_1d(norm.logpdf(x,mean,sigma),x)
        measured_mean, measured_var = dali.grid_moments_1d(pdf,x)
        self.assertAlmostEqual(float(simpson(pdf,x=x)),1.,places=12)
        self.assertAlmostEqual(log_integral,0.,places=9)
        self.assertAlmostEqual(measured_mean,mean,places=9)
        self.assertAlmostEqual(measured_var,sigma**2,places=9)

    def test_inverse_distance_roots_and_prior_transform(self):
        lower, upper, sigma = .35, 3.2, .12
        distance = np.linspace(lower,upper,5701)
        exact, log_z = dali.normalize_log_grid_1d(norm.logpdf(1,1/distance,sigma)-np.log(upper-lower), distance)
        amplitude = np.linspace(1/upper,1/lower,5701)
        log_prior_amplitude = -np.log(upper-lower)-2*np.log(amplitude)
        _, log_z_amplitude = dali.normalize_log_grid_1d(norm.logpdf(1,amplitude,sigma)+log_prior_amplitude,amplitude)
        back = np.exp(norm.logpdf(1,1/distance,sigma)-np.log(upper-lower)
                       -2*np.log(1/distance)-log_z_amplitude-2*np.log(distance))
        assert_allclose(back,exact,atol=1e-11)
        self.assertAlmostEqual(log_z_amplitude,log_z,places=11)
        # As raízes de delta-delta² e delta-delta²+delta³ demonstram o mecanismo.
        doublet = Polynomial([0,1,-1]).roots()
        triplet = Polynomial([0,1,-1,1]).roots()
        assert_allclose(np.sort(doublet.real),[0,1],atol=1e-14)
        self.assertEqual(sum(abs(z.imag)<1e-12 for z in triplet),1)

    def test_tail_integration_handles_float_rounding_without_near_duplicate_nodes(self):
        x = np.linspace(-.5,2,5001)
        pdf = norm.pdf(x,0,.25)
        # O nó mais próximo de 0.6 pode ser 0.6000000000000001.
        # Inserir literalmente 0.6 ao lado dele criaria um espaçamento quase nulo.
        tail = dali.grid_tail_mass_1d(pdf,x,.6)
        expected = norm.sf(.6/.25)-norm.sf(2/.25)
        self.assertAlmostEqual(tail,expected,places=12)
        perturbed = dali.grid_tail_mass_1d(pdf*(1+1e-12),x,.6)
        self.assertLess(abs(perturbed-tail*(1+1e-12)),1e-16)

    def test_simultaneous_spurious_roots_disappear_for_exact_quartic(self):
        coefficients = np.array([0.,1.,-1.,.1,.5])
        exact_model = Polynomial(coefficients)
        counts = []
        for order in (2,3,4):
            roots = Polynomial(coefficients[:order+1]).roots()
            counts.append(sum(abs(z.imag)<1e-10 and -.5<=z.real<=2 for z in roots))
        self.assertEqual(counts,[2,2,1])
        # A derivada positiva assegura que o modelo exato não tem a degenerescência no suporte.
        candidates = [-.5,2.]+[z.real for z in exact_model.deriv(2).roots() if abs(z.imag)<1e-12 and -.5<z.real<2]
        self.assertGreater(np.min(exact_model.deriv()(np.array(candidates))),.32)
        self.assertGreater(abs(exact_model(1)),.5)
        self.assertGreater(abs(exact_model(5-np.sqrt(15))),.7)

    def test_quartic_change_of_coordinate_preserves_posterior_and_evidence(self):
        x = np.linspace(-.5,2,5001)
        g = Polynomial([0.,1.,-1.,.1,.5])
        sigma = .12
        exact, log_z = dali.normalize_log_grid_1d(norm.logpdf(0,g(x),sigma)-np.log(2.5),x)
        u = g(x)
        jacobian = g.deriv()(x)
        log_product_u = norm.logpdf(0,u,sigma)-np.log(2.5)-np.log(jacobian)
        _, log_z_u = dali.normalize_log_grid_1d(log_product_u,u)
        back = np.exp(log_product_u-log_z_u+np.log(jacobian))
        assert_allclose(back,exact,atol=1e-9)
        self.assertAlmostEqual(log_z_u,log_z,places=9)


if __name__ == '__main__':
    unittest.main()