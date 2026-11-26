"""Validação científica das distâncias, projeção, derivadas e amostragem.

Da raiz do repositório: python -m unittest discover -s projeto/tests -v
"""
from pathlib import Path
from math import factorial
from itertools import product
import hashlib
import json
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad, simpson

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import dali, supernovas as sn, amostragem as mc


class TestBloco5(unittest.TestCase):
    def test_luminosity_distance_against_adaptive_quadrature_and_eds_limit(self):
        z = np.array([.015, .2, .7, 1.414])
        model = sn.FlatWCDM(z, 48)
        for omega, w in ((.28, -1.), (.05, -3.), (.9, -.2), (1., -1.)):
            integrals = np.array([quad(lambda u: 1/np.sqrt(omega*(1+u)**3+(1-omega)*(1+u)**(3*(1+w))),
                                      0, redshift, epsabs=1e-12, epsrel=1e-12)[0] for redshift in z])
            expected = 5*np.log10((1+z)*integrals)
            assert_allclose(model.predict([omega, w]), expected, atol=2e-12)
        # O universo só de matéria tem integral analítica 2[1-(1+z)^(-1/2)].
        assert_allclose(model.predict([1., -1.]), 5*np.log10(2*(1+z)*(1-1/np.sqrt(1+z))), atol=1e-12)

    def test_derivative_series_have_the_expected_remainder_at_every_order(self):
        model = sn.FlatWCDM([.015, .2, .7, 1.414])
        center, direction = np.array([.28, -1.01]), np.array([.2, .9])
        for chart in ('physical', 'logw', 'logboth'):
            tensors = model.derivatives(center, 4, chart)
            for order in (1, 2, 3, 4):
                errors = []
                for h in (.08, .04):
                    point = center+h*direction
                    delta = sn.coordinates(point, chart)-sn.coordinates(center, chart)
                    prediction = model.predict(center)+dali._taylor_shift(delta, tensors, order)
                    errors.append(max(abs(model.predict(point)-prediction)))
                ratio = errors[0]/errors[1]
                self.assertGreater(ratio, .65*2**(order+1))
                self.assertLess(ratio, 1.5*2**(order+1))

    def test_projection_matches_full_precision_and_integrated_offset(self):
        covariance = np.array([[.5, .1, .02], [.1, .8, .08], [.02, .08, .4]])
        projection = sn.CalibrationProjection(covariance)
        residual = np.array([.2, -.1, .35])
        precision = np.linalg.inv(covariance)
        ones = np.ones(3)
        s0, s1, s2 = ones@precision@ones, ones@precision@residual, residual@precision@residual
        exact = s2-s1*s1/s0
        assert_allclose(sum(projection.project(residual)**2), exact, atol=1e-14)
        assert_allclose(projection.offset(residual), s1/s0, atol=1e-14)
        assert_allclose(projection.project(residual+42.7), projection.project(residual), atol=3e-14)
        integral = quad(lambda offset: np.exp(-.5*(residual-offset)@precision@(residual-offset)), -10, 10, epsabs=1e-12)[0]
        assert_allclose(integral, np.sqrt(2*np.pi/s0)*np.exp(-.5*exact), atol=1e-12)

    def test_projection_derivatives_follow_the_same_linear_map(self):
        model = sn.FlatWCDM([.02, .1, .4, .8, 1.4])
        projection = sn.CalibrationProjection(np.linspace(.1, .3, 5)**2)
        center = np.array([.28, -1.])
        tensors = model.derivatives(center, 2)
        projected = projection.derivative_tensors(tensors)
        for i in (0, 1):
            step = np.zeros(2); step[i] = 1e-5
            finite = (projection.project(model.predict(center+step))-projection.project(model.predict(center-step)))/2e-5
            assert_allclose(projected[0][:, i], finite, atol=1e-8)
            finite_h = (projection.project(model.predict(center+step))-2*projection.project(model.predict(center))+projection.project(model.predict(center-step)))/1e-10
            assert_allclose(projected[1][:, i, i], finite_h, atol=3e-4)

    def test_qr_compression_preserves_all_orders_with_residual_and_mixed_terms(self):
        rng = np.random.default_rng(81)
        tensors = []
        for k in range(1, 5):
            tensor = np.empty((7,)+(2,)*k)
            coefficients = {a: rng.normal(size=7) for a in range(k+1)}
            for indices in product(range(2), repeat=k):
                tensor[(slice(None),)+indices] = coefficients[indices.count(0)]
            tensors.append(tensor)
        matrix = rng.normal(size=(7, 7)); covariance = matrix@matrix.T+np.eye(7)
        expansion = dali.prepare_dali([.1, -.3], tuple(tensors), covariance, rng.normal(size=7))
        compressed = dali.compress_dali(expansion)
        points = rng.normal(size=(80, 2))
        for order in (1, 2, 3, 4):
            assert_allclose(compressed.exponent(points, order), expansion.exponent(points, order), rtol=3e-14, atol=2e-12)
            self.assertTrue(np.all(compressed.exponent(points, order) >= 0))

    def test_physical_uniform_prior_needs_the_logw_jacobian(self):
        lower, upper = .2, 4.5
        v = np.linspace(np.log(lower), np.log(upper), 2001)
        prior_v = np.exp(v)/(upper-lower)
        self.assertAlmostEqual(float(simpson(prior_v, x=v)), 1., places=11)
        # Uma prior uniforme em v favorece valores pequenos de |w| e não é a mesma prior.
        physical_probability = (upper-2)/(upper-lower)
        changed_probability = np.log(upper/2)/np.log(upper/lower)
        self.assertGreater(abs(physical_probability-changed_probability), .25)

    def test_metropolis_recovers_a_correlated_gaussian(self):
        covariance = np.array([[1., .7], [.7, 2.]])
        precision = np.linalg.inv(covariance)
        center = np.array([.4, -.3])
        def logprob(points):
            delta = points-center
            return -.5*np.einsum('...i,ij,...j->...', delta, precision, delta)
        starts = center+np.array([[-3, -3], [2, 3], [-1, 2], [3, -2]])
        result = mc.metropolis(logprob, starts, covariance, steps=6500, warmup=1000, seed=58)
        flat = result.samples.reshape(-1, 2)
        assert_allclose(flat.mean(axis=0), center, atol=.08)
        assert_allclose(np.cov(flat.T), covariance, atol=.13)
        self.assertTrue(np.all((result.acceptance > .15) & (result.acceptance < .5)))
        for i in (0, 1):
            self.assertLess(mc.diagnostics(result.samples[:, :, i])['rhat'], 1.02)

    def test_diagnostics_distinguish_mixing_from_disagreement_between_chains(self):
        rng = np.random.default_rng(22)
        independent = rng.normal(size=(4, 4000))
        good = mc.diagnostics(independent)
        self.assertLess(good['rhat'], 1.01)
        self.assertGreater(good['ess_bulk'], 10000)
        independent[0] += 2.5
        bad = mc.diagnostics(independent)
        self.assertGreater(bad['rhat'], 1.1)
        self.assertLess(bad['ess_bulk'], 100)

    def test_official_data_integrity_and_covariance_order(self):
        folder = Path(__file__).resolve().parents[1]/'data/union21'
        records = json.loads((folder/'provenance.json').read_text())
        for record in records:
            self.assertEqual(hashlib.sha256((folder/record['file']).read_bytes()).hexdigest(), record['sha256'])
        catalog = sn.load_union21(folder/'SCPUnion2.1_mu_vs_z.txt')
        covariance = np.loadtxt(folder/'SCPUnion2.1_covmat_nosys.txt')
        assert_allclose(covariance, np.diag(catalog['sigma']**2), atol=2e-12)
        full = np.loadtxt(folder/'SCPUnion2.1_covmat_sys.txt')
        self.assertEqual(full.shape, (580, 580))
        assert_allclose(full, full.T, atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(full).min(), 0)


if __name__ == '__main__':
    unittest.main()
