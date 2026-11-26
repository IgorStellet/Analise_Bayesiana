"""Testes da aplicação à Lista 3 e do integrador por planos.

Da raiz: python -m unittest discover -s projeto/tests -v
"""
from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad, simpson
from scipy.signal import find_peaks
from scipy.stats import norm

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import dali
from src.lista3 import Lista3Model


class TestBloco4(unittest.TestCase):
    def test_streaming_three_dimensional_gaussian_marginals(self):
        means = [.2,-.4,.3]
        sigmas = [.6,.8,.5]
        axes = tuple(np.linspace(m-8*s,m+8*s,81) for m,s in zip(means,sigmas))
        def log_density(t):
            return sum(norm.logpdf(t[...,i],means[i],sigmas[i]) for i in range(3))
        result = dali.marginalize_grid_3d(log_density,axes,log_shift=float(log_density(np.array(means))))
        self.assertAlmostEqual(result.log_normalizer,0.,places=10)
        for i in range(3):
            assert_allclose(result.marginals[i],norm.pdf(axes[i],means[i],sigmas[i]),atol=2e-12)
        for i,j in ((0,1),(0,2),(1,2)):
            expected = np.outer(norm.pdf(axes[j],means[j],sigmas[j]),norm.pdf(axes[i],means[i],sigmas[i]))
            assert_allclose(result.pairs[i,j],expected,atol=3e-12)

    def test_density_moments_distinguish_gaussian_and_uniform(self):
        x = np.linspace(-8,8,4001)
        gaussian = dali.density_moments_1d(norm.pdf(x),x)
        self.assertAlmostEqual(gaussian['skewness'],0.,places=10)
        self.assertAlmostEqual(gaussian['excess_kurtosis'],0.,places=8)
        axis = np.linspace(-13,13,1001)
        uniform = dali.density_moments_1d(np.ones(axis.size)/26,axis)
        self.assertAlmostEqual(uniform['mean'],0.,places=12)
        self.assertAlmostEqual(uniform['std'],26/np.sqrt(12),places=10)
        self.assertAlmostEqual(uniform['excess_kurtosis'],-1.2,places=8)

    def test_model_gradient_and_hessian_match_finite_differences(self):
        model = Lista3Model()
        point = np.array([3.1,-.2,1.7])
        step = 1e-5
        basis = np.eye(3)*step
        gradient = np.array([(model.log_kernel(point+v)-model.log_kernel(point-v))/(2*step) for v in basis])
        hessian = np.column_stack([-(model.gradient(point+v)-model.gradient(point-v))/(2*step) for v in basis])
        assert_allclose(gradient,model.gradient(point),atol=3e-10)
        assert_allclose(hessian,model.hessian(point),atol=3e-10)

    def test_analytic_w_integral_matches_infinite_range_quadrature(self):
        model = Lista3Model()
        for x,y in [(3.,-1.),(-2.,2.),(5.,-3.)]:
            for include in (False,True):
                numeric = quad(lambda w: np.exp(model.log_kernel([x,y,w],include_factor=include)),
                               -np.inf,np.inf,epsabs=1e-10,epsrel=1e-11)[0]
                analytic = np.exp(model.log_integrated_w(x,y,include_factor=include))
                self.assertAlmostEqual(numeric/analytic,1.,places=10)

    def test_dali_second_to_fourth_orders_recover_full_given_kernel(self):
        model = Lista3Model()
        center = np.array([4.68317955,-1.,-5.62864643])
        residual = np.array([model.mu_x,0.])-model.prediction(center)
        expansion = dali.prepare_dali(center,model.derivatives(center),
                                      np.diag([model.sigma_x**2,model.sigma_y**2]),residual=residual)
        # As comparações são em log: caudas muito pequenas não podem passar por subfluxo a zero.
        points = np.random.default_rng(2026).uniform([-13,-13,-32],[13,13,32],size=(300,3))
        for order in (2,3,4):
            approximation = -expansion.exponent(points,order=order)+model.log_extra(points)
            assert_allclose(approximation,model.log_kernel(points),rtol=2e-13,atol=2e-11)

    def test_singleto_loses_y_at_the_joint_center(self):
        model = Lista3Model()
        center = np.array([4.68317955,-1.,-5.62864643])
        derivatives = model.derivatives(center)
        assert_allclose(derivatives[0][:,1],[0.,0.],atol=0)
        expansion = dali.prepare_dali(center,derivatives,np.diag([9.,36.]),
                                      residual=np.array([5.,0.])-model.prediction(center))
        points = np.column_stack((np.full(101,center[0]),np.linspace(-13,13,101),np.full(101,center[2])))
        values = -expansion.exponent(points,1)+model.log_extra(points)
        assert_allclose(values,np.full(values.size,values[0]),atol=0)
        self.assertGreater(model.hessian(center)[1,1],.01)

    def test_reference_streaming_matches_xy_analytic_integral_and_w_modes(self):
        model = Lista3Model()
        axes = (np.linspace(-13,13,81),np.linspace(-13,13,81),np.linspace(-32,32,321))
        result = dali.marginalize_grid_3d(model.log_kernel,axes,log_shift=3.2)
        xx,yy = np.meshgrid(*axes[:2])
        xy,log_integral = dali.normalize_log_grid(model.log_integrated_w(xx,yy),*axes[:2])
        assert_allclose(result.pairs[0,1],xy,atol=2e-11)
        self.assertAlmostEqual(result.log_normalizer,log_integral,places=10)
        peaks,_ = find_peaks(result.marginals[2],prominence=.01*result.marginals[2].max())
        self.assertEqual(len(peaks),2)
        self.assertLess(axes[2][peaks[0]],0)
        self.assertGreater(axes[2][peaks[1]],0)


if __name__ == '__main__':
    unittest.main()
