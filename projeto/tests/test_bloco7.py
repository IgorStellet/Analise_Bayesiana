"""Testa a matemática da oficina, incluindo as ordens acima do módulo geral."""
from pathlib import Path
from math import factorial
import sys
import unittest
import numpy as np
from numpy.testing import assert_allclose

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import oficina as art,dali


class TestBloco7(unittest.TestCase):
    def setUp(self):
        self.axes=(np.linspace(-2.2,2.2,161),np.linspace(-2.2,2.6,181))

    def test_linear_model_is_exact_at_first_order(self):
        model=art.ClockModel(amplitude=0,bend=0)
        points=np.random.default_rng(7).normal(size=(50,2))
        assert_allclose(art.log_kernel(model,points,1),art.log_kernel(model,points),atol=5e-14)

    def test_polynomial_ring_closes_at_second_order(self):
        model=art.ClockModel(kind='ring',amplitude=0,bend=0)
        points=np.random.default_rng(8).normal(size=(60,2))
        for angle in (0.,.7,2.8):
            center=model.center(angle)
            assert_allclose(art.log_kernel(model,points,2,center),art.log_kernel(model,points),rtol=3e-13,atol=3e-10)

    def test_coefficients_include_mixed_terms_with_total_degree(self):
        model=art.ClockModel(kind='ring')
        c=model.coefficients(model.center(.7),5)[0]
        self.assertTrue(any(a and b and abs(v)>1e-8 for (a,b),v in c.items()))
        self.assertTrue(all(a+b<=5 for a,b in c))

    def test_high_order_residual_has_the_expected_local_remainder(self):
        model=art.ClockModel(kind='ring')
        center=model.center(.3);direction=np.array([.75,-.2])
        for order in (4,6,8):
            errors=[]
            for h in (.16,.08):
                point=center+h*direction
                errors.append(np.max(abs(art.evaluate_series(model.coefficients(center,order),point,center)-model.residual(point))))
            self.assertGreater(errors[0]/errors[1],.5*2**(order+1))

    def test_first_four_orders_match_the_existing_dali_implementation(self):
        model=art.ClockModel(kind='ring');center=model.center(.4)
        coefficients=model.coefficients(center,4)
        tensors=[]
        from itertools import product
        for k in (1,2,3,4):
            tensor=np.zeros((1,)+(2,)*k)
            for indices in product(range(2),repeat=k):
                a=indices.count(0);b=indices.count(1)
                tensor[(0,)+indices]=coefficients[0].get((a,b),0.)*factorial(a)*factorial(b)
            tensors.append(tensor)
        expansion=dali.prepare_dali(center,tuple(tensors),np.eye(1),residual=-model.residual(center))
        points=center+np.random.default_rng(4).normal(size=(40,2))*.4
        for order in (1,2,3,4):
            assert_allclose(expansion.exponent(points,order),-art.log_kernel(model,points,order,center),rtol=1e-12,atol=1e-10)

    def test_ring_fisher_is_singular_without_invented_regularization(self):
        model=art.ClockModel(kind='ring')
        for angle in (0.,.6,np.pi/2): self.assertEqual(np.linalg.matrix_rank(art.fisher(model,model.center(angle))),1)
        self.assertEqual(np.linalg.matrix_rank(art.fisher(art.ClockModel())),2)

    def test_normalization_and_nonnegative_energy_hold_for_all_orders(self):
        model=art.ClockModel()
        for order in (1,4,8,12,16):
            p=art.density(model,self.axes,order)
            self.assertAlmostEqual(dali.integrate_grid_2d(p,*self.axes),1.,places=12)
            self.assertTrue(np.all(p>=0))
            self.assertLessEqual(float(art.log_kernel(model,[1.,2.],order)),0.)

    def test_sine_parity_produces_equal_adjacent_orders(self):
        model=art.ClockModel()
        points=np.random.default_rng(4).normal(size=(50,2))
        for order in (3,5,7,9,11,13,15):
            assert_allclose(art.log_kernel(model,points,order),art.log_kernel(model,points,order+1),atol=1e-10)

    def test_rational_taylor_converges_inside_and_diverges_outside_radius(self):
        model=art.ClockModel(kind='rational')
        def error(x,K):
            f=model.warp_coefficients(0.,K)
            return abs(np.polynomial.polynomial.polyval(x,f)-model.warp(x))
        self.assertLess(error(.5*model.radius,16),error(.5*model.radius,4)*1e-3)
        self.assertGreater(error(1.3*model.radius,16),error(1.3*model.radius,4)*10)

    def test_morph_preserves_both_integer_endpoints(self):
        model=art.ClockModel();points=np.random.default_rng(2).normal(size=(40,2))
        assert_allclose(art.log_kernel(model,points,3,blend=0),art.log_kernel(model,points,3))
        assert_allclose(art.log_kernel(model,points,3,blend=1),art.log_kernel(model,points,4),atol=2e-12)

    def test_known_mixture_preserves_mass_and_weights(self):
        model=art.ClockModel(amplitude=.3)
        parts=[art.placed_density(model,self.axes,offset=(x,0.),scale=.55) for x in (-.8,.8)]
        p=art.mixture(parts,[.35,.65])
        self.assertAlmostEqual(dali.integrate_grid_2d(p,*self.axes),1.,places=12)
        assert_allclose(p,.35*parts[0]+.65*parts[1])
        with self.assertRaises(ValueError): art.mixture(parts,[.5,.6])

    def test_invalid_parameters_and_order_are_rejected(self):
        for kwargs in ({'sx':0},{'kind':'unknown'},{'amplitude':float('nan')}):
            with self.assertRaises(ValueError): art.ClockModel(**kwargs)
        for order in (0,17,2.5):
            with self.assertRaises(ValueError): art.ClockModel().coefficients([0,0],order)


if __name__=='__main__':
    unittest.main()
