"""Confere métricas, custos e a correção exata de delayed acceptance."""
from pathlib import Path
import sys,unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import norm

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import dali,eficiencia as ef


class TestBloco8(unittest.TestCase):
    def test_marginal_quantiles_match_a_normal_reference(self):
        x=np.linspace(-7,7,4001);p=norm.pdf(x,loc=.3,scale=.8)
        assert_allclose(ef.quantiles(x,p),norm.ppf([.025,.16,.5,.84,.975],loc=.3,scale=.8),atol=1e-5)

    def test_probability_handles_cuts_and_support(self):
        x=np.linspace(-5,5,2001);p=norm.pdf(x)
        self.assertAlmostEqual(ef.probability_below(x,p,0),.5,places=12)
        self.assertEqual(ef.probability_below(x,p,-8),0.)
        self.assertEqual(ef.probability_below(x,p,8),1.)
        self.assertLess(abs(ef.probability_below(x,p,.37)-norm.cdf(.37)),2e-6)

    def test_summary_recovers_a_three_dimensional_gaussian(self):
        axes=tuple(np.linspace(-5,5,81) for _ in range(3))
        ref=dali.marginalize_grid_3d(lambda pts:-.5*np.sum(pts**2,axis=-1),axes)
        summary=ef.summaries(ref)
        assert_allclose(summary['mean'],0.,atol=1e-13)
        assert_allclose(summary['sd'],1.,atol=1e-5)
        errors=ef.precision_errors(ref,ref)
        for value in errors.values():assert_allclose(value,0.,atol=1e-14)

    def test_hpd_coverage_uses_the_same_region_for_both_densities(self):
        axes=tuple(np.linspace(-5,5,81) for _ in range(3))
        ref=dali.marginalize_grid_3d(lambda pts:-.5*np.sum(pts**2,axis=-1),axes)
        for row in ef.hpd_coverage(ref,ref):
            self.assertEqual(row['difference'],0.)
            self.assertLess(abs(row['exact_mass']-row['nominal']),.012)

    def test_break_even_includes_the_no_saving_case(self):
        self.assertEqual(ef.break_even(10.,.02,.01),1000.)
        self.assertTrue(np.isinf(ef.break_even(10.,.01,.02)))
        with self.assertRaises(ValueError):ef.break_even(-1.,1.,.5)

    def test_delayed_acceptance_obeys_detailed_balance(self):
        delta=np.array([-8.,-1.,.1,2.,7.]);cheap=np.array([2.,-5.,1.,-3.,8.])
        forward=sum(ef.delayed_log_acceptance(delta,cheap))
        reverse=sum(ef.delayed_log_acceptance(-delta,-cheap))
        assert_allclose(forward-reverse,delta,atol=2e-15)  # É o log da razão de fluxos necessária ao balanço detalhado.
        self.assertTrue(np.all(forward<=np.minimum(0.,delta)+1e-14))  # Para a mesma proposta, o filtro não aumenta a probabilidade de aceitação.

    def test_identical_approximation_recovers_the_exact_chain(self):
        density=lambda p:-.5*np.sum(p*p,axis=-1)
        starts=np.array([[-1.,.5],[1.,-.5]])
        kwargs=dict(steps=800,warmup=200,seed=29,bounds=[[-4,4],[-4,4]])
        exact=ef.metropolis(density,starts,np.eye(2),mode='exact',**kwargs)
        delayed=ef.metropolis(density,starts,np.eye(2),log_approx=density,mode='delayed',**kwargs)
        assert_allclose(exact.samples,delayed.samples,atol=0.)
        assert_allclose(exact.acceptance,delayed.acceptance,atol=0.)
        self.assertLess(sum(delayed.exact_evaluations.values()),sum(exact.exact_evaluations.values()))

    def test_counts_match_actual_calls_and_rejections_keep_states(self):
        actual={'exact':0,'approx':0}
        def full(p):actual['exact']+=len(p);return -.5*np.sum(p*p,axis=-1)
        def cheap(p):actual['approx']+=len(p);return -np.sum((p-.15)**2,axis=-1)
        result=ef.metropolis(full,[[0.,0.],[.2,-.2]],np.eye(2)*3,log_approx=cheap,mode='delayed',
            bounds=[[-2,2],[-2,2]],steps=1000,warmup=200,seed=3)
        self.assertEqual(actual['exact'],sum(result.exact_evaluations.values()))
        self.assertEqual(actual['approx'],sum(result.approximate_evaluations.values()))
        self.assertTrue(np.any(np.all(np.diff(result.samples,axis=1)==0,axis=-1)))
        self.assertTrue(np.all(abs(result.samples)<=2))

    def test_approximate_mode_does_not_evaluate_the_exact_target(self):
        def forbidden(p):raise AssertionError('O alvo completo foi chamado na rota aproximada.')
        density=lambda p:-.5*np.sum(p*p,axis=-1)
        result=ef.metropolis(forbidden,[[0.],[.2]],np.eye(1),log_approx=density,mode='approx',steps=100,warmup=20)
        self.assertEqual(sum(result.exact_evaluations.values()),0)

    def test_zero_approximation_inside_support_is_rejected(self):
        density=lambda p:-.5*np.sum(p*p,axis=-1)
        with self.assertRaises(ValueError):
            ef.metropolis(density,[[0.],[.2]],np.eye(1),log_approx=lambda p:np.full(len(p),-np.inf),mode='delayed',steps=100,warmup=20)

    def test_delayed_chain_recovers_a_normal_despite_wrong_approximation(self):
        full=lambda p:-.5*((p[:,0]-.4)/.8)**2
        cheap=lambda p:-.5*((p[:,0]+.2)/1.1)**2
        result=ef.metropolis(full,[[-1.],[0.],[1.],[2.]],np.array([[.64]]),log_approx=cheap,mode='delayed',
            steps=6000,warmup=1000,seed=83,bounds=[[-7,7]])
        self.assertLess(abs(result.samples.mean()-.4),.04)
        self.assertLess(abs(result.samples.std()-.8),.04)

    def test_invalid_cdf_and_sampler_configuration_are_rejected(self):
        with self.assertRaises(ValueError):ef.marginal_cdf([0,1],[-1,2])
        with self.assertRaises(ValueError):ef.quantiles([0,1],[1,1],[1.1])
        with self.assertRaises(ValueError):ef.metropolis(lambda p:np.zeros(len(p)),[[0.]],np.eye(1),mode='wrong')


if __name__=='__main__':unittest.main()
