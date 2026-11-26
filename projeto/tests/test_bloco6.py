"""Verifica convenções de BAO, derivadas, combinação e integração 3D."""
from pathlib import Path
import hashlib
import json
import sys
import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import cholesky, solve_triangular
from scipy.stats import multivariate_normal, norm

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import bao, dali, supernovas as sn


class TestBloco6(unittest.TestCase):
    def setUp(self):
        self.folder=Path(__file__).resolve().parents[1]/'data/boss_dr12'
        self.catalog=bao.load_boss_dr12(self.folder)
        self.model=bao.FlatBAO(self.catalog['z'])
        self.center=np.array([.31,-1.1,69.])

    def toy_combination(self):
        model=sn.FlatWCDM([.02,.2,.6,1.1])
        projection=sn.CalibrationProjection(np.array([.1,.12,.15,.2])**2)
        data=model.predict(self.center[:2])+43.+np.array([.03,-.02,.01,-.01])
        bdata=self.model.predict(self.center)+np.array([3,.2,-2,-.1,1,.05])
        return bao.DistanceCombination(model,projection,data,self.model,bdata,self.catalog['covariance'])

    def test_release_integrity_order_and_total_covariance(self):
        manifest=json.loads((self.folder/'provenance.json').read_text())
        for item in manifest['files']:
            self.assertEqual(hashlib.sha256((self.folder/item['file']).read_bytes()).hexdigest(),item['sha256'])
        assert_allclose(self.catalog['data'],[1512.39,81.2087,1975.22,90.9029,2306.68,98.9647])
        self.assertGreater(np.linalg.eigvalsh(self.catalog['covariance']).min(),0)
        self.assertGreater(abs(self.catalog['covariance'][0,2]),300)  # As faixas de redshift não são independentes.

    def test_native_and_dimensionless_vectors_have_identical_chi_squared(self):
        data,cov=self.catalog['data'],self.catalog['covariance']
        prediction=self.model.predict(self.center)
        transform=np.diag([1/bao.RD_FID_MPC,bao.RD_FID_MPC/bao.C_KMS]*3)
        residual=data-prediction
        expected=residual@np.linalg.solve(cov,residual)
        shifted=transform@residual
        measured=shifted@np.linalg.solve(transform@cov@transform.T,shifted)
        assert_allclose(measured,expected,rtol=1e-13)
        integral,e=self.model.shapes(self.center[:2]);nu=self.center[2]*bao.RD_FID_MPC/bao.C_KMS
        assert_allclose(transform@prediction,np.stack((integral/nu,nu*e),axis=-1).ravel())

    def test_fiducial_ruler_rescaling_is_a_convention_not_a_prior(self):
        factor=1.17
        transform=np.diag([factor,1/factor]*3)
        new_center=self.center.copy();new_center[2]/=factor  # B=H0*r_d/r_fid se transforma com o r_fid da tabela.
        assert_allclose(self.model.predict(new_center),transform@self.model.predict(self.center),atol=2e-12)
        original=self.catalog['data']-self.model.predict(self.center)
        changed=transform@self.catalog['data']-self.model.predict(new_center)
        assert_allclose(changed@np.linalg.solve(transform@self.catalog['covariance']@transform.T,changed),
                        original@np.linalg.solve(self.catalog['covariance'],original),rtol=2e-13)

    def test_all_mixed_derivatives_have_the_correct_taylor_remainder(self):
        derivatives=self.model.derivatives(self.center,4)
        direction=np.array([.2,.9,15.])  # Inclui termos mistos de B, não apenas derivadas cosmológicas.
        for order in (1,2,3,4):
            errors=[]
            for h in (.08,.04):
                predicted=self.model.predict(self.center)+dali._taylor_shift(h*direction,derivatives,order)
                errors.append(np.max(abs(self.model.predict(self.center+h*direction)-predicted)))
            self.assertGreater(errors[0]/errors[1],.6*2**(order+1))
            self.assertLess(errors[0]/errors[1],1.6*2**(order+1))

    def test_calibration_derivatives_agree_with_reciprocal_and_linear_scalings(self):
        derivatives=self.model.derivatives(self.center,4)
        from math import factorial
        prediction=self.model.predict(self.center)
        for k,tensor in enumerate(derivatives,start=1):
            measured=tensor[(slice(None),)+(2,)*k]
            assert_allclose(measured[::2],prediction[::2]*(-1)**k*factorial(k)/self.center[2]**k,atol=1e-12)
            assert_allclose(measured[1::2],prediction[1::2]/self.center[2] if k==1 else 0,atol=1e-12)

    def test_joint_likelihood_and_fisher_add_at_a_common_center(self):
        combo=self.toy_combination();points=self.center+np.array([[.02,.1,1.],[-.01,-.1,-.5]])
        sn_res=combo.sn_target-combo.sn_projection.project(combo.sn_model.predict(points[:,:2]))
        expected=-.5*np.sum(sn_res**2,axis=-1)+combo.loglike(points,'bao')
        assert_allclose(combo.loglike(points),expected,atol=2e-12)
        J=combo.derivatives(self.center,order=1)[0];n=combo.sn_target.size
        assert_allclose(J[:n,2],0.,atol=0)
        assert_allclose(J.T@J,J[:n].T@J[:n]+J[n:].T@J[n:],atol=2e-12)
        self.assertEqual(np.linalg.matrix_rank(J[:n].T@J[:n]),2)

    def test_joint_expansion_keeps_the_two_residual_blocks_at_every_order(self):
        combo=self.toy_combination();derivatives=combo.derivatives(self.center);residual=combo.residual(self.center)
        n=combo.sn_target.size
        full=dali.prepare_dali(self.center,derivatives,np.eye(len(residual)),residual)
        first=dali.prepare_dali(self.center,tuple(t[:n] for t in derivatives),np.eye(n),residual[:n])
        second=dali.prepare_dali(self.center,tuple(t[n:] for t in derivatives),np.eye(6),residual[n:])
        points=self.center+np.array([[.02,.1,1.],[-.03,-.15,-.8]])
        for k in (1,2,3,4):
            assert_allclose(full.exponent(points,k),first.exponent(points,k)+second.exponent(points,k),atol=1e-11)

    def test_cached_exact_and_dali_planes_equal_pointwise_evaluations(self):
        combo=self.toy_combination();axes=(np.linspace(.25,.4,9),np.linspace(-1.4,-.8,11),np.linspace(65,73,7))
        xx,yy=np.meshgrid(*axes[:2]);xy=np.stack((xx,yy),axis=-1)
        logsn=sn.loglike_grid(combo.sn_model,combo.sn_projection,combo.sn_data,xy)
        exact=combo.grid_evaluator(axes,logsn)
        expansions=combo.expansions(self.center)
        for calibration in axes[2][::3]:
            points=np.stack((xx,yy,np.full_like(xx,calibration)),axis=-1)
            assert_allclose(exact(points),combo.loglike(points),atol=4e-12)
            assert_allclose(combo.loglike(points,batch_size=7),combo.loglike(points),atol=4e-12)  # Confere a mesma likelihood com memória limitada por lotes.
            for k,expansion in expansions.items():
                evaluate=bao.dali_plane_evaluator(expansion,axes)
                assert_allclose(evaluate(points),-expansion.exponent(points,k),atol=4e-12)

    def test_three_dimensional_moments_recover_correlations(self):
        mean=np.array([.1,-.2,.3]);cov=np.array([[1.,.3,-.2],[.3,.8,.1],[-.2,.1,.7]])
        axes=tuple(np.linspace(-7,7,81) for _ in range(3))
        result=dali.marginalize_grid_3d(lambda p:multivariate_normal.logpdf(p,mean=mean,cov=cov),axes,log_shift=-3.)
        measured,mcov=dali.grid_moments_3d(result)
        assert_allclose(measured,mean,atol=2e-9);assert_allclose(mcov,cov,atol=2e-8)

    def test_total_variation_3d_matches_shifted_gaussian_result(self):
        axes=tuple(np.linspace(-8,8,81) for _ in range(3));shift=np.array([.7,-.2,.1])
        first=lambda p:multivariate_normal.logpdf(p,mean=[0,0,0],cov=np.eye(3))
        second=lambda p:multivariate_normal.logpdf(p,mean=shift,cov=np.eye(3))
        a=dali.marginalize_grid_3d(first,axes,log_shift=-3.);b=dali.marginalize_grid_3d(second,axes,log_shift=-3.)
        self.assertEqual(dali.total_variation_grid_3d(first,a,first,a),0.)
        expected=2*norm.cdf(np.linalg.norm(shift)/2)-1
        self.assertLess(abs(dali.total_variation_grid_3d(first,a,second,b)-expected),.0015)


if __name__=='__main__':
    unittest.main()
