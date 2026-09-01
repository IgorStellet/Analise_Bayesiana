import numpy as np

from analise_bayesiana import (
    gaussian_pdf,
    posterior_disease_given_positive,
    posterior_discrete,
)


def test_exemplo_1_3_4_reproduz_likelihoods_e_odds_do_gregory() -> None:
    predictions = np.array([100.0, 200.0])
    likelihoods = gaussian_pdf(120.0, predictions, sigma=40.0)
    posterior, evidence = posterior_discrete([0.5, 0.5], likelihoods)

    np.testing.assert_allclose(
        likelihoods,
        [0.008801633169107994, 0.0013497741628297015],
        rtol=1e-12,
    )
    np.testing.assert_allclose(posterior.sum(), 1.0)
    np.testing.assert_allclose(posterior[0] / posterior[1], 6.52081912, rtol=1e-6)
    np.testing.assert_allclose(evidence, 0.00507570, rtol=1e-6)


def test_exemplo_1_4_1_reproduz_efeito_da_taxa_base() -> None:
    original = posterior_disease_given_positive(1e-4, 0.986, 0.023)
    improved = posterior_disease_given_positive(1e-4, 0.986, 0.005)

    np.testing.assert_allclose(original, 0.00426908, rtol=1e-6)
    np.testing.assert_allclose(improved, 0.01934054, rtol=1e-6)
