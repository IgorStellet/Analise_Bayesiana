"""Funções de apoio para os exercícios do curso de Estatística Bayesiana."""

from .bayes import (
    gaussian_pdf,
    normalize,
    posterior_disease_given_positive,
    posterior_discrete,
)

__all__ = [
    "gaussian_pdf",
    "normalize",
    "posterior_disease_given_positive",
    "posterior_discrete",
]
