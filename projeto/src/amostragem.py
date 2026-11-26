"""Metropolis com cadeias independentes e diagnósticos escalares explícitos.

Fisher determina a forma inicial da proposta, não substitui a likelihood
que aparece na aceitação. A adaptação de escala termina no aquecimento.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.stats import norm, rankdata


@dataclass(frozen=True)
class MetropolisResult:
    samples: np.ndarray  # Forma (cadeias, passos retidos, parâmetros).
    acceptance: np.ndarray  # Fração aceita por cadeia depois do aquecimento.
    scales: np.ndarray  # Escalas finais, congeladas na fase de produção.


def metropolis(log_probability, starts, proposal_covariance, steps=16000, warmup=3000, seed=123):
    """Amostra uma log-posterior vetorizada, incluindo suas fronteiras.

    Uma proposta fora do suporte deve retornar -inf. Cada cadeia recebe
    seu próprio gerador; a escala é ajustada só durante warmup. Não há
    descarte periódico: os diagnósticos usam todos os passos retidos.
    """
    current = np.array(starts, dtype=float, copy=True)  # Mantém um ponto inicial por cadeia.
    chains, dimensions = current.shape  # Identifica o número de cadeias e parâmetros.
    if steps <= warmup or warmup < 0:
        raise ValueError('steps deve superar warmup>=0.')
    factor = np.linalg.cholesky(proposal_covariance)  # Correlaciona os saltos segundo a geometria fornecida.
    generators = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(chains)]  # Gera fluxos independentes e reproduzíveis.
    log_current = np.asarray(log_probability(current))  # Avalia a posterior completa nos pontos iniciais.
    if not np.all(np.isfinite(log_current)):
        raise ValueError('Todos os pontos iniciais precisam estar dentro do suporte.')
    scales = np.full(chains, 2.38/np.sqrt(dimensions))  # Escolhe apenas uma escala inicial, não uma garantia de eficiência.
    accepted = np.zeros(chains)  # Conta aceitações na fase de produção.
    window = np.zeros(chains)  # Conta aceitações do intervalo de adaptação.
    samples = np.empty((chains, steps-warmup, dimensions))  # Reserva somente a fase que será analisada.
    for step in range(steps):
        noise = np.stack([rng.normal(size=dimensions) for rng in generators])  # Sorteia um salto gaussiano independente por cadeia.
        proposed = current + scales[:, None]*(noise@factor.T)  # Usa Fisher apenas para orientar a proposta simétrica.
        log_proposed = np.asarray(log_probability(proposed))  # Reavalia a posterior exata no novo ponto.
        log_uniform = np.log([rng.uniform() for rng in generators])  # Faz a decisão em log para evitar subfluxo.
        move = log_uniform < np.minimum(0., log_proposed-log_current)  # Aplica a razão de Metropolis; o termo da proposta cancela.
        current[move], log_current[move] = proposed[move], log_proposed[move]  # Mantém o estado anterior nas rejeições.
        window += move  # Registra a taxa recente durante o aquecimento.
        if step < warmup and (step+1) % 100 == 0:
            scales *= np.exp(np.clip(window/100-.28, -.3, .3))  # Ajusta a escala sem modificar o alvo da posterior.
            window[:] = 0  # Inicia uma nova janela de adaptação.
        if step >= warmup:
            samples[:, step-warmup] = current  # Registra também estados repetidos, que fazem parte da cadeia.
            accepted += move  # Mede aceitação somente com a proposta já congelada.
    return MetropolisResult(samples, accepted/(steps-warmup), scales)


def _split(values):
    """Divide cada cadeia em duas para detectar diferenças entre início e fim."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] < 20 or not np.all(np.isfinite(values)):
        raise ValueError('Use séries finitas com forma (cadeias,passos), com pelo menos 20 passos.')
    half = values.shape[1]//2  # Retira no máximo um ponto para criar metades do mesmo tamanho.
    return np.concatenate((values[:, :half], values[:, -half:]), axis=0)


def _rank_normalize(values):
    """Transforma os postos conjuntos em quantis normais, incluindo empates."""
    ranks = rankdata(values.ravel(), method='average')  # Trata os estados repetidos pela média dos postos.
    return norm.ppf((ranks-.375)/(ranks.size+.25)).reshape(values.shape)  # Usa a transformação de postos de Blom.


def _rhat(values):
    """Compara a variância entre cadeias com a variância dentro de cada cadeia."""
    m, n = values.shape
    within = np.mean(np.var(values, axis=1, ddof=1))  # Estima a variância interna média.
    between = n*np.var(np.mean(values, axis=1), ddof=1)  # Mede a discordância entre médias das cadeias.
    if within == 0:
        return 1. if between == 0 else np.inf
    variance = (n-1)/n*within + between/n  # Combina as duas fontes de variação.
    return float(np.sqrt(variance/within))


def _ess(values):
    """ESS mult cadeia pela sequência inicial positiva e monótona de Geyer."""
    m, n = values.shape
    centered = values-values.mean(axis=1, keepdims=True)  # Retira a média de cada cadeia antes da autocovariância.
    length = 1 << (2*n-1).bit_length()  # Evita o retorno circular da correlação calculada por FFT.
    transform = np.fft.rfft(centered, n=length, axis=1)  # Transforma as séries para calcular todas as defasagens.
    autocov = np.fft.irfft(transform*transform.conj(), n=length, axis=1)[:, :n]/n  # Usa a autocovariância com denominador fixo n.
    within = np.mean(np.var(values, axis=1, ddof=1))
    between = n*np.var(values.mean(axis=1), ddof=1)
    variance = (n-1)/n*within + between/n  # Penaliza cadeias que ainda discordam entre si.
    if variance == 0:
        return float(m*n)
    rho = 1-(within-autocov.mean(axis=0))/variance  # Estima a correlação combinada entre cadeias.
    rho[0] = 1.  # Impõe a normalização da autocorrelação no atraso zero.
    previous, pair_sum = np.inf, 0.  # Acumula pares positivos e não crescentes.
    for lag in range(0, n-1, 2):
        pair = rho[lag]+rho[lag+1]  # Agrupa atrasos consecutivos para reduzir o ruído das caudas.
        if pair <= 0:
            break
        previous = min(previous, pair)  # Aplica a sequência inicial monótona.
        pair_sum += previous
    tau = max(1., -1+2*pair_sum)  # Usa uma estimativa conservadora quando há anticorrelação.
    return float(min(m*n, m*n/tau))


def diagnostics(values):
    """Calcula Rhat por postos e dobramento, ESS bulk/tail e MCSE da média.

    As fórmulas seguem Vehtari et al. (2021), para uma série escalar.
    São diagnósticos; valores bons não demonstram a ausência de modos remotos.
    A referência em malha faz uma verificação complementar neste projeto.
    """
    values = np.asarray(values, dtype=float)
    split = _split(values)  # Examina também a estabilidade temporal de cada cadeia.
    ranked = _rank_normalize(split)  # Mede a mistura da região central por postos.
    folded = _rank_normalize(abs(split-np.median(split)))  # Detecta diferenças na dispersão entre cadeias.
    lower, upper = np.quantile(values, [.05, .95])  # Seleciona probabilidades das duas caudas.
    ess_mean = _ess(split)  # Mantém a escala física para estimar o erro da média.
    return dict(rhat=max(_rhat(ranked), _rhat(folded)), ess_bulk=_ess(ranked),
                ess_tail=min(_ess((split <= lower).astype(float)), _ess((split >= upper).astype(float))),
                ess_mean=ess_mean, mcse_mean=float(np.std(values, ddof=1)/np.sqrt(ess_mean)))
