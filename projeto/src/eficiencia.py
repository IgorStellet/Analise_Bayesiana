"""Precisão, medições de custo e uso de DALI como filtro de MCMC.

As rotas exact e delayed têm o alvo completo. A rota approx amostra
somente a aproximação. O nome da rota não mede sua convergência: os
diagnósticos e o confronto com a referência ficam no notebook.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.integrate import cumulative_trapezoid
from . import dali


# Quantidades inferidas: os resumos usam as distribuições normalizadas.
# ------------------------------------------------------------------

def marginal_cdf(axis,pdf):
    """Integra uma marginal com pesos positivos, produzindo uma CDF monótona.

    A referência principal usa Simpson. Para inverter a CDF, usamos
    trapézios positivos e interpolação linear; a resolução dos quantis
    é conferida separadamente, sem prometer precisão além da malha.
    """
    axis,pdf=np.asarray(axis,dtype=float),np.asarray(pdf,dtype=float)
    if axis.ndim!=1 or axis.shape!=pdf.shape or np.any(np.diff(axis)<=0) or np.any(pdf<0):
        raise ValueError('Forneça um eixo crescente e uma densidade não negativa com a mesma forma.')
    if not np.all(np.isfinite(axis)) or not np.all(np.isfinite(pdf)):
        raise ValueError('A marginal deve ser finita.')
    cdf=cumulative_trapezoid(pdf,x=axis,initial=0.)  # Mantém incrementos não negativos mesmo em caudas pequenas.
    if cdf[-1]<=0: raise ValueError('A marginal precisa ter massa positiva.')
    return cdf/cdf[-1]


def quantiles(axis,pdf,probabilities=(.025,.16,.5,.84,.975)):
    """Obtém quantis da marginal, sem impor uma aproximação gaussiana."""
    probabilities=np.asarray(probabilities,dtype=float)
    if np.any((probabilities<0)|(probabilities>1)): raise ValueError('Use probabilidades entre zero e um.')
    return np.interp(probabilities,marginal_cdf(axis,pdf),axis)


def probability_below(axis,pdf,threshold):
    """Calcula P(theta<threshold) pela CDF da mesma marginal."""
    return float(np.interp(threshold,axis,marginal_cdf(axis,pdf),left=0.,right=1.))


def summaries(result):
    """Resume média, covariância, correlação e quantis de uma referência 3D."""
    mean,covariance=dali.grid_moments_3d(result)
    sd=np.sqrt(np.diag(covariance))
    return dict(mean=mean,covariance=covariance,sd=sd,
        correlation=covariance/np.outer(sd,sd),
        quantiles=np.stack([quantiles(x,p) for x,p in zip(result.axes,result.marginals)]))


def precision_errors(reference,approximation):
    """Separa o erro de médias, larguras, quantis e correlações, na mesma base."""
    if not all(np.array_equal(x,y) for x,y in zip(reference.axes,approximation.axes)):
        raise ValueError('Use as mesmas malhas e os mesmos limites da prior.')
    exact,approx=summaries(reference),summaries(approximation)
    return dict(mean_in_reference_sd=(approx['mean']-exact['mean'])/exact['sd'],
        relative_sd=approx['sd']/exact['sd']-1,
        quantile_in_reference_sd=(approx['quantiles']-exact['quantiles'])/exact['sd'][:,None],
        correlation=approx['correlation']-exact['correlation'])


def hpd_coverage(reference,approximation,pair=(0,1),probabilities=(.6827,.9545)):
    """Mede quanta probabilidade EXATA está nas regiões HPD da aproximação.

    Também retorna a massa aproximada da máscara discretizada. Sua
    diferença para a massa exata separa erro da aproximação do pequeno
    desvio de conteúdo causado pelos pixels da região escolhida.
    """
    i,j=pair;x,y=reference.axes[i],reference.axes[j]
    exact,approx=reference.pairs[pair],approximation.pairs[pair]
    levels=dali.hpd_thresholds_2d(approx,x,y,probabilities)
    result=[]
    for probability,level in zip(probabilities,levels):
        mask=approx>=level  # A região é definida uma vez, pela aproximação.
        p=dali.integrate_grid_2d(exact*mask,x,y)
        q=dali.integrate_grid_2d(approx*mask,x,y)
        result.append(dict(nominal=probability,exact_mass=p,approx_mass=q,difference=p-q))
    return result


# Cronômetro: preparo e avaliação são medidos separadamente.
# --------------------------------------------------------

def benchmark(function,repeats=7,min_seconds=.035,max_loops=2048):
    """Mede blocos repetidos e retorna mediana e quartis por chamada.

    Uma chamada inicial aquece a rotina. O número de repetições internas
    cresce até um bloco mensurável, sem introduzir pausas artificiais.
    A função deve fazer o trabalho real, sem devolver um resultado em cache.
    """
    if repeats<3 or min_seconds<=0 or max_loops<1: raise ValueError('Escolha pelo menos três repetições e limites positivos.')
    function()  # Retira da medição o primeiro acesso à rotina e às bibliotecas.
    loops=1
    while True:
        start=perf_counter()
        for _ in range(loops): function()
        elapsed=perf_counter()-start
        if elapsed>=min_seconds or loops>=max_loops: break
        loops=min(max_loops,max(loops+1,int(loops*min_seconds/max(elapsed,1e-9)*1.15)))
    durations=[]
    for _ in range(repeats):
        start=perf_counter()
        for _ in range(loops): function()
        durations.append((perf_counter()-start)/loops)
    return dict(median_seconds=float(np.median(durations)),q25_seconds=float(np.quantile(durations,.25)),
        q75_seconds=float(np.quantile(durations,.75)),loops=loops,repeats=repeats)


def break_even(preparation,exact_cost,approximate_cost):
    """Retorna N*=preparo/(c_exato-c_aprox), ou infinito se não há ganho marginal."""
    if preparation<0 or min(exact_cost,approximate_cost)<=0: raise ValueError('Use custos positivos e preparo não negativo.')
    saving=exact_cost-approximate_cost
    return float(preparation/saving) if saving>0 else np.inf


# MCMC com proposta simétrica: comparação de alvos e aceitação em duas etapas.
# -------------------------------------------------------------------------

def delayed_log_acceptance(log_exact_change,log_approx_change):
    """Retorna os logs das duas probabilidades, para proposta simétrica.

    alpha1=min(1,exp(Delta log p_tilde));
    alpha2=min(1,exp(Delta log p-Delta log p_tilde)).
    As constantes de normalização cancelam nas duas diferenças.
    """
    return np.minimum(0.,log_approx_change),np.minimum(0.,np.asarray(log_exact_change)-log_approx_change)


@dataclass(frozen=True)
class SamplingResult:
    samples: np.ndarray  # Forma (cadeias,passos retidos,dimensões).
    acceptance: np.ndarray  # Aceitação global, somente na produção.
    first_acceptance: np.ndarray  # Propostas que passam no filtro, na produção.
    second_acceptance: np.ndarray  # Aceitação condicionada a passar pelo filtro.
    scales: np.ndarray  # Escalas finais, congeladas depois do aquecimento.
    exact_evaluations: dict  # Pontos realmente enviados ao modelo completo, por fase.
    approximate_evaluations: dict  # Pontos enviados ao modelo aproximado, por fase.
    total_seconds: float  # Inclui inicialização, aquecimento e produção desta cadeia.
    production_seconds: float  # Permite conferir a influência do custo fixo inicial.


def metropolis(log_exact,starts,proposal_covariance,*,log_approx=None,mode='exact',bounds=None,
               steps=40000,warmup=5000,seed=2026):
    """Executa MH completo, MH aproximado ou delayed acceptance.

    A forma da proposta é a mesma nas três rotas; somente sua escala
    adapta durante warmup. Todos os estados repetidos são retidos.
    Os limites são o suporte conhecido da prior, não um corte por DALI.
    Em delayed, a aproximação deve ser finita e positiva nesse suporte.
    """
    beginning=perf_counter()
    if mode not in ('exact','approx','delayed'): raise ValueError('Use exact, approx ou delayed.')
    if mode!='exact' and log_approx is None: raise ValueError('Esta rota requer log_approx.')
    current=np.array(starts,dtype=float,copy=True)
    if current.ndim!=2 or not np.all(np.isfinite(current)): raise ValueError('Use inícios finitos, uma linha por cadeia.')
    if steps<=warmup or warmup<0: raise ValueError('steps deve superar warmup>=0.')
    chains,dimensions=current.shape
    limits=np.asarray(bounds,dtype=float) if bounds is not None else np.tile([-np.inf,np.inf],(dimensions,1))
    if limits.shape!=(dimensions,2) or np.any(limits[:,0]>=limits[:,1]): raise ValueError('Os limites devem ter uma linha crescente por parâmetro.')
    def inside(values): return np.all((values>=limits[:,0])&(values<=limits[:,1]),axis=-1)
    if not np.all(inside(current)): raise ValueError('Todos os inícios devem estar no suporte.')
    factor=np.linalg.cholesky(proposal_covariance)
    generators=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(chains)]
    scales=np.full(chains,2.38/np.sqrt(dimensions))
    exact_counts={name:0 for name in ('initial','warmup','production')}
    approx_counts=exact_counts.copy()
    def evaluate(function,values,counts,phase,require_positive=False):
        counts[phase]+=len(values)  # Conta pontos, não apenas o número de chamadas vetorizadas.
        answer=np.asarray(function(values),dtype=float)
        if answer.shape!=(len(values),) or np.any(np.isnan(answer)) or np.any(np.isposinf(answer)):
            raise ValueError('A log-densidade deve retornar um valor válido por ponto.')
        if require_positive and not np.all(np.isfinite(answer)):
            raise ValueError('A aproximação precisa ser positiva em todo o suporte usado pelo filtro.')
        return answer
    log_current=evaluate(log_exact,current,exact_counts,'initial') if mode!='approx' else None
    cheap_current=evaluate(log_approx,current,approx_counts,'initial',True) if mode!='exact' else None
    if log_current is not None and not np.all(np.isfinite(log_current)): raise ValueError('Os inícios precisam ter densidade completa positiva.')
    samples=np.empty((chains,steps-warmup,dimensions))
    accepted=np.zeros(chains);first_passed=np.zeros(chains);window=np.zeros(chains)
    production_start=None
    for step in range(steps):
        if step==warmup: production_start=perf_counter()
        phase='warmup' if step<warmup else 'production'
        noise=np.stack([rng.normal(size=dimensions) for rng in generators])
        proposed=current+scales[:,None]*(noise@factor.T)  # A covariância fica fixa; somente a escala se adapta no aquecimento.
        log_u1=np.log([rng.uniform() for rng in generators])
        log_u2=np.log([rng.uniform() for rng in generators])  # Mantém fluxos comparáveis, mesmo quando a segunda uniforme não é usada.
        valid=inside(proposed)
        move=np.zeros(chains,dtype=bool);passed=np.zeros(chains,dtype=bool)
        if mode=='exact':
            proposed_log=np.full(chains,-np.inf)
            if np.any(valid): proposed_log[valid]=evaluate(log_exact,proposed[valid],exact_counts,phase)
            move=log_u1<np.minimum(0.,proposed_log-log_current)
            log_current[move]=proposed_log[move]  # Guarda o valor no estado aceito para não recalcular o estado atual.
        else:
            proposed_cheap=np.full(chains,-np.inf)
            if np.any(valid): proposed_cheap[valid]=evaluate(log_approx,proposed[valid],approx_counts,phase,True)
            passed=log_u1<np.minimum(0.,proposed_cheap-cheap_current)  # Rejeições nesta etapa não chamam o modelo completo.
            if mode=='approx':
                move=passed
            elif np.any(passed):
                selected=np.flatnonzero(passed)
                proposed_log=evaluate(log_exact,proposed[selected],exact_counts,phase)
                _,log_alpha2=delayed_log_acceptance(proposed_log-log_current[selected],proposed_cheap[selected]-cheap_current[selected])
                accepted2=log_u2[selected]<log_alpha2  # Corrige a triagem para conservar a posterior completa como alvo estacionário.
                move[selected[accepted2]]=True
                log_current[selected[accepted2]]=proposed_log[accepted2]
            cheap_current[move]=proposed_cheap[move]  # Conserva ambos os valores em cache depois de cada decisão.
        current[move]=proposed[move]
        window+=move
        if step<warmup and (step+1)%100==0:
            scales*=np.exp(np.clip(window/100-.28,-.3,.3))  # Ajusta somente no período que será descartado.
            window[:]=0
        if step>=warmup:
            samples[:,step-warmup]=current  # Não remove rejeições nem faz thinning.
            accepted+=move;first_passed+=passed
    elapsed=perf_counter()-beginning
    production=perf_counter()-production_start
    conditional=np.divide(accepted,first_passed,out=np.full(chains,np.nan),where=first_passed>0)
    return SamplingResult(samples,accepted/(steps-warmup),first_passed/(steps-warmup),conditional,scales,
        exact_counts,approx_counts,elapsed,production)
