"""Oficina visual de DALI: modelos artificiais, séries e relógios deformados.

As imagens científicas vêm de likelihoods explícitas. As marcas do mostrador
e a paisagem da capa são elementos gráficos, identificados no notebook.
As séries guardam coeficientes de Taylor, já divididos pelos fatoriais.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from pathlib import Path
import numpy as np
from scipy.integrate import simpson
from . import dali


# Modelos e coeficientes: as hipóteses probabilísticas ficam visíveis aqui.
# --------------------------------------------------------------------

@dataclass(frozen=True)
class ClockModel:
    """Escolhe a geometria, sem usar dados observacionais.

    ``kind`` é gaussian, ring ou rational. Nos casos gaussian/rational,
    d=(0,0), mu=(x,y-f(x)) e C=diag(sx²,sy²). No aro, d=1,
    mu=(x/sx)²+((y-f(x))/sy)² e a incerteza do dado é tau.
    Os eixos x,y são parâmetros adimensionais; sx,sy são positivos.
    """
    kind: str = 'gaussian'
    amplitude: float = .72
    frequency: float = 2.35
    bend: float = .10
    sx: float = .78
    sy: float = .38
    radius: float = .72
    tau: float = .16

    def __post_init__(self):
        if self.kind not in ('gaussian','ring','rational'):
            raise ValueError('Escolha gaussian, ring ou rational.')
        numbers=np.array([self.amplitude,self.frequency,self.bend,self.sx,self.sy,self.radius,self.tau])
        if not np.all(np.isfinite(numbers)) or min(self.sx,self.sy,self.radius,self.tau)<=0:
            raise ValueError('Use parâmetros finitos e escalas estritamente positivas.')

    def warp(self,x):
        """Define a curva que dobra o mostrador no plano físico x,y."""
        x=np.asarray(x,dtype=float)
        if self.kind=='rational':
            return self.amplitude*x*x/(self.radius**2+x*x)  # A função é suave no eixo real, mas possui polos complexos.
        return self.amplitude*np.sin(self.frequency*x)+self.bend*x*x  # A senoide muda a forma; o termo quadrático quebra a simetria.

    def warp_coefficients(self,x0,order):
        """Obtém f^(n)(x0)/n!, evitando diferenças finitas de ordem alta."""
        _check_order(order)
        if self.kind=='rational':
            denominator=np.zeros(order+1)
            denominator[:min(3,order+1)]=[self.radius**2+x0*x0,2*x0,1.][:min(3,order+1)]
            inverse=np.zeros(order+1);inverse[0]=1/denominator[0]
            for n in range(1,order+1):
                inverse[n]=-sum(denominator[j]*inverse[n-j] for j in range(1,min(n,2)+1))/denominator[0]  # Resolve (rho²+x²)*série=1, coeficiente a coeficiente.
            result=-self.amplitude*self.radius**2*inverse
            result[0]+=self.amplitude  # Usa f=a-a*rho²/(rho²+x²).
            return result
        n=np.arange(order+1)
        result=self.amplitude*self.frequency**n*np.sin(self.frequency*x0+n*np.pi/2)/np.array([factorial(int(k)) for k in n])
        result[0]+=self.bend*x0*x0
        if order>=1: result[1]+=2*self.bend*x0
        if order>=2: result[2]+=self.bend  # O coeficiente de delta x² já inclui o fator 1/2!.
        result[np.abs(result)<1e-16]=0.  # Limpa apenas resíduos de ponto flutuante das simetrias trigonométricas.
        return result

    def residual(self,points):
        """Retorna C^-1/2*(mu-d), o resíduo cujo quadrado define a energia."""
        points=np.asarray(points,dtype=float)
        if points.shape[-1]!=2:
            raise ValueError('O último eixo deve conter x,y.')
        x=points[...,0]/self.sx
        y=(points[...,1]-self.warp(points[...,0]))/self.sy
        if self.kind=='ring':
            return ((x*x+y*y-1)/self.tau)[...,None]  # Um único dado gaussiano deixa uma curva inteira de soluções.
        return np.stack((x,y),axis=-1)  # Dois resíduos gaussianos geram a posterior curva do relógio preenchido.

    def center(self,angle=np.pi/2):
        """Escolhe um MAP conhecido; o aro permite trocar sua posição angular."""
        if self.kind!='ring': return np.zeros(2)
        x=self.sx*np.cos(angle)
        return np.array([x,self.warp(x)+self.sy*np.sin(angle)])  # Fica sobre mu=1, sem procurar um máximo por otimização.

    def coefficients(self,center,order):
        """Monta o Taylor multivariado do resíduo, com grau TOTAL até order.

        Cada chave (a,b) multiplica delta x**a * delta y**b. Os fatores
        mistos aparecem ao multiplicar as séries; não se retêm termos de
        grau maior só porque vieram de um produto conveniente.
        """
        _check_order(order)
        x0,y0=np.asarray(center,dtype=float)
        f=self.warp_coefficients(x0,order)
        first={(0,0):x0/self.sx,(1,0):1/self.sx}
        second={(n,0):-value/self.sy for n,value in enumerate(f)}
        second[0,0]+=y0/self.sy;second[0,1]=1/self.sy
        if self.kind!='ring': return first,second
        ring=_add(_multiply(first,first,order),_multiply(second,second,order))
        ring[0,0]=ring.get((0,0),0.)-1
        return ({key:value/self.tau for key,value in ring.items()},)  # O ruído do dado radial pondera o Taylor inteiro.


def _check_order(order):
    """Limita a oficina às ordens que os controles e as figuras apresentam."""
    if not isinstance(order,(int,np.integer)) or not 1<=order<=16:
        raise ValueError('A ordem deve ser um inteiro entre 1 e 16.')


def _add(first,second):
    """Soma duas séries esparsas, preservando as chaves dos termos mistos."""
    result=first.copy()
    for key,value in second.items(): result[key]=result.get(key,0.)+value
    return result


def _multiply(first,second,order):
    """Multiplica séries e descarta termos acima do grau total declarado."""
    result={}
    for (a,b),u in first.items():
        for (c,d),v in second.items():
            if a+b+c+d<=order:
                key=(a+c,b+d)
                result[key]=result.get(key,0.)+u*v  # Inclui automaticamente as duas contribuições de um termo cruzado.
    return result


def evaluate_series(coefficients,points,center):
    """Avalia os componentes do modelo truncado, sem expandir sua energia."""
    points=np.asarray(points,dtype=float)
    delta=points-np.asarray(center,dtype=float)
    order=max(a+b for component in coefficients for a,b in component)
    xp=[np.ones(points.shape[:-1])];yp=[np.ones(points.shape[:-1])]
    for _ in range(order):
        xp.append(xp[-1]*delta[...,0]);yp.append(yp[-1]*delta[...,1])  # Reutiliza potências entre todos os componentes.
    result=[]
    for component in coefficients:
        value=np.zeros(points.shape[:-1])
        for (a,b),weight in component.items(): value+=weight*xp[a]*yp[b]
        result.append(value)
    return np.stack(result,axis=-1)


def log_kernel(model,points,order=None,center=None,blend=0.):
    """Calcula -E; blend interpola MODELOS entre K e K+1 para a animação.

    blend=0 é DALI de ordem K. Valores interiores são uma homotopia visual,
    não recebem o nome de uma ordem fracionária de DALI. A prior uniforme
    é aplicada pelo domínio da malha, em todas as comparações.
    """
    if order is None:
        residual=model.residual(points)  # A referência usa o modelo completo.
    else:
        center=model.center() if center is None else np.asarray(center)
        residual=evaluate_series(model.coefficients(center,order),points,center)
        if blend:
            if not 0<=blend<=1 or order==16: raise ValueError('Use blend entre zero e um e ordem abaixo de 16.')
            next_residual=evaluate_series(model.coefficients(center,order+1),points,center)
            residual=(1-blend)*residual+blend*next_residual  # Interpola antes de elevar ao quadrado, conservando E>=0.
    return -.5*np.sum(residual*residual,axis=-1)


def density(model,axes,order=None,center=None,blend=0.):
    """Normaliza a posterior no mesmo retângulo e na medida dx dy."""
    x,y=np.meshgrid(*axes)
    values=log_kernel(model,np.stack((x,y),axis=-1),order,center,blend)
    return dali.normalize_log_grid(values,*axes)[0]


def total_variation(first,second,axes):
    """Mede o erro das densidades completas, sem renormalizar cortes."""
    return .5*dali.integrate_grid_2d(abs(first-second),*axes)


def fisher(model,center=None):
    """Obtém J^T J; o aro mantém uma direção sem informação local."""
    center=model.center() if center is None else center
    coefficients=model.coefficients(center,1)
    jacobian=np.array([[c.get((1,0),0.),c.get((0,1),0.)] for c in coefficients])
    return jacobian.T@jacobian


def mixture(densities,weights):
    """Combina componentes JÁ normalizados, com pesos conhecidos da oficina.

    Esta construção por componentes não é uma Taylor global da mistura.
    Não infere pesos ou descobre modos a partir de um único centro.
    """
    weights=np.asarray(weights,dtype=float)
    values=np.asarray(densities,dtype=float)
    if weights.ndim!=1 or len(weights)!=len(values) or np.any(weights<0) or not np.isclose(weights.sum(),1):
        raise ValueError('Forneça pesos não negativos cuja soma seja um.')
    return np.einsum('i,ijk->jk',weights,values)


def placed_density(model,axes,offset=(0.,0.),rotation=0.,scale=1.,order=None):
    """Posiciona uma componente por uma transformação afim conhecida.

    O jacobiano da transformação é constante e cancela ao normalizar
    cada componente na mesma caixa física. Os pesos da mistura são
    definidos depois dessa normalização, não antes dela.
    """
    if not np.isfinite(scale) or scale<=0: raise ValueError('A escala deve ser positiva.')
    x,y=np.meshgrid(*axes)
    first=(x-offset[0])/scale;second=(y-offset[1])/scale
    cosine,sine=np.cos(rotation),np.sin(rotation)
    points=np.stack((cosine*first+sine*second,-sine*first+cosine*second),axis=-1)  # Aplica a inversa da rotação e da escala.
    values=log_kernel(model,points,order)
    return dali.normalize_log_grid(values,*axes)[0]


# Figuras: cada efeito de apresentação é separado da densidade científica.
# ---------------------------------------------------------------------

NIGHT='#101e2a'
GOLD='#e9bd76'
TEAL='#70c4b6'
PAPER='#f5eddf'


def palette(name='teal'):
    """Escolhe cores com fundo escuro e uma região de alta densidade legível."""
    from matplotlib.colors import LinearSegmentedColormap
    sets={'teal':[NIGHT,'#164756','#2b8d95','#9bd3b6','#fff0c2'],
          'gold':[NIGHT,'#574232','#b27c43','#e8b66a','#fff0c2']}
    return LinearSegmentedColormap.from_list('oficina_'+name,sets[name])


def draw_density(ax,pdf,axes,title='',vmax=None,cmap='teal',contours=False):
    """Mostra p(x,y) em escala linear, com o máximo comum escolhido pelo leitor."""
    image=ax.imshow(pdf,origin='lower',extent=[axes[0][0],axes[0][-1],axes[1][0],axes[1][-1]],
        cmap=palette(cmap),vmin=0,vmax=vmax,aspect='equal',interpolation='bilinear')
    if contours:
        levels=np.sort(dali.hpd_thresholds_2d(pdf,*axes,(.6827,.9545)))
        ax.contour(*axes,pdf,levels=levels,colors=PAPER,linewidths=.9,linestyles=('--','-'),alpha=.8)
    ax.set(xlabel=r'$x$',ylabel=r'$y$',title=title)
    ax.set_facecolor(NIGHT)
    return image


def style_figure(fig):
    """Mantém a oficina com identidade visual própria sem alterar outros blocos."""
    fig.set_facecolor(PAPER)
    for ax in fig.axes:
        ax.spines[['top','right']].set_visible(False)


def draw_clock(ax,model,position=(0.,0.),scale=1.,rotation=0.,color=GOLD,level=1.75,fill=True):
    """Desenha o mostrador pela MESMA transformação que deforma a posterior.

    Os números e ponteiros orientam a analogia, mas não são dados,
    contornos HPD ou estruturas que DALI tenha inferido.
    """
    from matplotlib.patches import Polygon,Circle
    cosine,sine=np.cos(rotation),np.sin(rotation)
    def transform(u,v):
        x=model.sx*np.asarray(u);y=model.warp(x)+model.sy*np.asarray(v)
        return position[0]+scale*(cosine*x-sine*y),position[1]+scale*(sine*x+cosine*y)
    angle=np.linspace(0,2*np.pi,700)
    x,y=transform(level*np.cos(angle),level*np.sin(angle))
    if fill:
        ax.add_patch(Polygon(np.c_[x,y],closed=True,facecolor='#f1dfb9',edgecolor='none',alpha=.95,zorder=4))
    ax.plot(x,y,color=color,lw=2.4,zorder=7)
    inner=level*.94
    ax.plot(*transform(inner*np.cos(angle),inner*np.sin(angle)),color='#786043',lw=.7,alpha=.8,zorder=7)
    for tick in range(60):
        t=np.pi/2-tick*2*np.pi/60
        short=.90 if tick%5 else .83
        ax.plot(*transform(level*np.cos(t)*np.array([short,.96]),level*np.sin(t)*np.array([short,.96])),
                color='#4d5557',lw=(.65 if tick%5 else 1.1),zorder=8)
    for number in (12,3,6,9):
        t=np.pi/2-(number%12)*2*np.pi/12
        px,py=transform(level*.68*np.cos(t),level*.68*np.sin(t))
        ax.text(px,py,str(number),ha='center',va='center',fontsize=10*max(.8,scale),color='#344652',fontfamily='DejaVu Serif',zorder=8)
    for length,t in ((level*.62,1.1),(level*.43,3.8)):
        radial=np.linspace(0,length,100)
        ax.plot(*transform(radial*np.cos(t),radial*np.sin(t)),color='#334e59',lw=2.2,zorder=9)  # Até o ponteiro acompanha a dobra; não supomos uma haste rígida.
    px,py=transform(0.,0.)
    ax.add_patch(Circle((px,py),.035*scale,color=color,zorder=10))
    return transform


def cover(destination,with_text=True):
    """Compõe uma capa original com relógios matemáticos e paisagem decorativa.

    A capa é uma montagem gráfica. Não representa uma única posterior e
    não é uma reprodução digital de uma pintura de Dalí.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Polygon,Ellipse
    fig,ax=plt.subplots(figsize=(16,8.4),dpi=180)
    fig.subplots_adjust(0,0,1,1);ax.set(xlim=(0,16),ylim=(0,8.4));ax.axis('off')
    sky=np.tile(np.linspace(0,1,600)[:,None],(1,1200))
    sky_colors=LinearSegmentedColormap.from_list('horizonte',['#ceba8d','#849b9b','#35546a','#101e2a'])
    ax.imshow(sky,origin='lower',extent=(0,16,0,8.4),cmap=sky_colors,aspect='auto',zorder=0)
    ax.add_patch(Polygon([(0,0),(16,0),(16,3.25),(0,2.8)],color='#b49b70',zorder=1))
    ax.add_patch(Polygon([(10.5,3.1),(11.3,3.65),(12.6,3.8),(13.5,4.7),(14.8,4.9),(16,4.3),(16,2.9)],facecolor='#ceab6e',edgecolor='none',zorder=1))
    ax.add_patch(Polygon([(12,3.1),(13.4,4.35),(14.2,4.48),(14.8,4.9),(15.2,4.0),(16,3.7),(16,2.9)],facecolor='#8e7e60',edgecolor='none',alpha=.55,zorder=2))
    ax.plot([0,11],[2.84,3.17],color='#ede0b6',lw=.8,alpha=.7,zorder=2)
    ax.add_patch(Polygon([(0,0),(6.0,0),(5.2,2.7),(0,2.7)],facecolor='#596364',edgecolor='none',zorder=2))
    ax.add_patch(Polygon([(0,2.7),(5.2,2.7),(5.8,3.03),(.8,3.03)],facecolor='#b9b295',edgecolor='none',zorder=3))
    # As sombras pertencem à montagem; não são massa de probabilidade.
    for x,y,w,h in ((4.25,1.05,5.2,.48),(10.75,.72,6.0,.52),(10.0,3.02,2.6,.20)):
        for k in range(9):
            ax.add_patch(Ellipse((x,y),w*(1-k*.055),h*(1-k*.055),color=NIGHT,alpha=.015,zorder=3))
    clock1=ClockModel(amplitude=.58,frequency=2.4,bend=-.08,sx=.84,sy=.48)
    clock2=ClockModel(amplitude=.68,frequency=2.15,bend=.12,sx=.9,sy=.43)
    clock3=ClockModel(amplitude=.35,frequency=2.6,bend=-.15,sx=.7,sy=.40)
    draw_clock(ax,clock1,(3.75,2.35),1.55,-.18,GOLD)
    draw_clock(ax,clock2,(10.8,1.72),1.72,.05,TEAL)
    draw_clock(ax,clock3,(9.55,3.65),.79,-.45,'#c9d7cc')
    # Pequenos contornos à direita lembram a passagem de elipses para formas curvas.
    for radius,alpha in ((1.0,.28),(.78,.36),(.56,.46)):
        t=np.linspace(0,2*np.pi,500);x=radius*np.cos(t)
        ax.plot(13.6+.95*x,6.55+.65*radius*np.sin(t)+.22*np.sin(2.5*x),color=TEAL,alpha=alpha,lw=1.1,zorder=4)
    if with_text:
        ax.text(.78,7.65,'OFICINA VII  /  FISHER → DALI',color=GOLD,fontsize=10,fontweight='bold',va='center',zorder=12)
        ax.text(.75,6.96,'A persistência da',color=PAPER,fontsize=33,fontfamily='DejaVu Serif',va='top',zorder=12)
        ax.text(.75,6.18,'não gaussianidade',color=PAPER,fontsize=33,fontfamily='DejaVu Serif',va='top',zorder=12)
        ax.text(.80,5.18,'Relógios de probabilidade. Uma expansão de cada vez.',color='#c5d7d6',fontsize=11,zorder=12)
        ax.text(.80,.30,'IGOR STELLET  ·  ESTATÍSTICA BAYESIANA  ·  2026',color='#dfdacb',fontsize=8,zorder=12)
        ax.text(15.35,.30,'COMPOSIÇÃO MATEMÁTICA ORIGINAL',color='#e9e1ce',fontsize=7,ha='right',zorder=12)
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,dpi=180,facecolor=fig.get_facecolor())
    if destination.suffix=='.png': fig.savefig(destination.with_suffix('.svg'),facecolor=fig.get_facecolor())
    plt.close(fig)
    return destination


def export_explorer(destination):
    """Copia a oficina interativa autônoma; ela não exige servidor ou internet."""
    template=Path(__file__).with_name('oficina_visual.html')
    target=Path(destination);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(template.read_text(encoding='utf-8'),encoding='utf-8')
    return target


def save_animation(model,axes,destination,orders=range(1,17)):
    """Exporta uma sequência discreta de ordens, sempre com a mesma escala de cor."""
    import io
    import matplotlib.pyplot as plt
    from PIL import Image
    reference=density(model,axes)
    center=model.center();frames=[]
    for order in orders:
        approx=density(model,axes,order,center)
        tv=total_variation(reference,approx,axes)
        fig,axs=plt.subplots(1,2,figsize=(9,4.5),layout='constrained')
        for ax,pdf,title in zip(axs,(reference,approx),('Modelo completo',f'DALI K={order}  |  TV={tv:.4f}')):
            draw_density(ax,pdf,axes,title,vmax=reference.max(),contours=True)
        fig.suptitle('A persistência da não gaussianidade',fontfamily='DejaVu Serif',fontsize=14)
        style_figure(fig)
        buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=95);plt.close(fig)
        with Image.open(buffer) as image: frames.append(image.convert('RGB').copy())
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    frames[0].save(destination,save_all=True,append_images=frames[1:],duration=850,loop=0)
    return destination
