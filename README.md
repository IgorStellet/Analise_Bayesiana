# Análise Bayesiana

Material de estudo, exercícios e projeto computacional da disciplina de Estatística Bayesiana do doutorado (CBPF, 2026/2).

O repositório foi pensado para cobrir as seguintes etapas do curso:

1. **Estudos dos slides e aulas práticas**: reproduzdir exemplos, explorar os conceitos e discutir suas hipóteses, sua matemática e sua intepretação física, sempre deixando claro tudo que está sendo construído. Quero desenvolver, futuramente, esses notebooks como um material mais fluido e didático, para acompanhar as aulas sequenciamente e servir como um estudo iterativo para quem está vendo a disciplina com partes "hands-on" e exemplos de todos os conceitos abordados;
2. **Listas de exercício**: reune as questões de cada uma das 3 listas de exercícios da disciplina, com cálculos, gráficos e discussões mais profundas e mais bem explicadas do que os notebooks das aulas, visando explorar o máximo de cada questão. As resoluções escritas no Overleaf estão mais completar em discussão, complementando o código feito nos notebooks que tem um carater mais computacional por si sí
3. **Projeto final**: Método de DALI e sua aproximação, discutindo suas vantagens, limitações e aplicações. No momento o projeto ainda será desenvolvido

Os notebooks dos slides e das listas encontram-se na pasta notebooks. O projeto encontra-se na pasta projeto. Nessa última, está incluido os arquivos .py com funções que serão utilizadas de forma recorrente para construir os gráficos e imagens, isso é essencial para o projeto, mas ajudará também para estudos futuros, por tratarem-se de funções mais genéricas que podem ser reutilizadas e são recorrentes em qualquer estudo.

Dentro do projeto também há arquivos testes de validação.

## Estrutura


| Caminho | Conteúdo |
|---|---|
| [`notebooks/exercicios_slides/`](notebooks/exercicios_slides/README.md) | Exemplos e estudos dos slides, com um plano de revisão didática. |
| [`notebooks/aulas_praticas/`](notebooks/aulas_praticas/README.md) | Atividades práticas, amostragem MCMC, método de grid e dados utilizados em aula. |
| [`notebooks/listas/`](notebooks/listas/README.md) | Notebooks das três listas, figuras e tabelas produzidas nas resoluções. |
| [`projeto/`](projeto/README.md) | Espaço do projeto final. Ainda será feito. |
| `scripts/` | Preparação do ambiente compartilhado pelos conjuntos de notebooks. |


## Instalação 

Todos os instaladores utilizam **a mesma `.venv`, na raiz do repositório**, com o nome de exibição **Análise Bayesiana (.venv)** no Jupyter. O primeiro instalador cria o ambiente; os seguintes acrescentam as dependências necessárias dependendo do que queremos rodar.

Os scripts Bash são destinados a Linux, macOS e WSL, com Python 3.10 ou superior e suporte a `venv`.

```bash
git clone https://github.com/IgorStellet/Analise_Bayesiana.git
cd Analise_Bayesiana
```

Escolha qual dos notebooks quer ser rodado:

| O que executar | Comando |
|---|---|
| Apenas os estudos dos slides | `bash notebooks/exercicios_slides/instalar.sh` |
| Slides, incluindo os complementos de MCMC | `bash notebooks/exercicios_slides/instalar.sh mcmc` |
| Listas | `bash notebooks/listas/instalar.sh` |
| Aulas práticas | `bash notebooks/aulas_praticas/instalar.sh` |
| Projeto Fisher e DALI | `bash projeto/instalar.sh` |
| Repositório completo | `bash instalar.sh` |

Cada instalador usa o `requirements.txt` da respectiva pasta, verifica a consistência das dependências com `pip check` e testa os imports. Os requisitos cobrem o conjunto de notebooks daquela pasta e incluem JupyterLab e o kernel, além das dependências indiretas necessárias aos pacotes.

Depois, ative o ambiente e abra o JupyterLab:

```bash
source .venv/bin/activate
python -m jupyterlab
```

No Jupyter, selecione **Análise Bayesiana (.venv)**. No VS Code, selecione o interpretador `.venv/bin/python` da raiz. Abra cada notebook em sua pasta: alguns arquivos usam caminhos relativos para ler dados e salvar figuras ou tabelas.

### Acrescentar outro conjunto

Por exemplo, após estudar os slides:

```bash
bash notebooks/aulas_praticas/instalar.sh
```

O ambiente continua sendo o mesmo. Pacotes já instalados que satisfazem os requisitos são reaproveitados. Versões podem ser ajustadas quando uma nova dependência exigir.

A instalação completa reúne os arquivos de requisitos das quatro partes, incluindo Cobaya e emcee.

### Windows sem WSL

No PowerShell, também é possível usar os mesmos arquivos de requisitos. 

Na primeira instalação, a partir da raiz, crie o ambiente:

```powershell
py -m venv --prompt analise-bayesiana .venv
```

Para instalar e verificar o repositório completo:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe verificar_ambiente.py
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name analise-bayesiana --display-name "Análise Bayesiana (.venv)"
.\.venv\Scripts\python.exe -m jupyterlab
```

Para instalar só uma parte, troque os caminhos dos requisitos e do verificador pelos da pasta escolhida, por exemplo `projeto/requirements.txt` e `projeto/verificar_ambiente.py`. Reaproveite o mesmo Python nas instalações seguintes. No VS Code, selecione `.venv\Scripts\python.exe`. Use a cópia do repositório no sistema escolhido: ambientes Windows e WSL são distintos.

## Verificação

Após a instalação completa, os imports podem ser conferidos novamente sem reinstalar pacotes:

```bash
.venv/bin/python verificar_ambiente.py
```

Para conferir apenas uma parte, execute o `verificar_ambiente.py` daquela pasta. Esses verificadores conferem o ambiente; os testes científicos e as análises estão documentados nos notebooks e no [README do projeto](projeto/README.md#verificação-e-testes).

## To Do - Revisão didática das aulas

Futuramente, é de meu interesse melhorar a organização dos notebooks referente aos slides e aulas práticas, deixando todos em uma única pasta. Idealmente, os notebooks estariam divididos de forma fluida e didática, seguindo as semanas da aula ou os capítulos do Gregory, discutindo os exemplos feitos e os conceitos aprendidos, com exemplos iterativos e variáveis para serem alteradas e com partes "hands-on" para quem estiver acompanhando. Com atividades para os alunos, infelizmente não tive tempo de produzir algo tão organizado, mas seria algo legal de se fazer futuramente, talvez em uma monitoria. Esses notebooks referente aos slides ficaram um pouco "jogados" devido a quantidade de atividades a se fazer.

- [ ] Revisar os notebooks de estudo e integrar teoria e prática em um percurso por capítulos ou semanas, conforme o [plano dos slides](notebooks/exercicios_slides/README.md#to-do--revisão-didática-futura).

## Referências da disciplina

- Phil Gregory, *Bayesian Logical Data Analysis for the Physical Sciences* - Livro texto principal da disciplina.
- Roberto Trotta, *Bayesian Methods in Cosmology*.
- Luca Amendola, *Statistical Methods: Lecture Notes*.
- Slides e listas da disciplina, de Miguel Quartin.

As referências específicas do projeto estão reunidas por bloco ao final de seu notebook.
