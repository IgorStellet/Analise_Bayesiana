# Análise Bayesiana

Material de estudo, exercícios e projeto computacional da disciplina de Estatística Bayesiana do doutorado (CBPF, 2026/2).

O repositório foi pensado para cobrir as seguintes etapas do curso:

1. Reproduzir os exercícios e estudos feitos em sala, focando no que foi passado nos slides, tentando discutir a física e a matemática em cada um deles. O objetivo é que cada conceito possa ser explorado e que esses notebooks sirvam como uma maneira de também acompanhar as aulas e aprofundar o conteúdo. Um objetivo futuro é torná-los mais fluidos, didáticos e iterativos para os usuários;
2. Fazer as listas de exercícios (3 ao todo), mais explicadas e de didáticas do que os notebooks da aula, com interpretações claras e gráficos que explorem o máximo de cada questão. As explicações mais didáticas encontram-se no overleaf de cada lista;
3. Principal: Projeto final da disciplina. Fisher DALI como aproximação para posterior. Suas vantagens, limitações e aplicações. 

Os notebooks dos slides e das listas encontram-se na pasta notebooks. O projeto assim encontra-se na pasta projeto, nela será feita tudo referente ao projeto, incluindo os arquivos .py com funções que serão utilizadas de forma recorrente para construir os gráficos e imagens, constantemente utilizadas (isso será essencial para o projeto, mas ajudará também para estudos futuros, por tratarem-se de funções mais genéricas que podem ser reutilizadas).

Dentro do projeto também pretendo fazer arquivos de teste para validar se tudo está sendo feito de forma coerente.

## Estrutura

```text
Analise_Bayesiana/
├── notebooks/
│   ├── exercicios_slides/   # exemplos e exercícios discutidos em aula
│   ├── listas/              # listas avaliativas da disciplina
│   ├── aulas_praticas/      # notebooks de aula, instalador e teste do ambiente
│   └── projeto/             # projeto computacional final
├── src/analise_bayesiana/   # funções reutilizáveis em Python
└── tests/                   # testes dos resultados numéricos
```

## Instalação

```bash
git clone https://github.com/IgorStellet/Analise_Bayesiana.git
cd Analise_Bayesiana
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
jupyter lab
```

No Windows, a ativação do ambiente virtual é feita com `.venv\Scripts\activate`.

Para preparar apenas o ambiente das aulas práticas (Cobaya, GetDist, emcee, Astropy e os demais imports solicitados), veja [notebooks/aulas_praticas](notebooks/aulas_praticas/README.md). A instalação é independente da instalação geral acima.

Para verificar os resultados implementados:

```bash
pytest
```

## Referências principais utilizadas em aula

- Phil Gregory, *Bayesian Logical Data Analysis for the Physical Sciences* — livro-texto e referência principal.
- Roberto Trotta, *Bayesian Methods in Cosmology*.
- Luca Amendola, *Statistical Methods: Lecture Notes*.
- Slides da disciplina, ministrada por Miguel Quartin.

