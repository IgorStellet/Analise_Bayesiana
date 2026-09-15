# Análise Bayesiana

Material de estudo, exercícios e projeto computacional da disciplina de Estatística Bayesiana do doutorado (CBPF, 2026/2).

O repositório foi pensado para cobrir as seguintes etapas do curso:

1. Reproduzir os exemplos e fazer os exercícios numéricos passados em sala, discutindo sua interpretação física;
2. Fazer atividades da lista de exercícios, com discussões e gráficos para cada exercício;
3. Fazer o projeto referente ao curso, ainda a ser escolhido -- Fischer | Dali vs MCMC ou Inferência bayesiana de uma correlação GW-galáxia em modelos multitraçadores simplficado (com MCMC).

Além dos notebooks, terá uma pasta src com as funções recorrentes em python e que são constantemente utilizadas (isso será essencial para o projeto, mas ajudará também para estudos futuros)
Pretendo fazer um arquivo de teste para verificar as funções implementadas em src.

## Estrutura

```text
Analise_Bayesiana/
├── notebooks/
│   ├── exercicios_slides/   # exemplos e exercícios discutidos em aula
│   ├── listas/              # listas avaliativas da disciplina
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

Para verificar os resultados implementados:

```bash
pytest
```

## Referências principais utilizadas em aula

- Phil Gregory, *Bayesian Logical Data Analysis for the Physical Sciences* — livro-texto e referência principal.
- Roberto Trotta, *Bayesian Methods in Cosmology*.
- Luca Amendola, *Statistical Methods: Lecture Notes*.
- Slides da disciplina, ministrada por Miguel Quartin.

