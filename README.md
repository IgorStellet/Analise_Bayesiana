# Análise Bayesiana

Material de estudo, exercícios e projeto computacional da disciplina de Estatística Bayesiana do doutorado (CBPF, 2026/2).

O repositório foi pensado para unir três etapas:

1. estudar o formalismo e registrar as interpretações físicas;
2. reproduzir os exemplos e exercícios numericamente;
3. modificar hipóteses, priors e dados para entender o que muda na inferência.

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

O primeiro notebook, `01_capitulo_1.ipynb`, trabalha os exemplos 1.3.4 e 1.4.1 do Gregory: comparação entre dois modelos e atualização da probabilidade de uma doença após um teste positivo.

## Instalação

Requer Python 3.10 ou superior.

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

## Referências principais

- Phil Gregory, *Bayesian Logical Data Analysis for the Physical Sciences* — livro-texto e referência principal.
- Roberto Trotta, *Bayesian Methods in Cosmology*.
- Luca Amendola, *Statistical Methods: Lecture Notes*.
- Slides da disciplina, ministrada por Miguel Quartin.

## Método de trabalho

Cada bloco de estudo seguirá o ciclo: **teoria → reprodução → variação → interpretação**. Os notebooks conterão o código funcional e perguntas-guia; as interpretações finais serão reescritas com a linguagem do autor durante o estudo ativo.
