# Estudos dos slides e das aulas

Estes notebooks reúnem reproduções dos exemplos e explorações dos conceitos apresentados na disciplina. O objetivo é utilizá-los para entender como as escolhas de modelo, prior, dados e método numérico aparecem nos resultados, além de ser uma maneira de explorar e acompanhar o que se é passado em aula. Alguns deles ficaram extremamente confusos, mas serviu para visualizar melhor tudo que se era apresentado.

Ao longo do estudo, procuro deixar explícitos:

- o significado das quantidades probabilísticas e a notação utilizada;
- as hipóteses necessárias a cada cálculo;
- a relação entre a dedução, o código e a figura produzida;
- o que muda ao variar os dados, a prior ou uma aproximação;
- a interpretação física ou observacional do resultado e os limites do método.

O material foi desenvolvido em momentos diferentes do curso e ainda precisa de uma revisão pessoal de texto, código e encadeamento, sobretudo nos tópicos finais no qual tive menos tempo para organizar. Os notebooks servem como registros de estudo; tornar sua leitura mais gradual e didática é uma etapa futura.

## Organização atual

| Notebook | Tema principal |
|---|---|
| [`01_capitulo_1.ipynb`](01_capitulo_1.ipynb) | Papel das probabilidades na ciência. |
| [`02_capitulo_3.ipynb`](02_capitulo_3.ipynb) | Inferência bayesiana na prática. |
| [`03_capitulo_4.ipynb`](03_capitulo_4.ipynb) | Atribuição de probabilidades e construção de likelihoods. |
| [`04_capitulo_5.ipynb`](04_capitulo_5.ipynb) | Momentos, distribuições e teorema do limite central. |
| [`05_capitulo6.ipynb`](05_capitulo6.ipynb) | Estatísticas, distribuições amostrais e intervalos. |
| [`06_capitulo7.ipynb`](06_capitulo7.ipynb) | Testes de hipóteses frequentistas. |
| [`07_capitulo9.ipynb`](07_capitulo9.ipynb) | Inferência bayesiana com erros gaussianos. |
| [`08_capitulo_10.ipynb`](08_capitulo_10.ipynb) | Ajustes lineares, covariância e geometria da informação. |
| [`09_capitulo_11_MCMC.ipynb`](09_capitulo_11_MCMC.ipynb) | Modelos não lineares, integração numérica e MCMC. |
| [`10_laplace_profile_DALI.ipynb`](10_laplace_profile_DALI.ipynb) | Laplace, profile likelihood e DALI. |
| [`11_informacao_robustez.ipynb`](11_informacao_robustez.ipynb) | Ganho de informação, concordância e robustez. |

As [aulas práticas](../aulas_praticas/README.md) complementam esse conteúdo, especialmente com MCMC e o método de grid. 

## Preparação do ambiente

A partir da raiz do repositório:

```bash
bash notebooks/exercicios_slides/instalar.sh
source .venv/bin/activate
python -m jupyterlab
```

O instalador utiliza a `.venv` da raiz e instala as dependências do conjunto dos slides: NumPy, SciPy, Matplotlib, pandas e IPython, além de JupyterLab e ipykernel. O mesmo ambiente é reaproveitado pelos instaladores das listas, das práticas e do projeto. Para instalar todo o repositório, use `bash instalar.sh` na raiz. Para Windows sem WSL, veja o [README central](../../README.md#windows-sem-wsl).

O núcleo do notebook de MCMC pode ser estudado sem instalar os pacotes dos complementos opcionais. Para executar também os exemplos com Cobaya e emcee:

```bash
bash notebooks/exercicios_slides/instalar.sh mcmc
```

Essa opção acrescenta Cobaya e emcee à mesma `.venv`, juntamente com suas dependências. Se esses pacotes já estiverem instalados pelas aulas práticas, serão reaproveitados.

## To do — revisão didática futura

O objetivo é transformar este material em um percurso de estudo que permita acompanhar as aulas, retomar os conceitos aprendidos com mais autonomia e utilizar o código como visualização e também para aprendizado, com exercícios iterativos e partes "hands-on". Esta seção registra o plano; a reorganização dos notebooks ainda será feita.

- [ ] **Integrar teoria e prática no percurso de leitura.** Associar a cada tópico os exemplos dos slides, a atividade prática correspondente e exercícios de exploração, incluindo MCMC. Incluir perguntas iterativas e variáveis que podem ser mudadas em cada exemplo para avaliarmos
- [ ] **Reorganizar a divisão dos notebooks por capítulos ou semanas de aula.** Ajustar o tamanho dos blocos para que possam ser estudados aos poucos, com pré-requisitos claros e uma sequência de leitura que acompanhe o calendário da disciplina
- [ ] **Preparar um calendário de estudo.** Para cada semana ou unidade, indicar as referências, os trechos de leitura, os notebooks, a prática e uma estimativa de tempo. Conferir o alinhamento com o cronograma e distinguir a numeração das aulas daquela das referências.
- [ ] **Revisar as deduções e definir a notação antes de usá-la.** Explicar de onde vem cada expressão e quais hipóteses permitem aplicá-la, incluindo casos em que a aproximação falha.
- [ ] **Acrescentar perguntas-guia ao longo dos tópicos.** Pedir uma previsão antes de executar o código e retomar essa previsão depois do resultado. Exemplos: “O que espero que aconteça ao estreitar a prior?”, “Que informação se perde ao marginalizar?” e “Por que esta cadeia pode parecer convergida e ainda não explorar todos os modos?”, além de outras perguntas que permitem mudar o código 
- [ ] **Construir funções de exploração com parâmetros explícitos.** Permitir variar tamanho da amostra, ruído, prior, correlação, limites da malha e configurações do amostrador. Documentar os argumentos, o retorno e o significado estatístico de cada escolha.
- [ ] **Adicionar atividades interativas quando ajudarem a entender o conceito.** Usar controles para comparar distribuições, acompanhar atualizações da posterior e testar aproximações. Manter uma figura estática e instruções equivalentes para quem ler pelo GitHub.
- [ ] **Separar exemplos iniciais de aprofundamentos.** Começar com uma pergunta e um caso simples; apresentar extensões depois que as ferramentas necessárias tiverem sido definidas.
- [ ] **Incluir exercícios curtos e sínteses por unidade.** Retomar as hipóteses, os resultados e as limitações, com sugestões do que o leitor pode modificar e investigar sozinho.
- [ ] **Revisar a execução e a apresentação dos arquivos.** Conferir caminhos, sementes, tabelas, legendas e custos de execução, além de verificar se as figuras sustentam a discussão.
- [ ] **Reescrever e validar o material como um todo.** Conferir os exemplos com as referências, reconstruir os argumentos com minhas próprias palavras e manter uma identidade consistente no texto e no código.

Como referência para essa revisão, cada unidade poderá seguir o percurso: **pergunta física ou estatística → hipóteses → dedução → implementação → exploração → interpretação e limites**. A ideia também poderá servir de base para uma futura monitoria da disciplina.
