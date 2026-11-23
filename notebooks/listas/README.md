# Listas

Será feito um notebook para cada lista. 

O objetivo é fazer os gráficos, tabelas e contas de cada questão para poder escrever mais facilmente no overleaf, além de servir como um estudo de como programar.

As discussões no notebook serão mais diretas objetivas e curtas, a discussão completa abrindo as contas será reservada para o PDF com as resoluções (overleaf).

## Organização das saídas

Execute os notebooks com o kernel na pasta `notebooks/listas`. Cada lista cria suas próprias subpastas, preservando os nomes das figuras e tabelas:

| Notebook | Figuras PDF/PNG | Tabelas CSV/LaTeX |
|:--|:--|:--|
| `01_lista_1.ipynb` | `figuras/lista_1/` | `tabelas/lista_1/` |
| `02_lista_2.ipynb` | `figuras/lista_2/` | `tabelas/lista_2/` |
| `03_lista_3.ipynb` | `figuras/lista_3/` | `tabelas/lista_3/` |


## Lista 1.

**Questão 1:** Entendimento qualitativo sobre Likelihood, priors, evidência e posterior analisando os dados sobre um teste de uma doença.

**Questão 2:** Discussão sobre posterior conjunta, marginalização de variáveis e sombra de regiões. Posterior dada, foco é no entendimento sobre marginalização e modificação do best fit ao marginalizarmos.

**Questão 3:** Probabilidade de aniversários no mesmo dia. Questão para aprender a trabalhar com a hipótese complementar.

**Questão 4:** Probabilidade em jogos de dados. Questão que ajuda entender o teorema de Bayes para casos frequentistas e a trabalhar com probabilidades como um todo.

**Questão 5:** Problema do prêmio de Monty Hall. Novamente bom problema para entender a aplicação do teorema de Bayes para um caso cuja intuição é falha.

**Questão 6 e 7:** Problema do Gregory para aprendermos a aplicar o estudo de Bayes a dados reais, junta a ideia de todas as questões acimas, incluindo marginalização e posterior, navalha de Occam e fator de Bayes. Questão bem completa para introduzir Bayes.


## Lista 2.

**Questão 1:** Distribuição Binomial virando gaussiana para o exemplo de rolagem de dados. Entendimento de como essas distribuições afetam a posterior e o tratamento de diferentes prior para marginalizar quando temos um espaço de hipóteses.

**Questão 2:** Distribuição de Cauchy/Lorentziana. Como essa distribuição aparece para um determinado problema, como aborda-la e o tratamento que fazemos nos dados. Além disso, na questão observamos como o efeito de um número grande de medições afunilam sua posterior para um determinado valor.

**Questão 3:** Distribuição Beta e sua aplicação quando temos que marginalizar sobre um parâmetro da Likelihood!. Similar a questão 1, porém a versão contínua da marginalização e o tratamento dessa prior.

**Questão 4:** Distrbuição de uma amostragem. Trabalhamos com uma estatística de segundo nível, isto é, temos uma estatística já encontrada de uma variável aleatória, $X_i$ e queremos entender como a distribuição de sua média $\bar X_n$ se comporta. A questão discute essencialmente o TCL a passagem da distribuição Beta para gaussiana e o tratamento dessa estatística da média.

**Questão 5:** Distribuição $S^2$, $F$ e $\chi^2$. Nessa questão, a partir de dados, vemos como podemos encontrar o desvio padrão de forma amostral e não enviesada. No tem $b$ interpretamos p-value. No item $c$ discutimos montamos a distribuição F e interpretamos seu significado. Essa questão é um followup da questão anterior, trabalhando com estatística de segundo nível e suas interpretações.

**Questão 6:** Tratamento de dados do COVID 19 e como inferir a prevalência de uma cidade a partir dos dados obtidos do teste. Mostramos também como a distribuição aproximada torna-se ruim para posterior com probabilidades altas para prevalências altas. No desafioo mostramos como podemos utilizar dois resultados para obter a prevalência total do país.



## Lista 3.

**Questão 1:** Estatística Frequentista, tratamento de duas medições diferentes e teste de hipóteses nulas: Mesma média e desvio padrão, mesma média com desvios padrões diferentes e assim por diante. Problema interessante para aprender como abordar dois conjuntos de amostras e o tratamento da estatística frequentista junto ao teste de hipótese. Acho a questão ainda mais interessante do ponto de vista Bayesiano, porém não nos foi pedido isso nessa questão. No overleaf a discussão está completa para o caso frequentista, no notebook dessa lista há ainda uma extensão bayesiana, mas que não foi totalmente trabalhada. Distribuição student e F estão presentes nessa questão. Também somos obrigados a escolher nessa questão um limiar de corte para discartar uma hipótese e utilizar o teste unilateral ou bilateral

**Questão 2:** Estatística Frequentista, teste de Pearson. Aproximação dos dados a uma multinomial e como testar se a distribuição que acreditamos ser de fato é compatível. Na questão em particular assumimos que os dados seguem uma distribuição Poissoininana e queremos usar o teste de Pearson para verificar essa hipótese, ainda dentro da análise frequentista. Essa questão ajuda a entender como escolher corretamente os bins e como tratar os dados.

**Questão 3:** Estatística Bayesiana, intervalos HDI e escolhas de prior. Nessa questão haviliamos uma tabela e fazemos uma estimativa de valores para a média. Essa questão nos diz como tratar de forma Bayesiana o que foi discutido na questão 1, mostrando também como diferentes escolhas de priors afetam sua posterior, para uma prior larga e uma prior restritiva, mostrando também como a região de confiança muda a tomar diferentes escohas.

**Questão 4:**  Nessa questão nos é dado diretamente uma likelihood e queremos basicamente utilizar o método de Grid aprendido em sala e nas aulas práticas. A partir de uma likelihood multidimensional queremos ver como diferentes cortes afetam a likelihood, como visualizá-la e qual a diferença entre cortar em diferentes valores, no best-fit e também marginalizando sobre uma variável - nesta última perdemos informação. Nessa questão observamos também a multimodalidade e como isso aparece e podemos tentar explicar seu motivo através da forma da likelihood vendo qual é sua tendência e onde se encontram os máximos de cada exponencial. Também exploramos a regiões HDI utilizando o método de grid.

**Questão 5:** Nessa questão queremos ver como, a partir da likelihood da questão 4, a aproximação de Fisher se comporta. Nela observamos que sua aproximação não é tão boa para a likelihood dada, o que indica não gaussianidade nos dados. Escrevendo agora...após entregar a lista...percebo que a variável y possui uma distribuição mais platykurtica (coeficiente de kurtoses < 3). A variável x possui Skewness negativa, ou seja, assimetria. A variável w possui bimodalidade. Logo em todas as variáveis temos a presença da não gaussianidade, o que já era um indicativo de que a matriz de fisher não seria uma boa aproximação para nenhuma das nossas variáveis. 