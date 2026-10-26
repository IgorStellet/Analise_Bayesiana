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

As pastas são criadas automaticamente. Os arquivos antigos diretamente em `figuras/` e `tabelas/` permanecem disponíveis até sua limpeza manual; ao executar novamente, as saídas serão escritas nas subpastas da respectiva lista. Para listas futuras, siga a mesma convenção `lista_3`, `lista_4`, etc. Ao levar os arquivos ao Overleaf, ajuste os caminhos de `includegraphics` e `input` conforme as novas subpastas.

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

**Questão 6:** --



