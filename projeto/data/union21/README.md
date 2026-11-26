# Union2.1 — dados da aplicação de supernovas

Fonte científica: Suzuki et al. (2012), *The Hubble Space Telescope Cluster Supernova Survey: V. Improving the Dark Energy Constraints Above z>1 and Building an Early-Type-Hosted Supernova Sample*, ApJ 746, 85, [arXiv:1105.3470](https://arxiv.org/abs/1105.3470).

Página da equipe responsável: [Supernova Cosmology Project](https://supernova.lbl.gov/Union/). A [descrição oficial](https://supernova.lbl.gov/Union/descriptions.html) identifica as colunas e as covariâncias. Os três arquivos foram obtidos dessa fonte em 26/11/2026. `provenance.json` registra as URLs efetivamente usadas, tamanhos e hashes SHA-256. O sufixo `?download=1` permitiu obter as cópias diretamente do servidor; os hashes identificam exatamente os bytes usados nesta entrega.

| Arquivo | Conteúdo | Tratamento no notebook |
|---|---|---|
| `SCPUnion2.1_mu_vs_z.txt` | 580 linhas e cinco colunas, além do cabeçalho de calibração. | Usa as 580 observações, na ordem original. |
| `SCPUnion2.1_covmat_nosys.txt` | Matriz 580 × 580 sem sistemáticas. | Confere a igualdade com `diag(sigma_i**2)` até a precisão textual publicada. |
| `SCPUnion2.1_covmat_sys.txt` | Matriz 580 × 580 incluindo sistemáticas. | Substitui a matriz estatística em um controle separado. Não é somada novamente à diagonal. |

As colunas da tabela são: nome, redshift, módulo de distância, erro do módulo e probabilidade de a supernova estar em uma hospedeira de baixa massa. A quinta coluna fica disponível para documentação; não é um peso e não gera uma nova correção nesta análise. As correções de padronização já fazem parte do produto publicado.

Nenhuma supernova foi removida e nenhuma linha foi reordenada. Isso preserva o pareamento entre dados e matriz de covariância. Se você alterar a seleção ou ordenar a tabela, deverá aplicar a mesma permutação às linhas e colunas da matriz.

O notebook assume que as covariâncias publicadas são conhecidas e independentes dos parâmetros cosmológicos. Mantém os redshifts como entradas condicionadas e não refaz os ajustes de curvas de luz. Usa um único deslocamento aditivo livre, marginalizado com prior constante. Não reproduz a análise conjunta de todas as nuisances do lançamento oficial.

Há três objetos distintos no Bloco 5:

- **Dados observados:** módulos e erros desta tabela.
- **Curva ajustada:** previsão de wCDM nos parâmetros obtidos ao ajustar os dados observados.
- **Experimento fiducial:** módulos iguais à média em `(Omega_m,w)=(0.285,-1)`, nos mesmos redshifts e com a mesma covariância; não há sorteio de ruído nem simulação de um novo levantamento.

O notebook executa inteiramente com essas cópias locais. O script `baixar_dados.py` é opcional: permite recuperar os mesmos arquivos e exige os hashes registrados antes de substituir qualquer cópia. Uma alteração futura na fonte exigirá revisão da procedência, não a aceitação automática de outros bytes.

Para conferir os arquivos sem rede, execute os testes do projeto ou a célula de leitura do Bloco 5. Para uma recuperação intencional, a partir da raiz do repositório:

```bash
python projeto/data/union21/baixar_dados.py
```

O catálogo histórico foi escolhido para reconstruir metodologicamente a aplicação da Seção 3 e da Figura 1 de [Sellentin, Quartin e Amendola (2014)](https://arxiv.org/abs/1401.6892v3). As priors, o melhor ajuste observado, o tratamento explícito do resíduo, a quarta ordem e os controles são escolhas documentadas deste projeto. A extensão CPL tridimensional do artigo não faz parte desta etapa.
