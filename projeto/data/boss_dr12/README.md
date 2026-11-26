# BOSS DR12: consenso BAO para o Bloco 6

Este diretório contém **dois arquivos originais, sem arredondamento adicional**, do produto de consenso BAO pós-reconstrução de BOSS DR12. A análise não utiliza os produtos de full shape/RSD, nem soma diferentes ajustes da mesma amostra como se fossem independentes.

Referência: Alam et al. (2017), *The clustering of galaxies in the completed SDSS-III Baryon Oscillation Spectroscopic Survey: cosmological analysis of the DR12 galaxy sample*, MNRAS 470, 2617–2652, [arXiv:1607.03155](https://arxiv.org/abs/1607.03155).

- [Página oficial dos produtos](https://sdss3.org/science/boss_publications.php).
- [Arquivo oficial](https://data.sdss.org/sas/dr12/boss/papers/clustering/ALAM_ET_AL_2016_consensus_and_individual_Gaussian_constraints.tar.gz).
- Subdiretório do arquivo: `COMBINEDDR12_BAO_consensus_dM_Hz/`.
- Data de obtenção: 26/11/2026.

| Arquivo | Conteúdo |
| --- | --- |
| `BAO_consensus_results_dM_Hz.txt` | Seis medidas, intercalando distância e taxa nos três redshifts efetivos |
| `BAO_consensus_covtot_dM_Hz.txt` | Covariância total 6 × 6 na mesma ordem |
| `provenance.json` | URLs, membros do arquivo, tamanhos e hashes SHA-256 |
| `baixar_dados.py` | Recuperação explícita dos dois arquivos, com verificação dos hashes |

## Convenção e unidades

A ordem do vetor é

\[
\boldsymbol b=\left(D_M\frac{r_{d,\mathrm{fid}}}{r_d},\ H\frac{r_d}{r_{d,\mathrm{fid}}}\right)_{z=0.38,\,0.51,\,0.61}.
\]

As posições 0, 2 e 4 têm unidade Mpc. As posições 1, 3 e 5 têm unidade km/s/Mpc. Cada elemento da covariância tem o produto das unidades de sua linha e coluna. O arquivo usa os rótulos `dM(rsfid/rs)` e `Hz(rs/rsfid)`; neste projeto escrevemos a escala de arrasto como `r_d`.

O valor `r_d,fid = 147.78 Mpc` pertence à **convenção do produto**, não a uma medida externa do verdadeiro horizonte sonoro. O Bloco 6 deixa livre `B = H0 * r_d / r_d,fid`, em km/s/Mpc. Dessa forma, prevê diretamente o vetor publicado por `(c I/B, B E)`, sem fixar `r_d` e sem chamar `B` de `H0`.

Os bins de redshift se sobrepõem. A análise preserva as correlações entre direções e redshifts da covariância total. Uma figura com barras marginais não significa que seis erros independentes foram usados no ajuste.

Não transformamos a variável radial em seu inverso e reutilizamos erros gaussianos simétricos: essa transformação seria não linear. A transformação linear para `(D_M/r_d, H r_d/c)`, com a covariância também transformada, é verificada em `tests/test_bloco6.py`.

## Procedência e recuperação

O SHA-256 do arquivo oficial é:

```text
23c82f1b17279176167d7de62dac197be36f41c135fa38116f43abec771aac32
```

Os hashes de cada membro estão em `provenance.json` e são conferidos pelo notebook antes da análise. Os dados já acompanham a entrega; a execução do notebook não faz downloads.

Para recuperar explicitamente os dois arquivos, execute da pasta `projeto`:

```bash
python data/boss_dr12/baixar_dados.py
```

O script usa somente a biblioteca padrão. Ele verifica o arquivo comprimido e os dois membros antes de substituir qualquer arquivo local. Não extrai o restante do arquivo, não modifica a procedência e não baixa covariâncias alternativas. Se a equipe atualizar o arquivo oficial, a verificação irá interromper a execução para que a mudança seja investigada.

As hipóteses cosmológicas, a prior, a marginalização da calibração das supernovas e a hipótese de independência SN–BAO estão desenvolvidas no notebook. Este diretório registra o produto de dados, sem converter sua cosmologia fiducial em uma prior adicional.
