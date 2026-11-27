# Aulas práticas

| Arquivo | Conteúdo |
|---|---|
| `00_verificar_ambiente.ipynb` | Imports inicialmente solicitados pelo professor para a prática de MCMC. |
| `01_Pratica.ipynb` | Amostragem e aplicações com Cobaya, emcee, GetDist e Astropy. |
| `02_Grid_Method.ipynb` | Integração por malhas, distribuições, contornos e controles interativos. |
| `mymodule.py` e `data/` | Leitura dos dados utilizados na prática cosmológica. |

## Instalação em Linux, macOS ou WSL

Com Python 3.10 ou superior, execute a partir da raiz do repositório:

```bash
bash notebooks/aulas_praticas/instalar.sh
source .venv/bin/activate
python -m jupyterlab
```

O instalador cria ou reaproveita **a `.venv` da raiz**, compartilhada com os slides e as listas. Instala as dependências desta pasta, executa `pip check`, confere os imports de MCMC e grid e registra o kernel **Análise Bayesiana (.venv)** dentro do próprio ambiente. A ativação no terminal é feita depois que o script termina.

Além dos pacotes inicialmente pedidos para MCMC, os requisitos incluem SciPy, SymPy e ipywidgets para grid, e pandas, CAMB e h5py, importados por `mymodule.py`. A instalação completa do repositório também está disponível com `bash instalar.sh`, na raiz. Os controles interativos precisam do notebook aberto em uma interface compatível, como o JupyterLab; a visualização estática do GitHub não executa esses controles.

Para Windows sem WSL, siga o [README central](../../README.md#windows-sem-wsl), trocando o caminho do arquivo de requisitos e do verificador pelos desta pasta.

## Verificação

Para repetir a verificação sem executar a instalação:

```bash
.venv/bin/python notebooks/aulas_praticas/verificar_ambiente.py
```

No VS Code, selecione o interpretador `.venv/bin/python` da raiz. No JupyterLab, selecione **Análise Bayesiana (.venv)**. O caminho exibido por `sys.executable` permite conferir qual ambiente está sendo usado.

Abra os notebooks com o diretório de trabalho nesta pasta, para que os caminhos relativos para `mymodule.py` e `data/` sejam encontrados. Os arquivos de dados devem permanecer disponíveis com os nomes esperados pelo código.

O verificador testa o carregamento dos módulos e símbolos utilizados. A execução das cadeias, os resultados das integrações, a leitura dos dados e a interface dos controles são conferidos nos próprios notebooks.
