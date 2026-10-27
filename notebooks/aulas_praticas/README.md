# Aulas práticas

Notebooks trabalhados em aula e preparação do ambiente computacional. O arquivo `00_verificar_ambiente.ipynb` contém os imports solicitados pelo professor: NumPy, Matplotlib, Cobaya, GetDist, emcee e Astropy. JupyterLab e ipykernel completam a instalação para executar os notebooks localmente.

## Instalação em Linux, macOS ou WSL

Com Python 3.10 ou superior, execute a partir da raiz do repositório:

```bash
bash notebooks/aulas_praticas/instalar.sh
source notebooks/aulas_praticas/.venv/bin/activate
python -m jupyterlab
```

O script cria `.venv` dentro desta pasta, instala os pacotes de `requirements.txt`, confere as dependências e executa `verificar_ambiente.py`. Pode ser executado novamente para conferir o mesmo ambiente. A localização do script é resolvida automaticamente, permitindo chamá-lo de outra pasta. A ativação deve ser feita no terminal após o término do script.

Para escolher baixar as dependências em um ambiente que você já está acostumado a utilizar, basta fazer, com o ambiente já ativo:

```bash
python -m pip install -r notebooks/aulas_praticas/requirements.txt
python notebooks/aulas_praticas/verificar_ambiente.py
```

Se a criação do ambiente indicar ausência de `venv`, instale o suporte a ambientes virtuais da sua distribuição. Em Debian/Ubuntu, o pacote correspondente costuma ser `python3-venv`.

## Verificação no notebook e no VSCode

Abra `00_verificar_ambiente.ipynb`, escolha o interpretador `notebooks/aulas_praticas/.venv/bin/python` e execute todas as células. Confira o caminho do interpretador e as versões exibidas. O verificador pelo terminal usa os mesmos imports:

```bash
notebooks/aulas_praticas/.venv/bin/python notebooks/aulas_praticas/verificar_ambiente.py
```

Esse teste verifica o carregamento dos módulos e dos símbolos pedidos. As execuções de MCMC e o uso de dados cosmológicos serão conferidos nos próprios notebooks das aulas.