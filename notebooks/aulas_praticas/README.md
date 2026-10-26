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

Para escolher outro Python ou outra pasta de ambiente:

```bash
PYTHON_BIN=python3.12 VENV_DIR=/caminho/para/ambiente bash notebooks/aulas_praticas/instalar.sh
```

Se a criação do ambiente indicar ausência de `venv`, instale o suporte a ambientes virtuais da sua distribuição. Em Debian/Ubuntu, o pacote correspondente costuma ser `python3-venv`.

## Verificação no notebook e no VSCode

Abra `00_verificar_ambiente.ipynb`, escolha o interpretador `notebooks/aulas_praticas/.venv/bin/python` e execute todas as células. Confira o caminho do interpretador e as versões exibidas. O verificador pelo terminal usa os mesmos imports:

```bash
notebooks/aulas_praticas/.venv/bin/python notebooks/aulas_praticas/verificar_ambiente.py
```

Esse teste verifica o carregamento dos módulos e dos símbolos pedidos. As execuções de MCMC e o uso de dados cosmológicos serão conferidos nos próprios notebooks das aulas.

## Windows com PowerShell

Para usar Python nativo no Windows, os mesmos arquivos podem ser utilizados assim, a partir da raiz do repositório:

```powershell
py -3 -m venv notebooks/aulas_praticas/.venv
notebooks/aulas_praticas/.venv/Scripts/python.exe -m pip install -r notebooks/aulas_praticas/requirements.txt
notebooks/aulas_praticas/.venv/Scripts/python.exe -m pip check
notebooks/aulas_praticas/.venv/Scripts/python.exe notebooks/aulas_praticas/verificar_ambiente.py
notebooks/aulas_praticas/.venv/Scripts/python.exe -m jupyterlab
```

No VSCode, selecione `.venv/Scripts/python.exe` desta pasta como interpretador do notebook.

## Colab

Envie o notebook de verificação ao Colab e instale os pacotes no kernel da sessão:

```python
%pip install numpy matplotlib cobaya getdist emcee astropy
```

Execute então as células de imports e versões. Se a interface pedir reinicialização após instalar, reinicie o ambiente de execução e execute as células novamente.

## Escopo e referências

Este ambiente acompanha as aulas práticas. A consolidação das dependências de toda a disciplina fica para o encerramento do curso. `requirements.txt` deixa o pip resolver versões compatíveis com o Python escolhido; o verificador exibe as versões efetivamente instaladas.

Verificação em 26/10/2026, Linux, Python 3.12.14: NumPy 2.5.4, Matplotlib 3.11.2, Cobaya 3.6.2, GetDist 1.7.7, emcee 3.1.6 e Astropy 8.0.2. O instalador, a conferência de dependências e todos os imports foram executados nesse ambiente.

- [Instalação do Cobaya](https://cobaya.readthedocs.io/en/latest/installation.html).
- [Interface do GetDist com Cobaya](https://getdist.readthedocs.io/en/stable/_modules/getdist/mcsamples.html).
