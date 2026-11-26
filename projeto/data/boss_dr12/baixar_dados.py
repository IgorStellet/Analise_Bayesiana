"""Recupera somente os dois membros oficiais usados no Bloco 6, sem extração geral."""
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import tarfile
import tempfile
from urllib.request import Request, urlopen


def main():
    """Verifica todo o conteúdo antes de substituir os arquivos de dados."""
    folder=Path(__file__).resolve().parent  # Usa o diretório do próprio script, mesmo se chamado de outra pasta.
    provenance=json.loads((folder/'provenance.json').read_text())  # Mantém as identidades registradas na obtenção original.
    request=Request(provenance['download_url'],headers={'User-Agent':'DALI-course-project/1.0'})
    with urlopen(request,timeout=60) as response:
        archive=response.read()  # O lançamento é pequeno; não baixa catálogos de galáxias.
    if sha256(archive).hexdigest()!=provenance['archive_sha256']:
        raise ValueError('O arquivo oficial mudou ou o download está incompleto; nenhum dado foi substituído.')
    verified={}
    with tarfile.open(fileobj=BytesIO(archive),mode='r:gz') as release:
        for item in provenance['files']:
            name=item['file']
            if Path(name).name!=name:
                raise ValueError('O nome de saída deve ser um arquivo simples.')
            member=release.getmember(item['archive_member'])  # Seleciona somente o caminho registrado, sem extractall.
            if not member.isfile():
                raise ValueError(f'O membro esperado não é um arquivo: {name}')
            content=release.extractfile(member).read()
            if len(content)!=item['bytes'] or sha256(content).hexdigest()!=item['sha256']:
                raise ValueError(f'Conteúdo inesperado em {name}; nenhum dado foi substituído.')
            verified[name]=content
    staged=[]
    try:
        for name,content in verified.items():
            with tempfile.NamedTemporaryFile(dir=folder,prefix='.bao-',delete=False) as handle:
                handle.write(content)  # Prepara a cópia no mesmo sistema de arquivos.
                temporary=Path(handle.name)
            staged.append((temporary,folder/name))
        for temporary,destination in staged:
            os.replace(temporary,destination)  # Substitui cada arquivo somente depois de validar todos os membros.
    finally:
        for temporary,_ in staged:
            temporary.unlink(missing_ok=True)  # Remove temporários se uma escrita local falhar.
    print('Dois arquivos oficiais recuperados e verificados. Procedência preservada.')


if __name__=='__main__':
    main()
