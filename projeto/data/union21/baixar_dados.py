"""Recupera os bytes oficiais registrados, sem aceitar uma mudança silenciosa."""
from pathlib import Path
from urllib.request import Request, urlopen
import hashlib
import json


def main():
    folder = Path(__file__).resolve().parent  # Mantém os dados junto da sua documentação.
    records = json.loads((folder/'provenance.json').read_text())  # Usa as URLs e os hashes efetivamente verificados na entrega.
    downloads = []  # Confere o conjunto inteiro antes de substituir algum arquivo.
    for record in records:
        request = Request(record['url'], headers={'User-Agent': 'Mozilla/5.0'})  # Solicita o arquivo público na fonte oficial.
        with urlopen(request, timeout=60) as response:
            data = response.read()  # Obtém os bytes originais, sem reformatar números ou linhas.
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise RuntimeError(f"A fonte mudou: {record['file']}. Revise os dados antes de substituir a cópia.")
        downloads.append((folder/record['file'], data))  # Só prepara uma escrita depois de conferir a integridade.
    for path, data in downloads:
        temporary = path.with_name(path.name+'.download')  # Evita deixar um arquivo final parcialmente escrito.
        temporary.write_bytes(data)
        temporary.replace(path)  # Substitui somente pela cópia com o hash esperado.
        print('Recuperado e conferido:', path.name)


if __name__ == '__main__':
    main()
