from pathlib import Path

pasta = Path("features")

for arquivo in pasta.iterdir():
    if arquivo.is_file():
        print(arquivo.name)