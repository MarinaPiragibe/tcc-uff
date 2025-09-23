import numpy as np
from datasets.iris import Iris
from wisard.termometro import Termometro
from wisard.wisard_model import WisardModel
import logging
from sklearn.model_selection import train_test_split


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
iris = Iris()

n_bits=5
termometro = Termometro(n_bits=n_bits)

colunas_binarizadas = []

for col in iris.atributos_numericos:
    logging.info(f"Binarizando coluna '{col}' com {n_bits} bits (Termômetro).\n")
    colunas_binarizadas.append(termometro.codificador_termometro(iris.atributos[col].astype(float)))
    logging.info(f"\nColuna '{col}' binarizada com {n_bits} bits (Termômetro).\n")

entradas_binarizadas = np.hstack(colunas_binarizadas)
tamanho_entrada = entradas_binarizadas.shape[1]

logging.info(f'Dimensão de cada entrada após a binarização: {tamanho_entrada} bits')
wisard = WisardModel(tamanho_entrada, 5)

entradas_treino, entradas_teste, classes_treino, classes_test = train_test_split(
    entradas_binarizadas, iris.classes, test_size=0.3, random_state=42, stratify=iris.classes
)

    # Treinamento
for entrada, classe in zip(entradas_treino, classes_treino):
    wisard.fit(entrada, classe)

logging.info("Treinamento concluído!")

# Exemplo de previsão
pred, scores = wisard.predict(entradas_teste[0])
logging.info(f"Predição da primeira amostra de teste: {pred}, pontuações: {scores}")