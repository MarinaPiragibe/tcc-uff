from time import time
import numpy as np
from datasets.iris import Iris
from utils.logger import Logger
from utils.metricas import Metricas
from wisard.termometro import Termometro
from wisard.wisard_model_impl_propria import Wisard
import logging
from sklearn.model_selection import train_test_split

Logger.configurar_logger(nome_arquivo="wisard_application.log")

logger = logging.getLogger(__name__)

iris = Iris()

n_bits=10
tamanho_tupla = 5
termometro = Termometro(n_bits=n_bits)

colunas_binarizadas = []

for col in iris.atributos_numericos:
    logging.info(f"Binarizando coluna '{col}' com {n_bits} bits (Termômetro).")
    colunas_binarizadas.append(termometro.codificador_termometro(iris.atributos[col].astype(float)))
    logging.info(f"Coluna '{col}' binarizada\n")

entradas_binarizadas = np.hstack(colunas_binarizadas)
tamanho_entrada = entradas_binarizadas.shape[1]

logging.info(f'Dimensão de cada entrada após a binarização: {tamanho_entrada} bits')
wisard = Wisard(tamanho_entrada, tamanho_tupla)

classes = iris.classes['class'].values 

entradas_treino, entradas_teste, classes_treino, classes_test = train_test_split(
    entradas_binarizadas, classes, test_size=0.3, random_state=42, stratify=iris.classes
)
logging.info(f'Amostras separadas em {classes_treino.size} para treino e {classes_test.size} para teste')
logging.info("Treinamento da wisard iniciado")
inicio = time()
for k, (entrada, classe) in enumerate(zip(entradas_treino, classes_treino)):
    wisard.fit(entrada, classe)
    print(f"\rTreinando amostras: {k+1}/{len(entradas_treino)}", end='', flush=True)

print()
fim = time()
logging.info(f"Treinamento concluído em {fim-inicio}")

classes_reais = []
classes_preditas = []

logging.info("Teste da wisard iniciado")
start = time()
for entrada, classe in zip(entradas_teste, classes_test):
    resultado = wisard.predict(entrada, classe)
    classes_reais.append(classe)
    classes_preditas.append(resultado)
end = time()
logging.info(f"Teste concluído em {fim-inicio}")

metricas = Metricas(classes_reais=classes_reais, classes_preditas=classes_preditas)
metricas.calcular_e_imprimir_metricas()
