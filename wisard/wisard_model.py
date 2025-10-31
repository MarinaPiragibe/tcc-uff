
import logging
from time import time

import torch

from utils.metricas import Metricas
from utils.enums.tipos_transformacao_wisard import TiposDeTransformacao

class WisardModel():
	def __init__(self, modelo, tamanho_tupla, dados_treino, classes_treino, dados_teste, classes_teste, termometro,args):
		self.modelo = modelo
		self.tamanho_tupla = tamanho_tupla
		self.dados_treino = dados_treino
		self.classes_treino = classes_treino
		self.dados_teste = dados_teste
		self.classes_teste = classes_teste
		self.termometro = termometro
		self.args = args

	def treinar(self):
		
		logging.info(f"[TUPLA {self.tamanho_tupla}] Iniciando treino do WisardPKG ")

		inicio_treino = time()

		for k, (dado, classe) in enumerate(self.dados_treino):
			print(f"\r{k+1}/{len(self.dados_treino)}", end="", flush=True)
			dado_bin = self.termometro.binarize(dado)
			dado_bin = dado_bin.reshape(dado_bin.size(0), -1).tolist()
			self.modelo.train(dado_bin, classe)

		final_treino = time()

		tempo_total_treino = final_treino - inicio_treino

		logging.info(f"[TUPLA {self.tamanho_tupla}] Fim do treinamento do WisardPKG. Tempo total do treino: {tempo_total_treino}")

		return tempo_total_treino

	def testar(self):
		classes_preditas = []
		classes_reais = []

		logging.info(f"[TUPLA {self.tamanho_tupla}] Iniciando teste do WisardPKG ")

		inicio_teste = time()

		for k, (dado, classe) in enumerate(self.dados_teste):
			print(f"\r{k+1}/{len(self.dados_teste)}", end="", flush=True)
			dado_bin = self.termometro.binarize(dado)
			dado_bin = dado_bin.reshape(dado_bin.size(0), -1).tolist()

			preds = self.modelo.classify(dado_bin)
			classes_preditas.extend(preds)
			classes_reais.extend(classe)

		final_teste = time()
		tempo_total_teste = final_teste - inicio_teste

		logging.info(f"[TUPLA {self.tamanho_tupla}] Fim do teste do WisardPKG . Tempo total de inferência: {tempo_total_teste}")

		return tempo_total_teste, classes_reais, classes_preditas

	def executar_modelo(self):
		tempo_total_treino = self.treinar()

		tempo_total_teste, classes_reais, classes_preditas = self.testar()

		logging.info(f"[TUPLA {self.tamanho_tupla}] Execução do modelo  concluída em {tempo_total_treino + tempo_total_teste}")
		logging.info(f"[TUPLA {self.tamanho_tupla}] Tempo de execução do treino: {tempo_total_treino}")
		logging.info(f"[TUPLA {self.tamanho_tupla}] Tempo de execução do teste: {tempo_total_teste}")

		metricas = Metricas(classes_reais=classes_reais, classes_preditas=classes_preditas)
		
		logging.info(f"Calculando métricas de desempenho")
		metricas.calcular_e_imprimir_metricas()