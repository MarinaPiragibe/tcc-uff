
import logging
from time import time

import torch

from utils.metricas import Metricas
from utils.enums.tipos_transformacao_wisard import TiposDeTransformacao
from wisard.fisher_vector import FisherVectorTransform

class WisardModel():
	def __init__(self, modelo, tamanho_tupla, train_loader, termometro, test_loader, stride_hd, fisher_transform, vlad_transform, args):
		self.modelo = modelo
		self.tamanho_tupla = tamanho_tupla
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.termometro = termometro
		self.stride_hd = stride_hd
		self.args = args
		self.fisher_transform = fisher_transform
		self.vlad_transform = vlad_transform

	def transformar_dados_em_binarios(self, dados_do_lote):
		match(self.args['tipo_transformacao']):
			case TiposDeTransformacao.STRIDE_HD:
				pooled_windows = self.stride_hd.extract_and_pool(dados_do_lote)
				B, N, C, H_pool, W_pool = pooled_windows.shape
				dados_para_wisard = pooled_windows.contiguous().view(B, -1)
				dados_do_lote_bin = self.termometro.binarize(dados_para_wisard).numpy()
				return dados_do_lote_bin.reshape(B, -1).astype(int).tolist()

			case TiposDeTransformacao.FISHER_VECTOR:
				fv_features = dados_do_lote
				if fv_features.dim() == 3 and fv_features.shape[1] == 1:
					fv_features = fv_features.squeeze(1)
				bin_data = self.termometro.binarize(fv_features)
				if bin_data.dim() == 3:
					bin_data = bin_data.reshape(bin_data.size(0), -1)
				return bin_data.numpy().astype(int).tolist()

			case TiposDeTransformacao.VLAD:
				# Usa o VLAD + PCA para gerar vetores contínuos
				vlads = self.vlad_transform.transform(dados_do_lote)
				# termometro binariza os vetores
				bin_data = self.termometro.binarize(vlads)
				if isinstance(bin_data, torch.Tensor):
					bin_data = bin_data.cpu().int()
				dados_do_lote_bin = bin_data.reshape(bin_data.size(0), -1).tolist()
				return dados_do_lote_bin

			case _:
				return self.termometro.binarize(dados_do_lote).flatten(start_dim=1).numpy()

	def treinar(self):
		
		inicio_treino = time()

		for k, (dados_do_lote, classes_do_lote) in enumerate(self.train_loader):
			print(f"\r{k+1}/{len(self.train_loader)}", end="", flush=True)
			dados_do_lote_bin = self.transformar_dados_em_binarios(
				dados_do_lote=dados_do_lote,
			)

			classes_do_lote = classes_do_lote.numpy().astype(str)

			self.modelo.train(dados_do_lote_bin, classes_do_lote)

		final_treino = time()

		tempo_total_treino = final_treino - inicio_treino

		logging.info(f"[TUPLA {self.tamanho_tupla}] Fim do treinamento do WisardPKG. Tempo total do treino: {tempo_total_treino}")

		return tempo_total_treino

	def testar(self):
		classes_preditas = []
		classes_reais = []

		logging.info(f"[TUPLA {self.tamanho_tupla}] Iniciando teste do WisardPKG ")

		inicio_teste = time()

		for k, (dados_do_lote, classes_do_lote) in enumerate(self.test_loader):
			print(f"\r{k+1}/{len(self.test_loader)}", end="", flush=True)
			dados_do_lote_bin = self.transformar_dados_em_binarios(
				dados_do_lote=dados_do_lote,
			)

			classes_do_lote = classes_do_lote.numpy().astype(str)
			preds = self.modelo.classify(dados_do_lote_bin)
			classes_preditas.extend(preds)
			classes_reais.extend(classes_do_lote)

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