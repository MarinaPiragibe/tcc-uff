
import logging
from time import time

from utils.metricas import Metricas
from utils.enums.tipos_transformacao_wisard import TiposDeTransformacao

def executar_wisard(modelo, train_loader, test_loader, termometro, tamanho_tupla, tipo_transformacao, stride_hd):
	
	inicio_treino = time()

	for k, (dados_do_lote, classes_do_lote) in enumerate(train_loader):
		print(f"\r{k+1}/{len(train_loader)}", end="", flush=True)
		dados_do_lote_bin = transformar_dados_em_binarios(
			dados_do_lote=dados_do_lote,
			tipo_transformacao=tipo_transformacao,
			termometro=termometro,
			stride_hd=stride_hd
		)

		classes_do_lote = classes_do_lote.numpy().astype(str)

		modelo.train(dados_do_lote_bin, classes_do_lote)

	final_treino = time()

	tempo_total_treino = final_treino - inicio_treino

	logging.info(f"[TUPLA {tamanho_tupla}] Fim do treinamento do WisardPKG. Tempo total do treino: {tempo_total_treino}")

	classes_preditas = []
	classes_reais = []

	logging.info(f"[TUPLA {tamanho_tupla}] Iniciando teste do WisardPKG ")

	inicio_teste = time()

	for k, (dados_do_lote, classes_do_lote) in enumerate(test_loader):
		print(f"\r{k+1}/{len(test_loader)}", end="", flush=True)
		dados_do_lote_bin = transformar_dados_em_binarios(
			dados_do_lote=dados_do_lote,
			tipo_transformacao=tipo_transformacao,
			termometro=termometro,
			stride_hd=stride_hd
		)

		classes_do_lote = classes_do_lote.numpy().astype(str)
		preds = modelo.classify(dados_do_lote_bin)
		classes_preditas.extend(preds)
		classes_reais.extend(classes_do_lote)

	final_teste = time()
	tempo_total_teste = final_teste - inicio_teste

	logging.info(f"[TUPLA {tamanho_tupla}] Fim do teste do WisardPKG . Tempo total de inferência: {tempo_total_teste}")


	logging.info(f"[TUPLA {tamanho_tupla}] Execução do modeloo  concluída em {tempo_total_treino + tempo_total_teste}")
	logging.info(f"[TUPLA {tamanho_tupla}] Tempo de execução do treino: {tempo_total_treino}")
	logging.info(f"[TUPLA {tamanho_tupla}] Tempo de execução do teste: {tempo_total_teste}")

	metricas = Metricas(classes_reais=classes_reais, classes_preditas=classes_preditas)
	
	logging.info(f"Calculando métricas de desempenho")
	metricas.calcular_e_imprimir_metricas()

def transformar_dados_em_binarios(lote_dado, tipo_transformacao, termometro, stride_hd):
	if tipo_transformacao == TiposDeTransformacao.STRIDE_HD:
		pooled_windows = stride_hd.extract_and_pool(lote_dado)
		B, N, C, H_pool, W_pool = pooled_windows.shape
		dados_para_wisard = pooled_windows.contiguous().view(B, -1)

		dados_do_lote_bin = termometro.binarize(dados_para_wisard).numpy()
		dados_do_lote_bin = dados_do_lote_bin.reshape(B, -1).astype(int).tolist()

		return dados_do_lote_bin
	
	return termometro.binarize(lote_dado).flatten(start_dim=1).numpy()