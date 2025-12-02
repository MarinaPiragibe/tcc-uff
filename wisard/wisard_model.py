import logging
from time import time
import torch
import gc
import sys # Importante para flush de stdout

from utils.arquivo_utils import ArquivoUtils
from utils.metricas import Metricas

class WisardModel():
	def __init__(self, modelo, tamanho_tupla, dados_treino, classes_treino, dados_teste, classes_teste, termometro, args):
		self.modelo = modelo
		self.tamanho_tupla = tamanho_tupla
		self.dados_treino = dados_treino
		self.classes_treino = classes_treino
		self.dados_teste = dados_teste
		self.classes_teste = classes_teste
		self.termometro = termometro
		self.args = args

	# wisard_model.py

	def _processar_lote(self, dado, treinar=True, classe=None):
		with torch.no_grad():
			if not torch.is_tensor(dado):
				# Converte para tensor e binariza
				dado = torch.from_numpy(dado)
			
			# Binarização (Termômetro)
			# dado_bin agora é um tensor de 0s e 1s
			dado_bin = self.termometro.binarize(dado)
			
			# Otimização Crítica: Mover para CPU e NumPy (memória mais densa)
			# .cpu().numpy() é mais rápido e melhor para GC do que ir direto para .tolist()
			dado_numpy = dado_bin.cpu().numpy().astype(bool)
			
			# Liberar tensores imediatamente
			del dado_bin
			del dado

			# Conversão para lista (aqui está a maior alocação temporária, mas é necessária para a API do C++).
			# O reshape garante que seja [batch, features]
			# dado_lista = dado_bin.reshape(dado_bin.size(0), -1).tolist() # LINHA ANTIGA
			dado_lista = dado_numpy.reshape(dado_numpy.shape[0], -1).tolist() # LINHA NOVA
			
			del dado_numpy # Limpa NumPy array

			if treinar:
				classes_lista = [str(c) for c in classe] 
				self.modelo.train(dado_lista, classes_lista)
				del dado_lista
				return None
			else:
				preds = self.modelo.classify(dado_lista)
				del dado_lista
				return preds

	def treinar(self, num_execucao):
		logging.info(f"[EXECUCAO {num_execucao}] ... Iniciando treino ...")
		inicio_treino = time()

		total_batches = len(self.dados_treino)
		
		for k, (dado, classe) in enumerate(self.dados_treino):
			try:
				# Log de progresso mais limpo
				if k % 10 == 0: 
					sys.stdout.write(f"\r[Treino] Batch {k+1}/{total_batches}")
					sys.stdout.flush()

				self._processar_lote(dado, treinar=True, classe=classe)

				# Coleta de lixo estratégica
				if k % 25 == 0: 
					gc.collect()
			
			except Exception as e:
				logging.error(f"Erro no batch {k} do treino: {e}")
				continue # Pula o batch defeituoso mas continua o treino

		print("") # Quebra de linha após o loop
		final_treino = time()
		tempo_total_treino = final_treino - inicio_treino
		logging.info(f"[EXECUCAO {num_execucao}] [TUPLA {self.tamanho_tupla}] Fim do treino. Tempo: {tempo_total_treino:.2f}s")
		
		# Limpeza final pós-treino
		gc.collect()
		return tempo_total_treino

	def testar(self, num_execucao):
		classes_preditas = []
		classes_reais = []
		logging.info(f"[EXECUCAO {num_execucao}] [TUPLA {self.tamanho_tupla}] Iniciando teste...")
		inicio_teste = time()

		total_batches = len(self.dados_teste)

		for k, (dado, classe) in enumerate(self.dados_teste):
			try:
				if k % 10 == 0:
					sys.stdout.write(f"\r[Teste] Batch {k+1}/{total_batches}")
					sys.stdout.flush()
				
				preds = self._processar_lote(dado, treinar=False)
				
				if preds:
					classes_preditas.extend(preds)
					classes_reais.extend([str(c) for c in classe])
					
				if k % 25 == 0:
					gc.collect()

			except Exception as e:
				logging.error(f"Erro no batch {k} do teste: {e}")
				continue

		print("") 
		final_teste = time()
		tempo_total_teste = final_teste - inicio_teste

		logging.info(f"[EXECUCAO {num_execucao}] [TUPLA {self.tamanho_tupla}] Fim do teste. Tempo: {tempo_total_teste:.2f}s")
		return tempo_total_teste, classes_reais, classes_preditas

	def executar_modelo(self, num_execucao, tecnica_ext_feat):
		try:
			tempo_total_treino = self.treinar(num_execucao)
			tempo_total_teste, classes_reais, classes_preditas = self.testar(num_execucao)

			logging.info(f"Calculando métricas...")
			metricas = Metricas(classes_reais=classes_reais, classes_preditas=classes_preditas)
			metricas.calcular_e_imprimir_metricas()

			dados_execucao = {
					"execucao": num_execucao,
					"modelo_base": tecnica_ext_feat,
					"tupla": self.tamanho_tupla,
					"acuracia": metricas.acc,
					"precisao": metricas.precisao,
					"recall": metricas.recall,
					"f1": metricas.f1,
					"tempo_treino": tempo_total_treino,
					"tempo_teste": tempo_total_teste,
					"tempo_total": tempo_total_treino + tempo_total_teste,
					# Tratamento de erro caso o modelo não suporte getsizeof
					"tamanho": getattr(self.modelo, 'getsizeof', lambda: 0)() 
				}

			ArquivoUtils.salvar_csv(self.args, dados_execucao)
		
		except Exception as e:
			logging.critical(f"Falha fatal na execução do modelo: {e}")
			raise e