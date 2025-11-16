from time import time
import torch
import torch.nn.functional as F
import numpy as np
import logging
from utils.arquivo_utils import ArquivoUtils
from utils.enums.tipos_transformacao_wisard import TiposDeTransformacao

class StrideHD:
	def __init__(self, args, stride=2, pool_size=(2,2)):
		self.args = args
		self.stride = stride
		self.pool_size = pool_size

	def extract_and_pool(self, dado: torch.Tensor):
		# Obtemos as dimensões do tensor de entrada
		B, C, H, W = dado.shape  # B: batch size, C: canais, H: altura, W: largura

		# Aplicar o Max Pooling diretamente ao tensor (cada imagem do batch é processada separadamente)
		pooled = F.max_pool2d(dado, kernel_size=self.pool_size, stride=self.stride)

		return pooled

	@torch.inference_mode()
	def _encode_loader(self, loader):
		feats, labels = [], []
		for k, (imgs, cls) in enumerate(loader):
			if imgs.device.type != "cpu":
				imgs = imgs.to("cpu")
			
			# Sempre aplicar max pooling
			pooled = self.extract_and_pool(imgs)
			
			# "Flatten" cada imagem em um vetor
			B, C, H, W = pooled.shape
			pooled_flat = pooled.view(B, -1)  # cada imagem vira um vetor
			
			feats.append(pooled_flat.numpy())   
			labels.append(cls.numpy())
		
			print(f"\r{k+1}/{len(loader)}", end="", flush=True)

		X = np.concatenate(feats, axis=0)
		y = np.concatenate(labels, axis=0)
		logging.info(f"StrideHD -> amostras: {X.shape[0]}, dimensão: {X.shape[1]}")
		return X, y


	def executar_e_salvar(self, train_loader, test_loader, dataset_enum,
						  nome_tecnica_ext: str = "stride_hd",
						  agregacao: str = "max"):
		logging.info(f"[ÍNICIO] Extraindo caracteristicas com stride {self.stride}")
		logging.info(f"Executando StrideHD no conjunto de treino")
		inicio_treino = time()
		dados_treino, classes_treino = self._encode_loader(train_loader)
		fim_treino=time()
		tempo_total_treino = fim_treino - inicio_treino
		logging.info(f"Fim execução do StrideHD no conjunto treino. Tempo total: {tempo_total_treino}")

		logging.info(f"Executando StrideHD no conjunto de teste")
		inicio_teste = time()
		dados_teste, classes_teste = self._encode_loader(test_loader)
		fim_teste=time()
		tempo_total_teste = fim_teste - inicio_teste
		logging.info(f"Fim execução do StrideHD no conjunto teste. Tempo total: {tempo_total_teste}")

		logging.info("Salvando features no formato padrão")
		ArquivoUtils.salvar_features_imagem(
			nome_tecnica_ext=f"{nome_tecnica_ext}_stride{self.stride}",
			nome_dataset=dataset_enum.value,
			dados_treino=dados_treino,
			classes_treino=classes_treino,
			dados_teste=dados_teste,
			classes_teste=classes_teste
		)
		logging.info(f"[FIM] Tempo de execução do stride HD {tempo_total_treino + tempo_total_teste}")

		dados_execucao = {
			"nome_tecnica": f"{TiposDeTransformacao.STRIDE_HD.value}_{self.stride}",
			"tempo_treino": tempo_total_treino,
			"tempo_teste": tempo_total_teste,
			"tempo_total": tempo_total_treino + tempo_total_teste,
			"shape_treino": dados_treino.shape,
			"shape_teste": dados_teste.shape,
		}

		ArquivoUtils.salvar_csv(self.args, dados_execucao, self.args['arq_ext_caract'])

