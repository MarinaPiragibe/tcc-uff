import torch
import torch.nn.functional as F
import numpy as np
import logging
from utils.arquivo_utils import ArquivoUtils

class StrideHD:
	def __init__(self, stride=2, pool_size=(2,2)):
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
		logging.info(f"Executando StrideHD")
		X_tr, y_tr = self._encode_loader(train_loader)
		X_te, y_te = self._encode_loader(test_loader)

		logging.info("Salvando features no formato padrão")
		ArquivoUtils.salvar_features_imagem(
			nome_tecnica_ext=nome_tecnica_ext,
			nome_dataset=dataset_enum.value,
			dados_treino=X_tr,
			classes_treino=y_tr,
			dados_teste=X_te,
			classes_teste=y_te
		)
		logging.info("Pipeline StrideHD concluído e salvo com sucesso.")

