import logging
import numpy as np
import torch
import torch.nn.functional as F

from utils.arquivo_utils import ArquivoUtils

class StrideHD:
	def __init__(self, window_size=(4,4), stride=2, pool_size=(2,2), dimensao_hv=10000, num_hv_distribuido=100):
		self.window_size = window_size
		self.stride = stride
		self.pool_size = pool_size
		self.dimensao_hv = dimensao_hv
		self.num_hv_distribuido = num_hv_distribuido

	def extract_and_pool(self, dado: torch.Tensor):
		B, C, H, W = dado.shape

		# 1) Extrai as janelas diretamente usando unfold
		janelas = dado.unfold(2, self.window_size[0], self.stride).unfold(3, self.window_size[1], self.stride)
		# windows: [B, C, num_h, num_w, win_H, win_W]

		# 2) Combina dimensões num_h e num_w para formar num_windows
		B, C, num_janelas_altura, num_janelas_largura, altura_janela, largura_janela = janelas.shape
		janelas = janelas.permute(0, 2, 3, 1, 4, 5)  # [B, num_h, num_w, C, win_H, win_W]
		janelas = janelas.contiguous().view(B, num_janelas_altura*num_janelas_largura, C, altura_janela, largura_janela)  # [B, num_windows, C, win_H, win_W]

		# 3) Aplicar max pooling de forma eficiente
		# Transformamos [B*num_windows, C, win_H, win_W] para aplicar F.max_pool2d
		pooled = F.max_pool2d(janelas.view(-1, C, altura_janela, largura_janela), self.pool_size)
		pooled_altura, pooled_largura = pooled.shape[2], pooled.shape[3]
		pooled = pooled.view(B, num_janelas_altura*num_janelas_largura, C, pooled_altura, pooled_largura)

		return pooled

	@torch.inference_mode()
	def _encode_loader(self, loader, progress: bool = True):
		"""
		Gera X e y prontos para salvar, mantendo a lógica existente:
		usa extract_and_pool e apenas 'flatten' para serializar (nenhuma agregação nova).
		"""
		feats, labels = [], []
		for k, (imgs, cls) in enumerate(loader):
			# garante CPU para conversão em numpy
			if imgs.device.type != "cpu":
				imgs = imgs.to("cpu")

			pooled = self.extract_and_pool(imgs)      # [B, Wn, C, ph, pw]
			B = pooled.shape[0]
			feat = pooled.reshape(B, -1)              # apenas achata para 2D (não muda a lógica)
			feats.append(feat.numpy().astype(np.float32))
			labels.append(cls.numpy())

			print(f"\r{k+1}/{len(loader)}", end="", flush=True)

		X = np.concatenate(feats, axis=0)
		y = np.concatenate(labels, axis=0)
		logging.info(f"StrideHD -> amostras: {X.shape[0]}, dimensão: {X.shape[1]}")
		return X, y

	def executar_e_salvar(self, train_loader, test_loader, dataset_enum):
		"""
		Salva nos mesmos moldes do primeiro código, sem alterar a lógica:
		extract_and_pool -> flatten para serializar -> ArquivoUtils.salvar_features_imagem
		"""
		logging.info("Executando StrideHD (device=cpu)...")
		X_tr, y_tr = self._encode_loader(train_loader)
		X_te, y_te = self._encode_loader(test_loader)

		logging.info("Salvando features no formato padrão (compatível com seu pipeline)...")
		ArquivoUtils.salvar_features_imagem(
			nome_tecnica_ext=f"features/stride_hd_{self.window_size[0]}",
			nome_dataset=dataset_enum.value,
			dados_treino=X_tr,
			classes_treino=y_tr,
			dados_teste=X_te,
			classes_teste=y_te
		)
		logging.info("Pipeline StrideHD concluído e salvo com sucesso.")


if __name__ == "__main__":
	x = torch.randn(8, 3, 32, 32)
	stride_hd = StrideHD(window_size=(4,4), stride=2, pool_size=(2,2))
	pooled_windows = stride_hd.extract_and_pool(x)
	
	print("Shape após extração + pooling:", pooled_windows.shape)
