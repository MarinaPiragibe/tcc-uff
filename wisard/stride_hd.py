import torch
import torch.nn.functional as F

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

	def encode_to_hv(self, binary_pattern: torch.Tensor):
		if not isinstance(binary_pattern, torch.Tensor):
			binary_pattern = torch.tensor(binary_pattern, dtype=torch.float32)

		B = binary_pattern.shape[0]
		hv = torch.zeros((B, self.dimensao_hv), dtype=torch.bool, device=binary_pattern.device)

		for b in range(B):
			flat = binary_pattern[b].flatten()
			indices = torch.nonzero(flat).flatten()
			if len(indices) == 0:
				continue
			sel = indices[torch.randint(0, len(indices), (self.num_hv_distribuido,))]
			hv[b, sel % self.dimensao_hv] = 1
		return hv


if __name__ == "__main__":
	x = torch.randn(8, 3, 32, 32)
	stride_hd = StrideHD(window_size=(4,4), stride=2, pool_size=(2,2))
	pooled_windows = stride_hd.extract_and_pool(x)
	
	print("Shape após extração + pooling:", pooled_windows.shape)
