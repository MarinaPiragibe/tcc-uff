import cv2
import numpy as np
from skimage.transform import resize
from sklearn.decomposition import PCA


class Sift():
	def __init__(self, trainset, gmm_pool_size=100_000, max_desc_per_img=200, resize_to=80):
		self.trainset = trainset
		self.sift = cv2.SIFT_create()
		self.pca = PCA(n_components=64, whiten=True, random_state=42)
		self.resize_to = resize_to
		self.max_desc_per_img = max_desc_per_img
		self.gmm_pool_size = gmm_pool_size
		self.rng = np.random.default_rng(42)

	def to_rgb_u8(self, img_rgb_np):
		# resize em RGB e converte para uint8; SIFT do OpenCV aceita 3 canais e converte internamente
		img_resized = resize(img_rgb_np, (self.resize_to, self.resize_to), anti_aliasing=True)  # float [0,1]
		return (img_resized * 255).astype(np.uint8)


	def sift_desc_cv2(self, img_u8):
		kpts, desc = self.sift.detectAndCompute(img_u8, None)
		if desc is None or len(desc) == 0:
			return np.zeros((1, 128), dtype=np.float32)
		if len(desc) > self.max_desc_per_img:
			idx = self.rng.choice(len(desc), size=self.max_desc_per_img, replace=False)
			desc = desc[idx]
		return desc.astype(np.float32)

	def pool_descritores(self):
		pool_desc, total = [], 0
		for i, (img_pil, _) in enumerate(self.trainset):
			img_np = np.array(img_pil)         # RGB uint8 (32x32x3)
			img_u8 = self.to_rgb_u8(img_np)         # RGB uint8 (80x80x3)
			d = self.sift_desc_cv2(img_u8)
			falta = self.gmm_pool_size - total
			if falta <= 0:
				break
			if len(d) > falta:
				d = d[:falta]
			pool_desc.append(d)
			total += len(d)
			print(f"\r{i+1}/{len(self.trainset)}", end="", flush=True)

		return pool_desc
	
	def extrair_descritores_dataset(self, dataset, max_imgs=None, desc_target=None):
		"""
		Extrai descritores SIFT de cada imagem do dataset.
		Permite limitar o número de imagens e/ou de descritores totais
		para acelerar o ajuste do PCA e VLAD.

		Args:
			dataset: conjunto de imagens (ex: CIFAR10).
			max_imgs: número máximo de imagens a processar.
			desc_target: número máximo total de descritores a coletar.

		Returns:
			lista de arrays de descritores [ (Ni,128), (Nj,128), ... ]
		"""
		descs = []
		total_desc = 0

		# Gera uma amostragem aleatória de índices, se max_imgs for especificado
		idxs = np.arange(len(dataset))
		if max_imgs is not None and max_imgs < len(idxs):
			self.rng = getattr(self, "rng", np.random.default_rng(42))
			idxs = self.rng.choice(idxs, size=max_imgs, replace=False)

		for i, idx in enumerate(idxs):
			img_pil, _ = dataset[idx]
			img_np = np.array(img_pil)
			img_u8 = self.to_rgb_u8(img_np)
			d = self.sift_desc_cv2(img_u8)
			descs.append(d)
			total_desc += len(d)

			# Mostra progresso
			print(f"\r{i+1}/{len(idxs)} imagens processadas ({total_desc} descritores)", end="", flush=True)

			# Interrompe se já atingiu o limite total de descritores
			if desc_target is not None and total_desc >= desc_target:
				break

		print(f"\nTotal coletado: {total_desc} descritores de {len(descs)} imagens.")
		return descs

	def transformar_pca(self, lista_desc):
		# usa PCA JÁ AJUSTADO no treino
		return [self.pca.transform(d).astype(np.float32) for d in lista_desc]

	def aplicar_pca(self, pool_desc):
		pool_mat = np.vstack(pool_desc)
		self.pca.fit(pool_mat)
		return [self.pca.transform(d).astype(np.float32) for d in pool_desc]
