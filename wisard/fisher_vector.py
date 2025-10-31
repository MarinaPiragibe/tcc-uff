from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA 

from torchvision.datasets import CIFAR10
from skimage.transform import resize
from skimage.feature import fisher_vector, learn_gmm
import cv2

from utils.arquivo_utils import ArquivoUtils


class FisherVector():
	def __init__(self, train_set, test_set, args, K=16, max_desc_per_img=200, gmm_pool_size=100_000, pca_dim = 64, resize_to=80):
		self.pca = PCA(n_components=64, whiten=True, random_state=42)
		self.sift = cv2.SIFT_create()
		self.rng = np.random.default_rng(42)
		self.train_set = train_set
		self.test_set = test_set
		self.args = args
		self.resize_to = resize_to
		self.K = K
		self.max_desc_per_img = max_desc_per_img
		self.gmm_pool_size = gmm_pool_size
		self.pca_dim = pca_dim


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
		sift = cv2.SIFT_create()
		pool_desc, total = [], 0
		for i, (img_pil, _) in enumerate(self.train_set):
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
			print(f"\r{i+1}/{len(self.train_set)}", end="", flush=True)

		return pool_desc

	def aplicar_pca(self, pool_desc):
		pool_mat = np.vstack(pool_desc)
		self.pca.fit(pool_mat)
		return [self.pca.transform(d).astype(np.float32) for d in pool_desc]



	def calcular_fisher_vector(self, lote_dados, gmm):
		train_fvs, y_train = [], []
		for i, (img_pil, y) in enumerate(lote_dados):
			img_np = np.array(img_pil)
			img_u8 = self.to_rgb_u8(img_np)
			d = self.sift_desc_cv2(img_u8)
			d = self.pca.transform(d).astype(np.float32)

			fv = fisher_vector(d, gmm, improved=True).astype(np.float32)
			train_fvs.append(fv); y_train.append(y)
			print(f"\r{i+1}/{len(lote_dados)}", end="", flush=True)

		dados_fvs = np.vstack(train_fvs).astype(np.float32)
		classes = np.array(y_train)

		return dados_fvs, classes

	def executar_e_salvar(self):

		pool_desc = self.pool_descritores()
		pool_desc_pca = self.aplicar_pca(pool_desc)

		gmm = learn_gmm(pool_desc_pca, n_modes=self.K)

		fvs_treino, classes_treino = self.calcular_fisher_vector(self.dados_treino, gmm)
		fvs_teste, classes_teste = self.calcular_fisher_vector(self.dados_teste, gmm)

		ArquivoUtils.salvar_features_imagem(
			nome_tecnica_ext=f"sift_fv_k{self.K}",
			nome_dataset=self.args['nome_dataset'],
			dados_treino=fvs_treino,
			classes_treino=classes_treino,
			dados_teste=fvs_teste,
			classes_teste=classes_teste
		)