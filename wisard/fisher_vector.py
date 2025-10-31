import cv2
import numpy as np
from sklearn.decomposition import PCA
from skimage.transform import resize
from skimage.feature import fisher_vector, learn_gmm

from utils.arquivo_utils import ArquivoUtils

class FisherVector():
	def __init__(self, train_set, test_set, args, 
				 K=16, max_desc_per_img=200, gmm_pool_size=100_000, 
				 pca_dim=64, resize_to=80, feature_type='sift'):

		self.pca = PCA(n_components=pca_dim, whiten=True, random_state=42)
		self.feature_type = feature_type.lower()
		self.sift = cv2.SIFT_create() if self.feature_type == 'sift' else None
		self.orb = cv2.ORB_create(nfeatures=max_desc_per_img) if self.feature_type == 'orb' else None
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
		img_resized = resize(img_rgb_np, (self.resize_to, self.resize_to), anti_aliasing=True)
		return (img_resized * 255).astype(np.uint8)

	def extract_desc(self, img_u8):
		if self.feature_type == 'sift':
			kpts, desc = self.sift.detectAndCompute(img_u8, None)
			desc_dim = 128
		else:
			kpts, desc = self.orb.detectAndCompute(img_u8, None)
			desc_dim = 32
			if desc is not None:
				desc = desc.astype(np.float32)
		if desc is None or len(desc) == 0:
			return np.zeros((1, desc_dim), dtype=np.float32)
		if len(desc) > self.max_desc_per_img:
			idx = self.rng.choice(len(desc), size=self.max_desc_per_img, replace=False)
			desc = desc[idx]
		return desc.astype(np.float32)

	def pool_descritores(self):
		pool_desc, total = [], 0
		for i, (img_pil, _) in enumerate(self.train_set):
			img_np = np.array(img_pil)
			img_u8 = self.to_rgb_u8(img_np)
			d = self.extract_desc(img_u8)
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
		# Retorna os descritores já transformados por PCA
		return [self.pca.transform(d).astype(np.float32) for d in pool_desc]

	# NOVO: método para obter descritores PCA de qualquer conjunto (treino ou teste)
	def dataset_pca_desc(self, dataset):
		descs_pca, labels = [], []
		for img_pil, y in dataset:
			img_np = np.array(img_pil)
			img_u8 = self.to_rgb_u8(img_np)
			d = self.extract_desc(img_u8)
			d_pca = self.pca.transform(d).astype(np.float32)
			descs_pca.append(d_pca)
			labels.append(y)
		return descs_pca, labels

	# Modificado: recebe descs_pca prontos e não precisa recomputar nada!
	def calcular_fisher_vector(self, descs_pca, labels, gmm):
		fvs = []
		for i, d in enumerate(descs_pca):
			fv = fisher_vector(d, gmm, improved=True).astype(np.float32)
			fvs.append(fv)
			print(f"\r{i+1}/{len(descs_pca)}", end="", flush=True)
		return np.vstack(fvs).astype(np.float32), np.array(labels)

	def executar_e_salvar(self):
		print("Gerando pool de descritores...")
		pool_desc = self.pool_descritores()
		print("\nAplicando PCA ao pool de descritores...")
		pool_desc_pca = self.aplicar_pca(pool_desc)

		print("\nImprimindo GMM...")
		gmm = learn_gmm(pool_desc_pca, n_modes=self.K)

		print("\nProcessando descritores PCA do treino...")
		descs_train_pca, classes_treino = self.dataset_pca_desc(self.train_set)
		print("\nProcessando descritores PCA do teste...")
		descs_test_pca, classes_teste = self.dataset_pca_desc(self.test_set)

		print("\nCalculando Fisher Vectors do treino...")
		fvs_treino, classes_treino = self.calcular_fisher_vector(descs_train_pca, classes_treino, gmm)
		print("\nCalculando Fisher Vectors do teste...")
		fvs_teste, classes_teste = self.calcular_fisher_vector(descs_test_pca, classes_teste, gmm)

		ArquivoUtils.salvar_features_imagem(
			nome_tecnica_ext=f"sift_fv_k{self.K}_pca{self.pca_dim}",
			nome_dataset=self.args['dataset'],
			dados_treino=fvs_treino,
			classes_treino=classes_treino,
			dados_teste=fvs_teste,
			classes_teste=classes_teste
		)
