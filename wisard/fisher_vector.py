import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import torch

class FisherVectorTransform:
    def __init__(self, num_gaussians=16, n_components_pca=64):
        """
        num_gaussians: número de componentes do GMM
        n_components_pca: dimensionalidade final após PCA
        """
        self.num_gaussians = num_gaussians
        self.n_components_pca = n_components_pca
        self.gmm = None
        self.pca = None
        self.means_ = None
        self.covs_ = None
        self.weights_ = None

    @staticmethod
    def extrair_descritores_locais(imagem, tamanho_janela=4, passo=4):
        """
        Divide a imagem em patches e retorna vetores achatados.
        Funciona tanto para PyTorch tensors quanto para numpy arrays.
        """
        # Se for tensor PyTorch, converte para numpy
        if isinstance(imagem, torch.Tensor):
            imagem = imagem.cpu().numpy()
        # Se for PIL Image, converte para numpy
        elif "PIL" in str(type(imagem)):
            imagem = np.array(imagem).transpose(2, 0, 1)  # HWC -> CHW

        C, H, W = imagem.shape
        descritores = []

        for y in range(0, H - tamanho_janela + 1, passo):
            for x in range(0, W - tamanho_janela + 1, passo):
                patch = imagem[:, y:y+tamanho_janela, x:x+tamanho_janela]
                descritores.append(patch.flatten())

        return np.stack(descritores)  

    def fit_gmm(self, dataloader):
        todos_descritores = []
        for imgs, _ in dataloader:
            # imgs pode ser tensor ou PIL
            if isinstance(imgs, torch.Tensor):
                imgs = imgs.numpy()
            for img in imgs:
                d = self.extrair_descritores_locais(img)
                todos_descritores.append(d)
        todos_descritores = np.concatenate(todos_descritores, axis=0)

        # GMM
        self.gmm = GaussianMixture(
            n_components=self.num_gaussians,
            covariance_type='diag',
            random_state=42
        )
        self.gmm.fit(todos_descritores)
        self.means_ = self.gmm.means_
        self.covs_ = self.gmm.covariances_
        self.weights_ = self.gmm.weights_

        # PCA: nunca mais que a dimensão real
        n_features = todos_descritores.shape[1]
        n_components = min(self.n_components_pca, n_features)
        self.pca = PCA(n_components=n_components, random_state=42)
        self.pca.fit(todos_descritores)

    def extract_features(self, batch_imgs):
        """
        Retorna Fisher Vectors já em forma de tensor para o termometro.
        """
        batch_vf = []
        for img in batch_imgs:
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            descritores = self.extrair_descritores_locais(img)
            descritores_pca = self.pca.transform(descritores)

            # Fisher Vector (simplificado)
            fv = np.zeros((self.num_gaussians, descritores_pca.shape[1]))
            prob = self.gmm.predict_proba(descritores_pca)  # (num_patches, num_gaussians)
            for k in range(self.num_gaussians):
                diff = descritores_pca - self.means_[k]
                fv[k] = np.sum(prob[:, k][:, None] * diff, axis=0)

            # Normalização
            fv = np.sign(fv) * np.sqrt(np.abs(fv))
            fv = fv.flatten()
            fv /= np.linalg.norm(fv) + 1e-10  # evitar divisão por zero
            batch_vf.append(fv)

        batch_vf = np.stack(batch_vf)  # (B, dim)
        return torch.tensor(batch_vf, dtype=torch.float32)
