import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class VLADTransform:
    def __init__(self, num_centros=16, n_components_pca=24):
        self.num_centros = num_centros
        self.n_components_pca = n_components_pca
        self.kmeans = None
        self.pca = None

    def fit(self, dataloader, num_amostras=500):
        """
        Ajusta PCA e KMeans com amostras do dataloader.
        """
        descritores = []

        # coleta amostras
        count = 0
        for imgs, _ in dataloader:
            for img in imgs:
                desc = self.extrair_descritores_locais(img)
                descritores.append(desc)
                count += 1
                if count >= num_amostras:
                    break
            if count >= num_amostras:
                break

        todos_descritores = np.concatenate(descritores, axis=0)
        self.pca = PCA(n_components=self.n_components_pca, random_state=42)
        descritores_pca = self.pca.fit_transform(todos_descritores)

        self.kmeans = KMeans(n_clusters=self.num_centros, random_state=42)
        self.kmeans.fit(descritores_pca)

    @staticmethod
    def extrair_descritores_locais(imagem, tamanho_janela=4, passo=4):
        """
        Divide a imagem em patches e retorna vetores achatados.
        """
        C, H, W = imagem.shape
        descritores = []

        for y in range(0, H - tamanho_janela + 1, passo):
            for x in range(0, W - tamanho_janela + 1, passo):
                patch = imagem[:, y:y+tamanho_janela, x:x+tamanho_janela]
                descritores.append(patch.flatten().numpy())

        return np.stack(descritores)

    def calcular_vlad(self, descritores):
        descritores_pca = self.pca.transform(descritores)
        centros = self.kmeans.cluster_centers_
        num_centros, dim = centros.shape

        vlad = np.zeros((num_centros, dim))
        atribuicoes = np.argmin(
            np.linalg.norm(descritores_pca[:, None, :] - centros[None, :, :], axis=2),
            axis=1
        )

        for i in range(num_centros):
            desc_cluster = descritores_pca[atribuicoes == i]
            if len(desc_cluster) > 0:
                vlad[i] = np.sum(desc_cluster - centros[i], axis=0)

        # power normalization + L2
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
        vlad = vlad.flatten()
        vlad = vlad / np.linalg.norm(vlad)

        return vlad

    def transform(self, imgs):
        """
        Recebe batch de imagens, retorna matriz (B, D)
        """
        vlads = []
        for img in imgs:
            desc = self.extrair_descritores_locais(img)
            v = self.calcular_vlad(desc)
            vlads.append(v)
        return np.stack(vlads)  # shape (B, D)
