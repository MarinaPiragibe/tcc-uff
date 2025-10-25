from enum import Enum
import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import ORB, fisher_vector, learn_gmm
from torchvision.datasets import CIFAR10
from torchvision import transforms

from utils.arquivo_utils import ArquivoUtils

# -------------------------------------------------------------
# 1. Carregar CIFAR-10
# -------------------------------------------------------------

class TipoDescritor(Enum):
    SIFT = "sift"
    ORB = 'orb'


class FisherVectorORB:
    def __init__(self, train_loader, test_loader, dataset):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.dataset = dataset

    def converter_dados_para_np(self, loader):
        dados_convertidos = []
        classes_convertidas = []
        for k, (dado, classe) in enumerate(loader):
            dado = dado.permute(0, 2, 3, 1).numpy()  # [B,C,H,W] → [B,H,W,C]
            dados_convertidos.append(dado)
            classes_convertidas.append(classe.numpy())

        dados_convertidos = np.concatenate(dados_convertidos)
        classes_convertidas = np.concatenate(classes_convertidas)

        return dados_convertidos, classes_convertidas

    @staticmethod
    def converter_escala_de_cinza_e_redimensionar(dados):
        dados_escala_cinza = np.array([rgb2gray(dado) for dado in dados])
        dados_redimensionados = np.array([resize(dado, (80, 80)) for dado in dados_escala_cinza])
        return dados_redimensionados

    @staticmethod
    def extrair_descritores_orb(dados):
        descritores = []
        for dado in dados:
            orb = ORB(n_keypoints=5, harris_k=0.01)
            orb.detect_and_extract(dado)
            if orb.descriptors is not None:
                descritores.append(orb.descriptors.astype("float32"))
            else:
                # se não encontrar pontos, adiciona ruído pequeno
                descritores.append(np.zeros((5, 256), dtype="float32"))
        return descritores

    @staticmethod
    def extrair_descritores_sift(dados, nfeatures=20):
        sift = cv2.SIFT_create(nfeatures=nfeatures)
        descritores = []
        for dado in dados:
            # converter para uint8, necessário para o OpenCV
            dado_convertido = (dado * 255).astype(np.uint8)
            dado_escala_cinza = cv2.cvtColor(dado_convertido, cv2.COLOR_RGB2GRAY)
            kp, desc = sift.detectAndCompute(dado_escala_cinza, None)
            if desc is None:
                desc = np.zeros((5, 128), dtype=np.float32)
            descritores.append(desc.astype("float32"))
        return descritores

    @staticmethod
    def extrair_descritores_sift_colorido(dados, nfeatures=20):
        sift = cv2.SIFT_create(nfeatures=nfeatures)
        descritores = []

        for dado in dados:
            descs = []
            for c in range(3):  # R, G, B
                canal = (dado[..., c] * 255).astype(np.uint8)
                kp, desc = sift.detectAndCompute(canal, None)
                if desc is not None:
                    descs.append(desc)
            if descs:
                descritores.append(np.vstack(descs).astype("float32"))
            else:
                descritores.append(np.zeros((5, 128), dtype="float32"))
        return descritores

    @staticmethod
    def treinar_gmm(descritores, n_modes=16):
        print("Treinando GMM...")
        # Junta todos os descritores (de todas as imagens) num único array
        todos_descritores = np.vstack(descritores)
        gmm = learn_gmm(todos_descritores, n_modes=n_modes)
        return gmm
    
    @staticmethod
    def calcular_fisher_vectors(descritores, gmm):
        return np.array([fisher_vector(d, gmm) for d in descritores])

    def executar_fisher_vector(self, tipo_descritor: TipoDescritor):
        dados_treino, classes_treino = self.converter_dados_para_np(self.train_loader)
        dados_teste, classes_teste = self.converter_dados_para_np(self.test_loader)

        descritores_treino = []
        descritores_teste = []

        if tipo_descritor == TipoDescritor.SIFT:
            descritores_treino = self.extrair_descritores_sift(dados_treino)
            descritores_teste = self.extrair_descritores_sift(dados_teste)
            
        if tipo_descritor == TipoDescritor.ORB:
            dados_treino = self.converter_escala_de_cinza_e_redimensionar(dados_treino)
            dados_teste = self.converter_escala_de_cinza_e_redimensionar(dados_teste)

            descritores_treino = self.extrair_descritores_orb(dados_treino)
            descritores_teste = self.extrair_descritores_orb(dados_teste)

        gmm = self.treinar_gmm(descritores_treino)
        fisher_vector_treino = self.calcular_fisher_vectors(descritores_treino, gmm)
        fisher_vector_teste = self.calcular_fisher_vectors(descritores_teste, gmm)


        ArquivoUtils.salvar_features_imagem(
            nome_tecnica_ext=tipo_descritor.value,
            nome_dataset=self.dataset,
            dados_treino=fisher_vector_treino,
            classes_treino=classes_treino,
            dados_teste=fisher_vector_teste,
            classes_teste=classes_teste
        )

