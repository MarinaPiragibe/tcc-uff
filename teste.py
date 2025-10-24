import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import wisardpkg as wp

from utils.metricas import Metricas

# ============================================
# 1️⃣ Carregar dataset CIFAR-10
# ============================================
transformacao = transforms.Compose([
    transforms.ToTensor(),
])

dataset_treino = torchvision.datasets.CIFAR10(
    root='./datasets',
    train=True,
    download=False,
    transform=transformacao
)

# Pegamos 50 imagens pra testar o pipeline
amostras = [dataset_treino[i][0] for i in range(50)]
rotulos = [dataset_treino[i][1] for i in range(50)]
amostras = torch.stack(amostras)  # (50, 3, 32, 32)
rotulos = np.array(rotulos)

# ============================================
# 2️⃣ Extrair descritores locais (patches RGB)
# ============================================
def extrair_descritores_locais(imagem, tamanho_janela=4, passo=4):
    """
    Divide a imagem colorida em janelas (patches) e retorna vetores achatados.
    """
    C, H, W = imagem.shape
    descritores = []

    for y in range(0, H - tamanho_janela + 1, passo):
        for x in range(0, W - tamanho_janela + 1, passo):
            patch = imagem[:, y:y+tamanho_janela, x:x+tamanho_janela]
            descritores.append(patch.flatten().numpy())  # vetor 3*tam*tam = 48

    return np.stack(descritores)  # (num_patches, 48)


# Extrai descritores de todas as imagens
todos_descritores = []
for img in amostras:
    todos_descritores.append(extrair_descritores_locais(img))

todos_descritores = np.concatenate(todos_descritores, axis=0)
print("Forma total dos descritores:", todos_descritores.shape)
# Exemplo: (32000, 48)

# ============================================
# 3️⃣ Aplicar PCA para reduzir dimensionalidade
# ============================================
pca = PCA(n_components=24, random_state=42)  # reduz de 48 → 24 dimensões
descritores_pca = pca.fit_transform(todos_descritores)
print("Após PCA:", descritores_pca.shape)

# ============================================
# 4️⃣ Treinar KMeans (vocabulário visual)
# ============================================
num_centros = 16
kmeans = KMeans(n_clusters=num_centros, random_state=42)
kmeans.fit(descritores_pca)

# ============================================
# 5️⃣ Função VLAD (igual antes, só muda o input)
# ============================================
def calcular_vlad(descritores, centros):
    num_centros, dim = centros.shape
    vlad = np.zeros((num_centros, dim))

    atribuicoes = np.argmin(
        np.linalg.norm(descritores[:, None, :] - centros[None, :, :], axis=2),
        axis=1
    )

    for i in range(num_centros):
        descritores_cluster = descritores[atribuicoes == i]
        if len(descritores_cluster) > 0:
            vlad[i] = np.sum(descritores_cluster - centros[i], axis=0)

    # Normalização
    vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
    vlad = vlad.flatten()
    vlad = vlad / np.linalg.norm(vlad)
    return vlad

# ============================================
# 6️⃣ Gerar VLAD + PCA para cada imagem
# ============================================
vlads = []
inicio = 0
descritores_por_img = todos_descritores.shape[0] // len(amostras)

for img in amostras:
    descritores = extrair_descritores_locais(img)
    descritores_pca = pca.transform(descritores)
    v = calcular_vlad(descritores_pca, kmeans.cluster_centers_)
    vlads.append(v)

vlads = np.stack(vlads)
print("Forma final dos vetores VLAD+PCA:", vlads.shape)
# Esperado: (50, num_centros * n_components) = (50, 16*24 = 384)

# ============================================
# 7️⃣ Normalização e binarização pro WiSARD
# ============================================
vlads_norm = (vlads - vlads.min()) / (vlads.max() - vlads.min())
limiar = np.median(vlads_norm)
entradas_binarias = (vlads_norm > limiar).astype(int)
print("Entradas binárias:", entradas_binarias.shape)

# ============================================
# reinando WiSARD
wisard = wp.Wisard(8)
rotulos_str = [str(r) for r in rotulos]
wisard.train(entradas_binarias.tolist(), rotulos_str)

predicoes = wisard.classify(entradas_binarias.tolist())
metricas = Metricas(classes_reais=rotulos_str, classes_preditas=predicoes)

# Calcula e imprime
metricas.calcular_e_imprimir_metricas()
