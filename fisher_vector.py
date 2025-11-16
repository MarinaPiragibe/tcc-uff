from datetime import datetime
import logging
from time import time
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from torchvision.datasets import CIFAR10
from skimage.feature import fisher_vector, learn_gmm
import cv2

from utils.arquivo_utils import ArquivoUtils
from utils.enums.datasets_name_enum import DatasetName
from utils.enums.tipos_transformacao_wisard import TiposDeTransformacao
from utils.logger import Logger

# ---------- Parâmetros ----------

args = {
    "arq_ext_caract": "extracao_caracteristicas_resultados"
}
MAX_DESC_PER_IMG = 200
GMM_POOL_SIZE = 100_000
K = 16
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
PCA_DIM = 128

data_execucao = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

Logger.configurar_logger(
	nome_arquivo=f"fisher_vector_{data_execucao}.log"
)

def sift_desc_cv2(img_u8, sift):
    kpts, desc = sift.detectAndCompute(img_u8, None)

    # if len(kpts) > 0:
    #     img_kp = cv2.drawKeypoints(
    #         img_u8, kpts, None,
    #         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    #     )
    #     cv2.imwrite('sift_keypoints.jpg', img_kp)
    #     cv2.imwrite('img_original.jpg', img_u8)

    if desc is None or len(desc) == 0:
        return np.zeros((1, 128), dtype=np.float32)
    if len(desc) > MAX_DESC_PER_IMG:
        idx = rng.choice(len(desc), size=MAX_DESC_PER_IMG, replace=False)
        desc = desc[idx]
    return desc.astype(np.float32)
# =======================================================

# ---------- Carrega CIFAR-10 ----------
trainset = CIFAR10(root="./datasets", train=True,  download=False)

inicio_treino = time()

logging.info("[ÍNICIO] Extração de características pelo fisher vector do conjunto de treino")

# ---------- 1) GMM: coleta pool de descritores ----------
sift = cv2.SIFT_create()
pool_desc, total = [], 0
for i, (img_pil, _) in enumerate(trainset):
    img_np = np.array(img_pil)         # RGB uint8 (32x32x3)
    
    # img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(f"img_{i:05d}.jpg", img_bgr)
    # RGB uint8 (80x80x3)
    d = sift_desc_cv2(img_np, sift)
    falta = GMM_POOL_SIZE - total
    if falta <= 0:
        break
    if len(d) > falta:
        d = d[:falta]
    pool_desc.append(d)
    total += len(d)
    print(f"\r{i+1}/{len(trainset)}", end="", flush=True)

# ---------- (Opcional) PCA antes do GMM ----------
pool_mat = np.vstack(pool_desc)  # (N, 128)
pca = PCA(n_components=PCA_DIM, whiten=True, random_state=RANDOM_STATE)
pca.fit(pool_mat)
pool_desc = [pca.transform(d).astype(np.float32) for d in pool_desc]
D_FV = PCA_DIM
# ---------- Treina GMM ----------
gmm = learn_gmm(pool_desc, n_modes=K)

# ---------- 2) Fisher Vectors do TRAIN ----------
train_fvs, y_train = [], []
for i, (img_pil, y) in enumerate(trainset):
    img_np = np.array(img_pil)
    # img_u8 = to_rgb_u8(img_np)
    d = sift_desc_cv2(img_np, sift)
    d = pca.transform(d).astype(np.float32)
    fv = fisher_vector(d, gmm, improved=True).astype(np.float32)
    train_fvs.append(fv); y_train.append(y)
    print(f"\r{i+1}/{len(trainset)}", end="", flush=True)
train_fvs = np.vstack(train_fvs).astype(np.float32)
y_train = np.array(y_train)

fim_treino = time()

tempo_total_treino = fim_treino - inicio_treino
		
logging.info(f"[FIM] Extração de características pelo fisher vector do conjunto de treino concluido em {tempo_total_treino}")


# ---------- 3) Fisher Vectors do TEST ----------

testset  = CIFAR10(root="./datasets", train=False, download=False)

inicio_teste = time()

logging.info("[ÍNICIO] Extração de características pelo fisher vector do conjunto de teste")

test_fvs, y_test = [], []
for i,(img_pil, y) in enumerate(testset):
    img_np = np.array(img_pil)
    # img_u8 = to_rgb_u8(img_np)
    d = sift_desc_cv2(img_np, sift)
    d = pca.transform(d).astype(np.float32)
    fv = fisher_vector(d, gmm, improved=True).astype(np.float32)
    test_fvs.append(fv); y_test.append(y)
    print(f"\r{i+1}/{len(testset)}", end="", flush=True)

test_fvs = np.vstack(test_fvs).astype(np.float32)
y_test = np.array(y_test)

fim_teste = time()

tempo_total_teste = fim_teste - inicio_teste
		
logging.info(f"[FIM] Extração de características pelo fisher vector do conjunto de teste concluido em {tempo_total_teste}")


logging.info("[FIM] Extração de características pelo fisher vector")
logging.info("Dimensão do FV:", 2*K*D_FV + K)
logging.info(f"Tempo total da execução do fisher vectors {tempo_total_treino + tempo_total_teste}")

ArquivoUtils.salvar_features_imagem(
    nome_tecnica_ext=f"fisher_vector_sift_k{K}_pca{PCA_DIM}",
    nome_dataset=DatasetName.CIFAR10,
    dados_treino=train_fvs,
    classes_treino=y_train,
    dados_teste=test_fvs,
    classes_teste=y_test
)

dados_execucao = {
    "nome_tecnica": TiposDeTransformacao.FISHER_VECTOR.value,
    "tempo_treino": tempo_total_treino,
    "tempo_teste": tempo_total_teste,
    "tempo_total": tempo_total_treino + tempo_total_teste,
    "shape_treino": train_fvs.shape,
    "shape_teste": test_fvs.shape,
}

ArquivoUtils.salvar_csv(args, dados_execucao, args['arq_ext_caract'])
