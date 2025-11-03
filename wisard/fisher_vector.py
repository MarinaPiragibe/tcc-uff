from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.decomposition import PCA  # (opcional)

from torchvision.datasets import CIFAR10
from skimage.transform import resize
from skimage.feature import fisher_vector, learn_gmm
import cv2

from utils.arquivo_utils import ArquivoUtils

# ---------- Parâmetros ----------
RESIZE_TO = 80
MAX_DESC_PER_IMG = 200
GMM_POOL_SIZE = 100_000
K = 16
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

USE_PCA = True
PCA_DIM = 64

# ======= ALTERAÇÃO MÍNIMA: manter RGB, sem cinza =======
def to_rgb_u8(img_rgb_np):
    # resize em RGB e converte para uint8; SIFT do OpenCV aceita 3 canais e converte internamente
    img_resized = resize(img_rgb_np, (RESIZE_TO, RESIZE_TO), anti_aliasing=True)  # float [0,1]
    return (img_resized * 255).astype(np.uint8)

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
testset  = CIFAR10(root="./datasets", train=False, download=False)

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
if USE_PCA:
    pool_mat = np.vstack(pool_desc)  # (N, 128)
    pca = PCA(n_components=PCA_DIM, whiten=True, random_state=RANDOM_STATE)
    pca.fit(pool_mat)
    pool_desc = [pca.transform(d).astype(np.float32) for d in pool_desc]
    D_FV = PCA_DIM
else:
    D_FV = 128

# ---------- Treina GMM ----------
gmm = learn_gmm(pool_desc, n_modes=K)

# ---------- 2) Fisher Vectors do TRAIN ----------
train_fvs, y_train = [], []
for i, (img_pil, y) in enumerate(trainset):
    img_np = np.array(img_pil)
    # img_u8 = to_rgb_u8(img_np)
    d = sift_desc_cv2(img_np, sift)
    if USE_PCA:
        d = pca.transform(d).astype(np.float32)
    fv = fisher_vector(d, gmm, improved=True).astype(np.float32)
    train_fvs.append(fv); y_train.append(y)
    print(f"\r{i+1}/{len(trainset)}", end="", flush=True)
train_fvs = np.vstack(train_fvs).astype(np.float32)
y_train = np.array(y_train)

# ---------- 3) Fisher Vectors do TEST ----------
test_fvs, y_test = [], []
for i,(img_pil, y) in enumerate(testset):
    img_np = np.array(img_pil)
    # img_u8 = to_rgb_u8(img_np)
    d = sift_desc_cv2(img_np, sift)
    if USE_PCA:
        d = pca.transform(d).astype(np.float32)
    fv = fisher_vector(d, gmm, improved=True).astype(np.float32)
    test_fvs.append(fv); y_test.append(y)
    print(f"\r{i+1}/{len(testset)}", end="", flush=True)

test_fvs = np.vstack(test_fvs).astype(np.float32)
y_test = np.array(y_test)

print("Dimensão do FV:", 2*K*D_FV + K)

nome_dataset = "CIFAR10"

ArquivoUtils.salvar_features_imagem(
    nome_tecnica_ext=f"sift_fv_k{K}" + (f"_pca{PCA_DIM}" if USE_PCA else "" + "noi_resize"),
    nome_dataset=nome_dataset,
    dados_treino=train_fvs,
    classes_treino=y_train,
    dados_teste=test_fvs,
    classes_teste=y_test
)