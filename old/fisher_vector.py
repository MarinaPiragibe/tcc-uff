import logging
from enum import Enum
import cv2
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import ORB, fisher_vector, learn_gmm
import numpy as np
from utils.arquivo_utils import ArquivoUtils

class TipoDescritor(Enum):
    SIFT = "sift"
    SIFT_COLORIDO = "sift_colorido"
    ORB = 'orb'

class FisherVector:
    def __init__(self, train_loader, test_loader, dataset):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.dataset = dataset

    def converter_dados_para_np(self, loader):
        logging.info("Convertendo DataLoader para NumPy arrays...")
        dados_convertidos = []
        classes_convertidas = []
        for k, (dado, classe) in enumerate(loader):
            dado = dado.permute(0, 2, 3, 1).numpy()  # [B,C,H,W] → [B,H,W,C]
            dados_convertidos.append(dado)
            classes_convertidas.append(classe.numpy())
        dados_convertidos = np.concatenate(dados_convertidos)
        classes_convertidas = np.concatenate(classes_convertidas)
        logging.info(f"Conversão concluída: {dados_convertidos.shape[0]} imagens.")
        return dados_convertidos, classes_convertidas

    @staticmethod
    def converter_escala_de_cinza_e_redimensionar(dados):
        logging.info("Convertendo imagens para escala de cinza e redimensionando para 64x64...")
        dados_escala_cinza = np.array([rgb2gray(dado) for dado in dados])
        dados_redimensionados = np.array([resize(dado, (64, 64)) for dado in dados_escala_cinza])  # 80->64
        logging.info("Conversão e redimensionamento concluídos.")
        return dados_redimensionados

    @staticmethod
    def extrair_descritores_orb(dados):
        descritores = []
        # ALTERADO: 50 -> 32 para reduzir RAM/tempo
        orb = ORB(n_keypoints=32, harris_k=0.01, fast_threshold=0.0)

        for idx, dado in enumerate(dados):
            img = dado.astype(np.float32)
            img = np.clip(img, 0.0, 1.0)

            try:
                orb.detect_and_extract(img)
                desc = orb.descriptors
                if desc is None or desc.size == 0:
                    # fallback: mantém dimensão [5, 256]
                    desc = np.zeros((5, 256), dtype=np.float32)
                else:
                    # skimage retorna bool; converte p/ float32
                    desc = desc.astype(np.float32, copy=False)

                descritores.append(desc)

            except Exception as e:
                logging.warning(f"ORB sem features na imagem {idx}: {e}. Usando fallback zeros.")
                descritores.append(np.zeros((5, 256), dtype=np.float32))

        return descritores

    @staticmethod
    def extrair_descritores_sift(dados, nfeatures=20):
        sift = cv2.SIFT_create(nfeatures=nfeatures)
        descritores = []
        for idx, dado in enumerate(dados):
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
        for idx, dado in enumerate(dados):
            descs = []
            for c in range(3):  # R, G, B
                canal = (dado[..., c] * 255).astype(np.uint8)
                kp, desc = sift.detectAndCompute(canal, None)
                if desc is not None:
                    descs.append(desc)
            if descs:
                descritores.append(np.vstack(descs).astype("float32"))
            else:
                descritores.append(np.zeros((5, 128), dtype=np.float32))
        return descritores

    @staticmethod
    def treinar_gmm(descritores, n_modes=8, max_desc_total=50000, max_por_imagem=64):
        """
        Treino do GMM robusto e simples:
        - Converte para float64
        - Remove linhas 100% zero (fallback)
        - Limita descritores por imagem e o total
        - Ajusta n_modes automaticamente se a amostra for pequena
        """
        logging.info("Treinando GMM (simples e robusto)...")
        amostra = []
        total = 0

        for d in descritores:
            if d is None or len(d) == 0:
                continue

            # garante float64 (ORB pode vir bool)
            d = d.astype(np.float64, copy=False)

            # remove linhas 100% zero (fallback)
            mask = ~np.all(d == 0.0, axis=1)
            d = d[mask]
            if d.size == 0:
                continue

            # limita por imagem para não enviesar
            if len(d) > max_por_imagem:
                idx = np.random.choice(len(d), size=max_por_imagem, replace=False)
                d = d[idx]

            amostra.append(d)
            total += len(d)
            if total >= max_desc_total:
                break

        if not amostra:
            raise RuntimeError("Sem descritores válidos para treinar o GMM.")

        amostra = np.vstack(amostra).astype(np.float64, copy=False)
        logging.info(f"Amostra p/ GMM: {amostra.shape}")

        # Ajuste automático de K se a amostra for pequena
        min_por_comp = 10
        max_k = max(1, amostra.shape[0] // min_por_comp)
        if n_modes > max_k:
            logging.warning(f"n_modes={n_modes} alto para {amostra.shape[0]} descritores. Ajustando para {max_k}.")
            n_modes = max_k

        gmm = learn_gmm(amostra, n_modes=n_modes)
        logging.info(f"GMM treinado com {n_modes} modos.")
        return gmm


    @staticmethod
    def calcular_fisher_vectors(descritores, gmm):
        """
        Cálculo de Fisher Vectors tolerante:
        - Converte para float64
        - Remove linhas 100% zero
        - Se ficar vazio ou a dimensão não bater com a do GMM, retorna um FV de zeros do tamanho correto
        - Nunca quebra o pipeline
        """
        logging.info("Calculando Fisher Vectors (tolerante a vazios)...")

        # Dimensão de descritor esperada pelo GMM (dict do skimage ou objeto sklearn)
        means = gmm["means"] if isinstance(gmm, dict) else gmm.means_
        dim_d = int(means.shape[1])

        # Descobre a dimensão do FV com um dummy mínimo
        fv_dummy = fisher_vector(np.zeros((1, dim_d), dtype=np.float64), gmm)
        dim_fv = fv_dummy.shape[0]
        
        # Descobre a dimensão do FV com um dummy mínimo
        fv_dummy = fisher_vector(np.zeros((1, dim_d), dtype=np.float64), gmm)
        dim_fv = fv_dummy.shape[0]

        fvs = []
        for i, d in enumerate(descritores):
            try:
                if d is None or len(d) == 0:
                    fvs.append(np.zeros(dim_fv, dtype=np.float64))
                    continue

                # float64 e remove linhas-zeradas
                d = d.astype(np.float64, copy=False)
                d = d[~np.all(d == 0.0, axis=1)]
                if d.size == 0 or d.shape[1] != dim_d:
                    fvs.append(np.zeros(dim_fv, dtype=np.float64))
                    continue

                fv = fisher_vector(d, gmm).astype(np.float64, copy=False)
                fvs.append(fv)

            except Exception as e:
                logging.warning(f"Falha ao gerar FV da amostra {i}: {e}. Usando zero-FV.")
                fvs.append(np.zeros(dim_fv, dtype=np.float64))

        fvs = np.vstack(fvs).astype(np.float64, copy=False)
        n_zero = int(np.sum(np.all(fvs == 0.0, axis=1)))
        logging.info(f"FVs zero: {n_zero}/{fvs.shape[0]}")
        logging.info(f"Fisher Vectors calculados: {fvs.shape}")
        return fvs

    def _extrair_descritores_e_classes_do_loader(self, loader, tipo_descritor: TipoDescritor, nfeatures_sift=20):
        """
        Extrai descritores e classes iterando por batches, sem materializar todo o dataset em RAM.
        Mantém a sua lógica de extração (reutiliza os métodos existentes).
        """
        descritores = []
        classes = []

        logging.info(f"[INICIO] Extraindo descritores {tipo_descritor.value}")

        for k, (imgs, cls) in enumerate(loader):
            # [B,C,H,W] -> [B,H,W,C] numpy
            batch = imgs.permute(0, 2, 3, 1).numpy()

            match(tipo_descritor):
                case TipoDescritor.ORB:
                    # ORB requer cinza+resize — ALTERADO: 80 -> 64
                    batch_gray = [rgb2gray(im) for im in batch]
                    batch_res = [resize(im, (64, 64)) for im in batch_gray]
                    descritores.extend(self.extrair_descritores_orb(batch_res))

                case TipoDescritor.SIFT:
                    # Mantemos seu fluxo original (converte para uint8 dentro do extrator)
                    descritores.extend(self.extrair_descritores_sift(batch, nfeatures=nfeatures_sift))

                case TipoDescritor.SIFT_COLORIDO:
                    descritores.extend(self.extrair_descritores_sift_colorido(batch, nfeatures=nfeatures_sift))

            classes.append(cls.numpy())

            print(f"\r{k+1}/{len(loader)}", end="", flush=True)

        logging.info(f"[FIM] Extraindo descritores {tipo_descritor.value}")

        classes = np.concatenate(classes)
        return descritores, classes

    def executar_fisher_vector(self, tipo_descritor: TipoDescritor):
        logging.info(f"Executando pipeline Fisher Vector para {tipo_descritor.value}...")

        # === 1) Extraia APENAS o TREINO primeiro
        descritores_treino, classes_treino = self._extrair_descritores_e_classes_do_loader(
            self.train_loader, tipo_descritor, nfeatures_sift=20
        )

        # === 2) Treine o GMM com o TREINO
        gmm = self.treinar_gmm(descritores_treino)

        # === 3) FV do TREINO
        fisher_vector_treino = self.calcular_fisher_vectors(descritores_treino, gmm)

        # === 4) Liberar descritores do TREINO para reduzir pico de RAM
        del descritores_treino
        import gc; gc.collect()

        # === 5) Agora extraia o TESTE
        descritores_teste, classes_teste = self._extrair_descritores_e_classes_do_loader(
            self.test_loader, tipo_descritor, nfeatures_sift=20
        )

        # === 6) FV do TESTE
        fisher_vector_teste = self.calcular_fisher_vectors(descritores_teste, gmm)

        # === 7) Salvar
        logging.info("Salvando features para uso posterior...")
        ArquivoUtils.salvar_features_imagem(
            nome_tecnica_ext=tipo_descritor.value,
            nome_dataset=self.dataset.value,
            dados_treino=fisher_vector_treino,
            classes_treino=classes_treino,
            dados_teste=fisher_vector_teste,
            classes_teste=classes_teste
        )
        logging.info("Pipeline Fisher Vector concluído com sucesso.")
