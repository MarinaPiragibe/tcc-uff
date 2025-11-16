#!/usr/bin/env python3
"""
VLAD (tim-hilt/vlad) + PCA no estilo do StrideHD (args como dict)
-----------------------------------------------------------------
- Usa a biblioteca tim-hilt/vlad para codificar VLAD a partir de descritores
  Dense SIFT.
- Redução de dimensionalidade **opcional** no vetor VLAD com PCA.
- **Salva** as features com as mesmas chaves do seu pipeline
  (`dados_treino`, `classes_treino`, `dados_teste`, `classes_teste`) via
  `ArquivoUtils.salvar_features_imagem`.
- Configuração via **args = { ... }** (sem argparse), no mesmo espírito do seu código.

Como usar:
  1) Instale a lib:  git clone https://github.com/tim-hilt/vlad && cd vlad && pip install -e .
  2) Dependências:   pip install opencv-contrib-python scikit-learn torch torchvision numpy
  3) Ajuste o dict `args` e rode este arquivo.
"""
from __future__ import annotations
import logging
from time import time
import numpy as np
import torchvision

from vlad import VLAD
from utils.arquivo_utils import ArquivoUtils
from utils.enums.tipos_transformacao_wisard import TiposDeTransformacao
from wisard.sift import Sift

logging.basicConfig(level=logging.INFO,
					format="%(asctime)s - VLAD - %(levelname)s - %(message)s")


class Vlad:
	def __init__(self, args: dict):
		self.args = args
		self.vlad = VLAD(k=self.args["k"], norming="RN")


	def executar_e_salvar(self):

		logging.info(f"[ÍNICIO] Executando StrideHD no conjunto de treino")

		inicio_treino = time()
		trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True,  download=False, transform=None)

		classes_treino = np.array(list(trainset.targets))

		sift = Sift(trainset=trainset)

		# ================== TREINO ==================
		descritores_treino = sift.extrair_descritores_dataset(trainset, max_imgs=50_000, desc_target=100_000)
		descritores_treino_pca = sift.aplicar_pca(descritores_treino)
		self.vlad.fit(descritores_treino_pca)

		descritores_treino_full = sift.extrair_descritores_dataset(trainset)
		descritores_treino_full_pca = sift.transformar_pca(descritores_treino_full)
		vlad_treino = self.vlad.transform(descritores_treino_full_pca)

		# Filtra e empilha
		vlad_treino_validos = [f for f in vlad_treino if f is not None and np.size(f) > 0]
		classes_treino_validos = classes_treino[[f is not None and np.size(f) > 0 for f in vlad_treino]]

		dados_treino_np = ArquivoUtils.transformar_dados_memmap(
			dados=vlad_treino_validos,
			nome_arquivo="vlad_treino.dat"
		)

		classes_treino = np.array(classes_treino_validos, dtype=np.int64)
		fim_treino = time()

		tempo_total_treino = fim_treino - inicio_treino
		
		logging.info(f"[FIM] StrideHD concluído no conjunto de treino em {tempo_total_treino}")

		logging.info(f"[ÍNICIO] Executando StrideHD no conjunto de teste")

		inicio_teste = time()
		testset  = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=False, transform=None)
		classes_teste = np.array(list(testset.targets))

		descritores_teste = sift.extrair_descritores_dataset(testset)
		descritores_teste_pca = sift.transformar_pca(descritores_teste)
		vlad_teste = self.vlad.transform(descritores_teste_pca)

		vlad_teste_validos = [f for f in vlad_teste if f is not None and np.size(f) > 0]
		classes_teste_validos = classes_teste[[f is not None and np.size(f) > 0 for f in vlad_teste]]

		dados_teste_np = ArquivoUtils.transformar_dados_memmap(
			vlad_teste_validos,
			nome_arquivo="vlad_teste.dat"
		)

		classes_teste = np.array(classes_teste_validos, dtype=np.int64)
		fim_teste = time()

		tempo_total_teste = fim_teste - inicio_teste
		
		logging.info(f"[FIM] StrideHD concluído no conjunto de treino em {tempo_total_teste}")

		ArquivoUtils.salvar_features_imagem(
			nome_tecnica_ext=f"vlad_k{self.args['k']}",
			nome_dataset=self.args['dataset'],
			dados_treino=dados_treino_np,
			classes_treino=classes_treino,
			dados_teste=dados_teste_np,
			classes_teste=classes_teste
		)

		dados_execucao = {
			"nome_tecnica": TiposDeTransformacao.VLAD.value,
			"tempo_treino": tempo_total_treino,
			"tempo_teste": tempo_total_teste,
			"tempo_total": tempo_total_treino + tempo_total_teste,
			"shape_treino": dados_treino_np.shape,
			"shape_teste": dados_teste_np.shape,
		}

		ArquivoUtils.salvar_csv(self.args, dados_execucao, self.args['arq_ext_caract'])


if __name__ == "__main__":
	# === Exemplo de args no MESMO estilo do StrideHD ===
	args = {
		"k": 32,
		"nome_dataset": "CIFAR10",
		"saida_dir": "resultados",
		"nome_tecnica_prefixo": "vlad_timlib",
	}
	Vlad(args).executar_e_salvar()
