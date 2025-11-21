from datetime import datetime
import logging
from pathlib import Path
from codecarbon import EmissionsTracker
import numpy as np
import torch
from torchwnn.encoding import DistributiveThermometer
import wisardpkg as wp

from utils.arquivo_utils import ArquivoUtils
from utils.dataset_utils import DatasetUtils
from utils.enums.datasets_name_enum import DatasetName
from utils.enums.tipos_transformacao_wisard import TiposDeTransformacao
from utils.logger import Logger
from wisard.wisard_model import WisardModel


args = {
    "num_exec": 1,
	"tamanho_lote": 32 ,
	"dataset": DatasetName.CIFAR10,
	"download_dataset": False,
	"tamanhos_tuplas": [6, 8, 12, 20, 32, 64],
	"num_bits_termometro": 8,
	"debug": False,
    "pasta_features": Path("features"),
	"tipo_transformacao": TiposDeTransformacao.ESCALA_DE_CINZA,
	"carregar_dados_salvos": False
}
args["data_execucao"] = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

def executar_wisard(tecnica_ext_feat):
	logging.info(f"Inicializando o modelo {args['modelo_base']} com os seguintes args: {args}")
	torch.set_num_threads(1)

	dados_treino = dados_teste = classes_treino = classes_teste = None

	if args["carregar_dados_salvos"]:
		dados_treino, classes_treino, dados_teste, classes_teste = ArquivoUtils.carregar_caracteristicas_salvas(arquivo)

	else:
		dataset_treino, dataset_teste = DatasetUtils.carregar_dados_treinamento(
			args["dataset"],
			args["tipo_transformacao"],
			args["download_dataset"]
		)

		def extrair(dataset):
			dados = []
			classes = []
			for x, y in dataset:
				dados.append(x.flatten().numpy())   
				classes.append(str(y))
			return np.array(dados, dtype=np.float32), np.array(classes)

		dados_treino, classes_treino = extrair(dataset_treino)
		dados_teste,  classes_teste  = extrair(dataset_teste)

	def make_chunks(X, y, n=32):
		assert len(X) == len(y), "X e y têm comprimentos diferentes!"
		return [(X[i:i+n], y[i:i+n]) for i in range(0, len(X), n)]

	dados_treino_lote = make_chunks(dados_treino, classes_treino, args['tamanho_lote'])
	dados_teste_lote = make_chunks(dados_teste,  classes_teste,  args['tamanho_lote'])  

	logging.info(f"Configurando termômetro distributivo")

	# Inicializa termômetro
	termometro = DistributiveThermometer(args['num_bits_termometro'])
	termometro.fit(dados_treino_lote[0][0])

	for tamanho in args['tamanhos_tuplas']:
		for i in range(args['num_exec']):
			tracker = EmissionsTracker(
                    project_name=f"{args['modelo_base']}_tupla{tamanho}",
                    output_dir="results/code_carbon",
					output_file=f"{args['modelo_base']}_emissions.csv",
					log_level="error"
                )
			logging.info(f"[EXECUCAO {i+1}] [TUPLA {tamanho}] Iniciando wisard com {tecnica_ext_feat}")
			modelo_wisard = wp.Wisard(tamanho)
			
			logging.info(f"[EXECUCAO {i+1}] [TUPLA {tamanho}] Iniciando modelo da wisard")
			modelo = WisardModel(
				modelo=modelo_wisard,
				tamanho_tupla=tamanho,
				dados_treino=dados_treino_lote,
				dados_teste=dados_teste_lote,
				classes_treino=classes_treino,
				classes_teste=classes_teste,
				termometro=termometro,
				args=args,
			)

			tracker.start()
			modelo.executar_modelo(num_execucao=i+1, tecnica_ext_feat=tecnica_ext_feat)
			tracker.stop()

if args["carregar_dados_salvos"]:
	for arquivo in args['pasta_features'].iterdir():

		tecnica_ext_feat = arquivo.name.split("_")[0]
		args['modelo_base'] = f"wisard_{tecnica_ext_feat}"
		Logger.configurar_logger(
			nome_arquivo=f"{args['modelo_base']}_{args['data_execucao']}.log"
		)
		executar_wisard(tecnica_ext_feat)
else:
	tecnica_ext_feat = args["tipo_transformacao"].value
	args['modelo_base'] = f"wisard_{tecnica_ext_feat}"
	Logger.configurar_logger(
			nome_arquivo=f"{args['modelo_base']}_{args['data_execucao']}.log"
		)
	executar_wisard(tecnica_ext_feat)