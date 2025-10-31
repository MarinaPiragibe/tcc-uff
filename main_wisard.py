from datetime import datetime
import logging
import torch
from torchwnn.encoding import DistributiveThermometer
import wisardpkg as wp

from utils.arquivo_utils import ArquivoUtils
from utils.enums.datasets_name_enum import DatasetName
from utils.enums.tipos_transformacao_wisard import TiposDeTransformacao
from utils.logger import Logger
from wisard.wisard_model import WisardModel


args = {
	"modelo_base": "wisard",
	"tamanho_lote": 32,
	"dataset": DatasetName.CIFAR10,
	"download_dataset": False,
	"tipo_transformacao": TiposDeTransformacao.VLAD,
	"tamanhos_tuplas": [8, 12, 16, 20, 32, 64],
	"num_bits_termometro": 12,
	"debug": False
}
args["data_execucao"] = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

Logger.configurar_logger(
	nome_arquivo=f"{args['modelo_base']}_application_{args['data_execucao']}.log"
)

logging.info(f"Inicializando o modelo da wisard com os seguintes argumentos: {args}")
torch.set_num_threads(1)

# 2) (Opcional) Criar DataLoaders
dados_treino, classes_treino, dados_teste, classes_teste = ArquivoUtils.carregar_caracteristicas_salvas("features/convnext_DatasetName.CIFAR10_features.npz")

logging.info(f"Configurando termômetro distributivo")

def make_chunks(X, y, n=32):
    assert len(X) == len(y), "X e y têm comprimentos diferentes!"
    return [(X[i:i+n], y[i:i+n]) for i in range(0, len(X), n)]

dados_treino_lote = make_chunks(dados_treino, classes_treino, args['tamanho_lote'])
dados_teste_lote = make_chunks(dados_teste,  classes_teste,  args['tamanho_lote'])  

# Inicializa termômetro
termometro = DistributiveThermometer(args['num_bits_termometro'])
termometro.fit(dados_treino_lote[0][0])


# Agora cria o modelo com o fisher_transform e termometro treinados
for tamanho in args['tamanhos_tuplas']:
	modelo_wisard = wp.Wisard(tamanho)

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
	modelo.executar_modelo()
