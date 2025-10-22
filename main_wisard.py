from datetime import datetime
import logging
import random
import torch
from torchwnn.encoding import DistributiveThermometer
import wisardpkg as wp

from utils.dataset_utils import DatasetUtils
from utils.enums.datasets_name_enum import DatasetName
from utils.enums.tipos_transformacao_wisard import TiposDeTransformacao
from utils.logger import Logger
from wisard.stride_hd import StrideHD

from torch.utils.data import Subset

from wisard.wisard_utils import WisardModel


args = {
	"modelo_base": "wisard",
	"tamanho_lote": 32,
	"dataset": DatasetName.CIFAR10,
	"download_dataset": False,
	"tipo_transformacao": TiposDeTransformacao.STRIDE_HD,
	"tamanhos_tuplas": [16, 20, 32, 64],
	"num_bits_termometro": 10,
	"debug": True
}
args["data_execucao"] = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

Logger.configurar_logger(
	nome_arquivo=f"{args['modelo_base']}_application_{args['data_execucao']}.log"
)

logging.info(f"Inicializando o modelo da wisard com os seguintes argumentos: {args}")
torch.set_num_threads(1)

logging.info(f"Carregando dataset do {args['dataset'].value} com dowload:False e transform: {args['tipo_transformacao'].value}")
dados_treino, dados_teste  = DatasetUtils.carregar_dados_treinamento(args['dataset'], args['tipo_transformacao'], args['download_dataset'])

if args['debug']:
	logging.info("Iniciando modelo no modo de depuração com 1000 entradas para treino e 200 para teste")
	indices_treino = random.sample(range(len(dados_treino)), 1000)
	indice_teste = random.sample(range(len(dados_teste)), 20)

	dados_treino = Subset(dados_treino, indices_treino)
	dados_teste = Subset(dados_teste, indice_teste)

train_loader = torch.utils.data.DataLoader(
	dados_treino, batch_size=args['tamanho_lote'], shuffle=True, drop_last=False, num_workers=0
)
test_loader = torch.utils.data.DataLoader(
	dados_teste, batch_size=args['tamanho_lote'], shuffle=False,drop_last=False, num_workers=0
)

logging.info(f"Configurando termômetro distributivo")
termometro = DistributiveThermometer(args['num_bits_termometro'])

stride_hd = StrideHD(window_size=(4,4), stride=1, pool_size=(2,2))

def calcular_dados_para_termometro(args, stride_hd: StrideHD):
	dados_para_termometro, _ = next(iter(train_loader))

	if args['tipo_transformacao'] == TiposDeTransformacao.STRIDE_HD:
		pooled_dados_para_termometro = stride_hd.extract_and_pool(dados_para_termometro)
		B, N, C, H_pool, W_pool = pooled_dados_para_termometro.shape
		dados_para_termometro = pooled_dados_para_termometro.contiguous().view(B, -1)
	
	return dados_para_termometro

dados_para_termometro = calcular_dados_para_termometro(args, stride_hd)
termometro.fit(dados_para_termometro)

for tamanho in args['tamanhos_tuplas']:
	logging.info(f"[TUPLA {tamanho}] Iniciando treinamento do WisardPKG")
	modelo_wisard = wp.Wisard(tamanho)

	modelo = WisardModel(
		modelo=modelo_wisard,
		tamanho_tupla=tamanho,
		train_loader=train_loader,
		test_loader=test_loader,
		termometro=termometro,
		stride_hd=stride_hd,
		args=args
	)

	modelo.executar_modelo()
	
	# executar_wisard(
	# 	modelo,
	# 	train_loader,
	# 	test_loader,
	# 	termometro,
	# 	tamanho,
	# 	args['tipo_transformacao'],
	# 	stride_hd=None
		
	# )
	