from datetime import datetime
import logging
import random

import torch
from utils.dataset_utils import DatasetUtils
from utils.enums.datasets_name_enum import DatasetName
from utils.enums.tipos_transformacao_wisard import TiposDeTransformacao
from utils.logger import Logger
from torch.utils.data import Subset

from old.stride_hd import StrideHD
from wisard.fisher_vector import FisherVector



args = {
	"tamanho_lote": 32,
	"dataset": DatasetName.CIFAR10,
	"download_dataset": False,
	"tipo_transformacao": TiposDeTransformacao.FISHER_VECTOR,
	"debug": False
}
args["data_execucao"] = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

Logger.configurar_logger(
	nome_arquivo=f"{args['tipo_transformacao'].value}_{args['dataset'].value}_{args['data_execucao']}.log"
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

match(args['tipo_transformacao']):

	case TiposDeTransformacao.STRIDE_HD:
		stride_hd = StrideHD(window_size=(2,2), stride=1)
		stride_hd.executar_e_salvar(train_loader, test_loader, args['dataset'])

	case TiposDeTransformacao.FISHER_VECTOR:
		fv = FisherVector(
			train_set=dados_treino,
			test_set=dados_teste,
			args=args
		)
		fv.executar_e_salvar()

	