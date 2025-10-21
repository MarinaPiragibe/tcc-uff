from datetime import datetime
import logging
import random
from time import time
import torch
import torchvision
from torchwnn.encoding import DistributiveThermometer
import wisardpkg as wp

from utils.enums.tipos_transformacao_wisard import TiposDeTransformacao
from utils.logger import Logger
from utils.metricas import Metricas
from wisard.wisard_image_transform import WisardImageTransform

from torch.utils.data import DataLoader, Subset

from wisard.wisard_utils import executar_wisard


args = {
	"modelo_base": "wisard",
	"tamanho_lote": 32,
	"dataset":"cifar-10",
	"debug": False,
	"tipo_transformacao": TiposDeTransformacao.BASICA,
	"tamanhos_tuplas": [16, 20, 32, 64],
	"debug": False
}
args["data_execucao"] = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

Logger.configurar_logger(
	nome_arquivo=f"{args['modelo_base']}_application_{args['data_execucao']}.log"
)

logging.info(f"Inicializando o modelo da wisard com os seguintes argumentos: {args}")
torch.set_num_threads(1)

logging.info(f"Carregando dataset do cifar10 com dowload:False e transform: {args['tipo_transformacao'].value}")
train_set = torchvision.datasets.FashionMNIST('./', train=True, download=True, transform=WisardImageTransform.get_image_transformation(args["tipo_transformacao"]))
test_set = torchvision.datasets.FashionMNIST('./', train=False, download=True, transform=WisardImageTransform.get_image_transformation(args["tipo_transformacao"]))

if args['debug']:
	logging.info("Iniciando modelo no modo de depuração com 1000 entradas para treino e 200 para teste")
	train_indices = random.sample(range(len(train_set)), 1000)
	test_indices = random.sample(range(len(test_set)), 20)
	train_set = Subset(train_set, train_indices)
	test_set = Subset(test_set, test_indices)

train_loader = torch.utils.data.DataLoader(
	train_set, batch_size=args['tamanho_lote'], shuffle=True, drop_last=False, num_workers=0
)
test_loader = torch.utils.data.DataLoader(
	test_set, batch_size=args['tamanho_lote'], shuffle=False,drop_last=False, num_workers=0
)

logging.info(f"Configurando termômetro distributivo")
termometro = DistributiveThermometer(10)
sample_x, _ = next(iter(train_loader))
termometro.fit(sample_x)


for tamanho in args['tamanhos_tuplas']:
	logging.info(f"[TUPLA {tamanho}] Iniciando treinamento do WisardPKG")
	modelo = wp.Wisard(tamanho)
	
	executar_wisard(
		modelo,
		train_loader,
		test_loader,
		termometro,
		tamanho,
		args['tipo_transformacao'],
		stride_hd=None
		
	)
	