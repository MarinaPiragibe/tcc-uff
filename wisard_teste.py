from datetime import datetime
import logging
import random
from time import time
import torch
import torchvision
from torchwnn.encoding import DistributiveThermometer
import wisardpkg as wp

from utils.logger import Logger
from utils.metricas import Metricas
from utils.tipos_transformacao_wisard import TiposDeTransformacao
from wisard.wisard_image_transform import WisardImageTransform

from torch.utils.data import DataLoader, Subset


args = {
	"modelo_base": "wisard",
	"tamanho_lote": 32,
	"dataset":"cifar-10-3-thresholds",
	"debug": False,
	"tipo_transformacao": TiposDeTransformacao.BASICA,
	"debug": False
}
args["data_execucao"] = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

Logger.configurar_logger(
	nome_arquivo=f"{args['modelo_base']}_application_{args['data_execucao']}.log"
)

logging.info(f"Inicializando o modelo da wisard com os seguintes argumentos: {args}")
torch.set_num_threads(1)

batch_size = 128 
adress_list = [16, 20, 32, 64]

logging.info(f"Carregando dataset do cifar10 com dowload:False e transform: {args['tipo_transformacao'].value}")
train_set = torchvision.datasets.CIFAR10('./', train=True, download=False, transform=WisardImageTransform.get_image_transformation(args["tipo_transformacao"]))
test_set = torchvision.datasets.CIFAR10('./', train=False, download=False, transform=WisardImageTransform.get_image_transformation(args["tipo_transformacao"]))

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
thermometer = DistributiveThermometer(10)
sample_x, _ = next(iter(train_loader))
thermometer.fit(sample_x)


for tupple_size in adress_list:
	logging.info(f"[TUPLA {tupple_size}] Iniciando treinamento do WisardPKG")
	model = wp.Wisard(tupple_size)
	
	inicio_treino = time()

	for k, (x_batch, y_batch) in enumerate(train_loader):
		print(f"\r{k+1}/{len(train_loader)}", end="", flush=True)
		x_batch_bin = thermometer.binarize(x_batch).flatten(start_dim=1).numpy()
		y_batch = y_batch.numpy().astype(str)
		model.train(x_batch_bin, y_batch)
	final_treino = time()

	tempo_total_treino = final_treino - inicio_treino

	logging.info(f"[TUPLA {tupple_size}] Fim do treinamento do WisardPKG. Tempo total do treino: {tempo_total_treino}")

	classes_preditas = []
	classes_reais = []

	logging.info(f"[TUPLA {tupple_size}] Iniciando teste do WisardPKG ")

	inicio_teste = time()

	for k, (x_batch, y_batch) in enumerate(test_loader):
		print(f"\r{k+1}/{len(test_loader)}", end="", flush=True)
		x_batch_bin = thermometer.binarize(x_batch).flatten(start_dim=1).numpy()
		y_batch = y_batch.numpy().astype(str)
		preds = model.classify(x_batch_bin)
		classes_preditas.extend(preds)
		classes_reais.extend(y_batch)

	final_teste = time()
	tempo_total_teste = final_teste - inicio_teste

	logging.info(f"[TUPLA {tupple_size}] Fim do teste do WisardPKG . Tempo total de inferência: {tempo_total_teste}")


	logging.info(f"[TUPLA {tupple_size}] Execução do modelo  concluída em {tempo_total_treino + tempo_total_teste}")
	logging.info(f"[TUPLA {tupple_size}] Tempo de execução do treino: {tempo_total_treino}")
	logging.info(f"[TUPLA {tupple_size}] Tempo de execução do teste: {tempo_total_teste}")

	metricas = Metricas(classes_reais=classes_reais, classes_preditas=classes_preditas)
	logging.info(f"Calculando métricas de desempenho")
	metricas.calcular_e_imprimir_metricas()
