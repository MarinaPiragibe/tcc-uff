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
from wisard.stride_hd import StrideHD
from wisard.wisard_image_transform import WisardImageTransform

from torch.utils.data import DataLoader, Subset


args = {
    "modelo_base": "wisard",
    "tamanho_lote": 32,
    "dataset": "cifar-10-3-thresholds",
    "debug": False,
    "tipo_transformacao": TiposDeTransformacao.STRIDE_HD,
}
args["data_execucao"] = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

Logger.configurar_logger(
    nome_arquivo=f"{args['modelo_base']}_application_{args['data_execucao']}.log"
)

logging.info(f"Inicializando o modelo da wisard com os seguintes argumentos: {args}")
torch.set_num_threads(1)

adress_list = [8, 12, 16, 20, 32, 64]

logging.info(f"Carregando dataset do CIFAR-10")
train_set = torchvision.datasets.CIFAR10(
    './', train=True, download=False,
    transform=WisardImageTransform.get_image_transformation(args["tipo_transformacao"])
)
test_set = torchvision.datasets.CIFAR10(
    './', train=False, download=False,
    transform=WisardImageTransform.get_image_transformation(args["tipo_transformacao"])
)

if args['debug']:
    logging.info("Modo depuração: 1000 treino / 20 teste")
    train_indices = random.sample(range(len(train_set)), 1000)
    test_indices = random.sample(range(len(test_set)), 20)
    train_set = Subset(train_set, train_indices)
    test_set = Subset(test_set, test_indices)

train_loader = DataLoader(train_set, batch_size=args['tamanho_lote'], shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=args['tamanho_lote'], shuffle=False, num_workers=0)

stride_hd = StrideHD(window_size=(8,8), stride=2, pool_size=(2,2))

logging.info("Configurando termômetro distributivo")
thermometer = DistributiveThermometer(10)

# Ajustar termômetro após StrideHD
sample_x, _ = next(iter(train_loader))
pooled_sample = stride_hd.extract_and_pool(sample_x)
B, N, C, H_pool, W_pool = pooled_sample.shape
x_for_fit = pooled_sample.contiguous().view(B, -1)
thermometer.fit(x_for_fit)

for tupple_size in adress_list:
    logging.info(f"[TUPLA {tupple_size}] Iniciando treinamento do WisardPKG")
    model = wp.Wisard(tupple_size)

    inicio_treino = time()
    for k, (x_batch, y_batch) in enumerate(train_loader):
        print(f"\rTreino {k+1}/{len(train_loader)}", end="", flush=True)

        pooled_windows = stride_hd.extract_and_pool(x_batch)
        B, N, C, H_pool, W_pool = pooled_windows.shape
        x_for_wisard = pooled_windows.contiguous().view(B, -1)

        # Binariza e achata
        x_batch_bin = thermometer.binarize(x_for_wisard).numpy()
        x_batch_bin = x_batch_bin.reshape(B, -1).astype(int).tolist()

        # Converte rótulos para strings
        y_batch_list = y_batch.numpy().astype(str).tolist()

        model.train(x_batch_bin, y_batch_list)
    final_treino = time()
    logging.info(f"\n[TUPLA {tupple_size}] Treinamento concluído em {final_treino - inicio_treino:.2f}s")

    # --- Teste ---
    classes_preditas = []
    classes_reais = []

    inicio_teste = time()
    for k, (x_batch, y_batch) in enumerate(test_loader):
        print(f"\rTeste {k+1}/{len(test_loader)}", end="", flush=True)

        pooled_windows = stride_hd.extract_and_pool(x_batch)
        B, N, C, H_pool, W_pool = pooled_windows.shape
        x_for_wisard = pooled_windows.contiguous().view(B, -1)

        x_batch_bin = thermometer.binarize(x_for_wisard).numpy()
        x_batch_bin = x_batch_bin.reshape(B, -1).astype(int).tolist()
        y_batch_list = y_batch.numpy().astype(str).tolist()

        preds = model.classify(x_batch_bin)

        # Todas as classes como strings
        classes_preditas.extend(preds)
        classes_reais.extend(y_batch_list)
    final_teste = time()

    logging.info(f"\n[TUPLA {tupple_size}] Teste concluído em {final_teste - inicio_teste:.2f}s")
    logging.info(f"[TUPLA {tupple_size}] Execução total: {final_treino - inicio_treino + final_teste - inicio_teste:.2f}s")

    metricas = Metricas(classes_reais=classes_reais, classes_preditas=classes_preditas)
    logging.info("Calculando métricas de desempenho")
    metricas.calcular_e_imprimir_metricas()
