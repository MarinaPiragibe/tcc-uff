import random
import torch
from torch import nn

from torch.utils.data import DataLoader, Subset

from cnn.modelo import Modelo
from utils.logger import Logger

import logging
from torchvision import datasets

def criar_modelo_cnn(args, transform, num_execucao):

    # Logger.configurar_logger(nome_arquivo=f"cnn_application_{args['modelo_base']}_{args['data_execucao']}.log")

    logging.info(f"[EXECUCAO {num_execucao}] Utilizando como base do fine tuning o modelo: {args['modelo_base']}")

    if torch.cuda.is_available():
        args['dispositivo'] = torch.device('cuda')
        logging.info("CUDA disponível. Usando GPU para treinamento.")
    else:
        args['dispositivo'] = torch.device('cpu')
        logging.info("CUDA não disponível. Usando CPU para treinamento.")

    logging.info("Carregando conjunto de treino CIFAR10")
    train_set = datasets.CIFAR10('./datasets', train=True, transform=transform, download=False)

    logging.info("Carregando conjunto de teste CIFAR10")
    test_set = datasets.CIFAR10('./datasets', train=False, transform=transform, download=False)

    if args['debug']:
        logging.info("Iniciando modelo no modo de depuração com 1000 entradas para treino e 200 para teste")
        train_indices = random.sample(range(len(train_set)), 5)
        test_indices = random.sample(range(len(test_set)), 1)
        train_set = Subset(train_set, train_indices)
        test_set = Subset(test_set, test_indices)

    train_loader = DataLoader(train_set, batch_size=args['tamanho_lote'], shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=args['tamanho_lote'], shuffle=False, num_workers=8)

    logging.info(f"Tamanho do lote: {args['tamanho_lote']}")
    logging.info(f"Número de classes: {args['qtd_classes']}")

    criterio = nn.CrossEntropyLoss().to(args['dispositivo'])

    return Modelo(
        train_loader=train_loader,
        test_loader=test_loader,
        args=args,
        criterio=criterio
    )
