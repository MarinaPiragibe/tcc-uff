import torch
from torch import nn, optim

# Carregamento de Dados e Modelos
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision import transforms

# Plots e análises
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import time, os

from cnn.modelo import Modelo
from utils.imagem_utils import ImagemUtils
from utils.metricas import Metricas

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

args = {
    'num_epocas': 1,     
    'taxa_aprendizado': 1e-3,           
    'penalidade': 8e-4, 
    'tamanho_lote': 20,
    'qtd_classes': 10     
}

if torch.cuda.is_available():
    args['dispositivo'] = torch.device('cuda')
    logging.info("CUDA disponível. Usando GPU para treinamento.")
else:
    args['dispositivo'] = torch.device('cpu')
    logging.info("CUDA não disponível. Usando CPU para treinamento.")

logging.info("Definindo transformações para imagens (ImageNet)")
transform = ImagemUtils.opcoes_transformacao_imagenet()

logging.info("Carregando conjunto de treino CIFAR10")
train_set = datasets.CIFAR10('.',
                      train=True,
                      transform= transform, # transformação composta
                      download=True)

logging.info("Carregando conjunto de teste CIFAR10")
test_set = datasets.CIFAR10('.',
                      train=False,
                      transform= transform, # transformação composta
                      download=False)

train_loader = DataLoader(train_set,
                          batch_size=args['tamanho_lote'],
                          shuffle=True)
test_loader = DataLoader(test_set,
                          batch_size=args['tamanho_lote'],
                          shuffle=True)

logging.info(f"Tamanho do lote: {args['tamanho_lote']}")
logging.info(f"Número de classes: {args['qtd_classes']}")

criterio = nn.CrossEntropyLoss().to(args['dispositivo'])

logging.info("Inicializando modelo CNN (VGG16)")
modelo = Modelo(
    train_loader=train_loader,
    test_loader=test_loader,
    args=args,
    criterio=criterio
)

logging.info("Iniciando treinamento do modelo CNN")
modelo.treinar()
logging.info("Treinamento concluído!")

logging.info("Iniciando teste do modelo CNN")
classes_preditas, classes_reais = modelo.testar()
logging.info("Teste concluído!")

metricas = Metricas(classes_reais=classes_reais, classes_preditas=classes_preditas)
logging.info("Calculando métricas de desempenho")
metricas.calcular_e_imprimir_metricas()