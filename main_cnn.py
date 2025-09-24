# Implementação e treinamento da rede
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

args = {
    'num_epocas': 1,     
    'taxa_aprendizado': 1e-3,           
    'penalidade': 8e-4, 
    'tamanho_lote': 20,
    'qtd_classe': 10     
}

if torch.cuda.is_available():
    args['dispositivo'] = torch.device('cuda')
else:
    args['dispositivo'] = torch.device('cpu')

transform = ImagemUtils.opcoes_transformacao_imagenet()

train_set = datasets.CIFAR10('.',
                      train=True,
                      transform= transform, # transformação composta
                      download=True)

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

criterio = nn.CrossEntropyLoss().to(args['device'])

modelo = Modelo(
    train_loader=train_loader,
    test_loader=test_loader,
    args=args,
    criterio=criterio
)

modelo.treinar()
classes_reais, classes_preditas = modelo.testar()

metricas = Metricas(classes_reais=classes_reais, classes_preditas=classes_preditas)
metricas.calcular_e_imprimir_metricas()