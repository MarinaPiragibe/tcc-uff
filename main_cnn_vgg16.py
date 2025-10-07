from datetime import datetime
import random
from sklearn import datasets
import torch
from torch import nn

from torch.utils.data import DataLoader, Subset

from cnn.modelo import Modelo
from utils.imagem_utils import ImagemUtils
from utils.logger import Logger

import logging

logger = logging.getLogger(__name__)

args = {
    'modelo_base': "vgg16",
    'num_epocas': 1,     
    'taxa_aprendizado': 1e-3,           
    'penalidade': 8e-4, 
    'tamanho_lote': 16,
    'qtd_classes': 10,
    'debug': True     
}

Logger.configurar_logger(nome_arquivo=f"cnn_application_{args['modelo_base']}_{datetime.now()}.log")

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
                    transform= transform,
                    download=True)

logging.info("Carregando conjunto de teste CIFAR10")
test_set = datasets.CIFAR10('.',
                    train=False,
                    transform= transform, 
                    download=False)

if args['debug']:
    logging.info("Iniciando modelo no modo de depuração com 1000 entradas para treino e 200 para teste")
    train_indices = random.sample(range(len(train_set)), 1000)
    test_indices = random.sample(range(len(test_set)), 200)

    train_set = Subset(train_set, train_indices)
    test_set = Subset(test_set, test_indices)


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
modelo.iniciar_modelo_vgg16()
modelo.executar_modelo()