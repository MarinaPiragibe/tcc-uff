

import torchvision
from utils.enums.datasets_name_enum import DatasetName
from wisard.wisard_image_transform import WisardImageTransform


class DatasetUtils():
    @staticmethod
    def carregar_dados_treinamento(nome_dataset, tipo_transformacao, download):
        dados_treino = None
        dados_teste = None

        match(nome_dataset):
            case DatasetName.CIFAR10:
                dados_treino = torchvision.datasets.CIFAR10('./datasets', train=True, download=download, transform=WisardImageTransform.get_image_transformation(tipo_transformacao))
                dados_teste = torchvision.datasets.CIFAR10('./datasets', train=False, download=download, transform=WisardImageTransform.get_image_transformation(tipo_transformacao))

            case DatasetName.MNIST:
                dados_treino = torchvision.datasets.MNIST('./datasets', train=True, download=download, transform=WisardImageTransform.get_image_transformation(tipo_transformacao))
                dados_teste = torchvision.datasets.MNIST('./datasets', train=False, download=download, transform=WisardImageTransform.get_image_transformation(tipo_transformacao))

            case DatasetName.FASHION_MNIST:
                dados_treino = torchvision.datasets.FashionMNIST('./datasets', train=True, download=download, transform=WisardImageTransform.get_image_transformation(tipo_transformacao))
                dados_teste = torchvision.datasets.FashionMNIST('./datasets', train=False, download=download, transform=WisardImageTransform.get_image_transformation(tipo_transformacao))

        return dados_treino, dados_teste