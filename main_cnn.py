from datetime import datetime

from cnn.cnn_utils import criar_modelo_cnn
from utils.imagem_utils import ImagemUtils


from utils.modelos_base_enum import ModeloBase


args = {
    'num_epocas': 5,     
    'taxa_aprendizado': 1e-3,           
    'penalidade': 8e-4, 
    'tamanho_lote': 16,
    'qtd_classes': 10,
    'debug': True     
}

for modelo_base in ModeloBase:
    transform = ImagemUtils.opcoes_transformacao_inceptionv3() if modelo_base == ModeloBase.INCEPTION_V3 else ImagemUtils.opcoes_transformacao_imagenet()
    args['modelo_base'] = modelo_base.value
    args['data_execucao'] = datetime.now()
    modelo = criar_modelo_cnn(
        args=args,
        transform=transform,
    )
    modelo.iniciar_modelo(modelo_base=modelo_base)
    modelo.executar_modelo()