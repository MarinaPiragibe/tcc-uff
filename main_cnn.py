from datetime import datetime
import logging

from cnn.cnn_utils import criar_modelo_cnn
from utils.imagem_utils import ImagemUtils


from utils.logger import Logger
from utils.modelos_base_enum import ModeloBase


args = {
    'num_epocas': 5,     
    'taxa_aprendizado': 1e-3,           
    'penalidade': 8e-4, 
    'tamanho_lote': 32,
    'qtd_classes': 10,
    'debug': False     
}
args['data_execucao'] = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')

for modelo_base in ModeloBase:
    try:
        args['modelo_base'] = modelo_base.value

        Logger.configurar_logger(nome_arquivo=f"cnn_application_{args['modelo_base']}_{args['data_execucao']}.log")
        
        transform = ImagemUtils.opcoes_transformacao_inceptionv3() if modelo_base == ModeloBase.INCEPTION_V3 else ImagemUtils.opcoes_transformacao_imagenet()
        modelo = criar_modelo_cnn(
            args=args,
            transform=transform,
        )
        modelo.iniciar_modelo(modelo_base=modelo_base)
        modelo.executar_modelo()
    
    except Exception as e:
        logging.error(f"Failed to run fine tuning for model {modelo_base.value}: {e}")
        