from datetime import datetime
import gc
import logging

from codecarbon import EmissionsTracker
import torch

from cnn.cnn_utils import criar_modelo_cnn
from utils.enums.datasets_name_enum import DatasetName
from utils.enums.modelos_base_enum import ModeloBase
from utils.imagem_utils import ImagemUtils


from utils.logger import Logger


args = {
    'num_epocas': 2,
    'num_execucoes': 2,
    'extrair_caracteristicas': False,     
    'taxa_aprendizado': 1e-3,           
    'penalidade': 8e-4, 
    'tamanho_lote': 128,
    'qtd_classes': 10,
    "dataset": DatasetName.CIFAR10,
    'debug': True,
    "arq_ext_caract": "extracao_caracteristicas_resultados"     
}
args['data_execucao'] = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')


for modelo_base in ModeloBase:
    for execucao in range(args['num_execucoes']):
        try:
            args['modelo_base'] = modelo_base.value

            Logger.configurar_logger(nome_arquivo=f"cnn_application_{args['modelo_base']}_{args['data_execucao']}.log")
            
            transform = ImagemUtils.opcoes_transformacao_inceptionv3() if modelo_base == ModeloBase.INCEPTION_V3 else ImagemUtils.opcoes_transformacao_imagenet()

            # Monitoramento inicial da memória GPU
            mem_alloc_before = torch.cuda.memory_allocated()
            mem_reserved_before = torch.cuda.memory_reserved()
            logging.info(f"[INÍCIO] Memória GPU alocada: {mem_alloc_before}, reservada: {mem_reserved_before}")
            
            modelo = criar_modelo_cnn(
                args=args,
                transform=transform,
                num_execucao=execucao+1
            )

            
            modelo.iniciar_modelo(modelo_base=modelo_base)

            if args['extrair_caracteristicas']:

                tracker = EmissionsTracker(
                    project_name=f"{args['modelo_base']}",
                    output_dir="results/code_carbon",
					output_file="emissions_ext_carac.csv",
					log_level="error"
                )

                tracker.start()
                modelo.extrair_e_salvar_features_cnn(modelo_base=modelo_base)
                tracker.stop()
                break
            else:
                tracker = EmissionsTracker(
                    project_name=f"{args['modelo_base']}",
                    output_dir="results/code_carbon",
                    output_file=f"{args['modelo_base']}_emissions.csv",
                    log_level="error"
                )

                tracker.start()
                modelo.executar_modelo(num_execucao=execucao+1)
                tracker.stop()
        
        except Exception as e:
            logging.error(f"Falha na execução do modelo {modelo_base.value}: {e}")

        finally:
            if 'modelo' in locals():
                del modelo
            torch.cuda.empty_cache()
            gc.collect()

            mem_alloc_after = torch.cuda.memory_allocated()
            mem_reserved_after = torch.cuda.memory_reserved()
            logging.info(f"[FIM] Memória GPU - Alocada: {mem_alloc_after}, Reservada: {mem_reserved_after}")
            