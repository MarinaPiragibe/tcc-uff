from datetime import datetime
import logging
from pathlib import Path
from codecarbon import EmissionsTracker
import numpy as np
import torch
from torchwnn.encoding import DistributiveThermometer
import wisardpkg as wp
import gc

from utils.arquivo_utils import ArquivoUtils
from utils.dataset_utils import DatasetUtils
from utils.enums.datasets_name_enum import DatasetName
from utils.enums.tipos_transformacao_wisard import TiposDeTransformacao
from utils.logger import Logger
from wisard.wisard_model import WisardModel

# Configuração global
args = {
    "num_exec": 1,
    "tamanho_lote": 1, 
    "dataset": DatasetName.CIFAR10,
    "download_dataset": False,
    "tamanhos_tuplas": [6, 8, 12, 20, 32, 64],
    "num_bits_termometro": 8,
    "debug": False,
    "pasta_features": Path("features"),
    "tipo_transformacao": TiposDeTransformacao.ESCALA_DE_CINZA,
    "carregar_dados_salvos": True
}
args["data_execucao"] = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

def make_chunks(X, y, n):
    """Gera chunks (views) sem copiar os dados."""
    # Validação de tamanho
    if len(X) != len(y):
        raise ValueError(f"X ({len(X)}) e y ({len(y)}) têm comprimentos diferentes!")
    
    # Usa yield para ser um gerador se possível, mas como WisardModel itera várias vezes,
    # retornamos lista de views. NumPy slicing cria views, não cópias.
    return [(X[i:i+n], y[i:i+n]) for i in range(0, len(X), n)]

def extrair_dados_otimizado(dataset):
    """Carrega dados com footprint de memória controlado."""
    try:
        n_samples = len(dataset)
    except:
        dataset = list(dataset)
        n_samples = len(dataset)
    
    # Pega shape
    sample_x, _ = dataset[0]
    input_dim = sample_x.flatten().shape[0]
    
    print(f"Alocando memória para {n_samples} amostras de dimensão {input_dim}...")
    # float32 é CRÍTICO para economizar 50% de RAM vs float64
    dados = np.zeros((n_samples, input_dim), dtype=np.float32)
    classes = []
    
    # Preenchimento
    print("Preenchendo array numpy...")
    for i, (x, y) in enumerate(dataset):
        if i % 5000 == 0:
            print(f"Carregando: {i}/{n_samples}")
        
        # Flatten direto
        dados[i] = x.flatten().numpy() if hasattr(x, 'numpy') else x.flatten()
        classes.append(str(y))
    
    # Limpa o dataset original da memória se não for mais usado
    del dataset
    gc.collect()
            
    return dados, np.array(classes)

def executar_wisard(tecnica_ext_feat):
    logging.info(f"Inicializando modelo {args['modelo_base']}")
    # Limita threads para evitar concorrência excessiva no DataLoader/Torch
    torch.set_num_threads(2) 

    if args["carregar_dados_salvos"]:
        dados_treino, classes_treino, dados_teste, classes_teste = ArquivoUtils.carregar_caracteristicas_salvas(arquivo)
    else:
        dataset_treino, dataset_teste = DatasetUtils.carregar_dados_treinamento(
            args["dataset"],
            args["tipo_transformacao"],
            args["download_dataset"]
        )

        dados_treino, classes_treino = extrair_dados_otimizado(dataset_treino)
        dados_teste,  classes_teste  = extrair_dados_otimizado(dataset_teste)

    # Criação dos lotes
    dados_treino_lote = make_chunks(dados_treino, classes_treino, args['tamanho_lote'])
    dados_teste_lote = make_chunks(dados_teste,  classes_teste,  args['tamanho_lote'])  

    logging.info(f"Configurando termômetro distributivo...")
    
    # Ajusta termômetro (Fit)
    # Importante: converter para tensor explicitamente para evitar warnings
    amostra_fit = torch.from_numpy(dados_treino_lote[0][0]) if not torch.is_tensor(dados_treino_lote[0][0]) else dados_treino_lote[0][0]
    
    termometro = DistributiveThermometer(args['num_bits_termometro'])
    termometro.fit(amostra_fit)
    
    del amostra_fit # Limpa aux

    for tamanho in args['tamanhos_tuplas']:
        for i in range(args['num_exec']):
            try:
                tracker = EmissionsTracker(
                        project_name=f"{args['modelo_base']}_tupla{tamanho}",
                        output_dir="results/code_carbon",
                        output_file=f"{args['modelo_base']}_emissions.csv",
                        log_level="error",
                        save_to_file=True
                    )
                
                logging.info(f"[EXECUCAO {i+1}] [TUPLA {tamanho}] Iniciando Wisard...")
                
                # Inicializa modelo C++
                modelo_wisard = wp.Wisard(tamanho)
                
                modelo = WisardModel(
                    modelo=modelo_wisard,
                    tamanho_tupla=tamanho,
                    dados_treino=dados_treino_lote, # Passa as views
                    dados_teste=dados_teste_lote,   # Passa as views
                    classes_treino=classes_treino,
                    classes_teste=classes_teste,
                    termometro=termometro,
                    args=args,
                )

                tracker.start()
                modelo.executar_modelo(num_execucao=i+1, tecnica_ext_feat=tecnica_ext_feat)
                tracker.stop()
                
                # Força limpeza entre execuções de tuplas diferentes
                del modelo_wisard
                del modelo
                gc.collect()
                
            except Exception as e:
                logging.error(f"Erro crítico na execução {i+1} tupla {tamanho}: {e}")
                if 'tracker' in locals(): tracker.stop()

# Bloco principal de execução
if __name__ == "__main__":
    if args["carregar_dados_salvos"]:
        if not args['pasta_features'].exists():
            logging.error(f"Pasta {args['pasta_features']} não encontrada.")
        else:
            for arquivo in args['pasta_features'].iterdir():
                tecnica_ext_feat = arquivo.name.split("_")[0]
                args['modelo_base'] = f"wisard_{tecnica_ext_feat}"
                Logger.configurar_logger(
                    nome_arquivo=f"{args['modelo_base']}_{args['data_execucao']}.log"
                )
                executar_wisard(tecnica_ext_feat)
    else:
        tecnica_ext_feat = args["tipo_transformacao"].value if hasattr(args["tipo_transformacao"], 'value') else str(args["tipo_transformacao"])
        args['modelo_base'] = f"wisard_{tecnica_ext_feat}"
        Logger.configurar_logger(
                nome_arquivo=f"{args['modelo_base']}_{args['data_execucao']}.log"
            )
        executar_wisard(tecnica_ext_feat)