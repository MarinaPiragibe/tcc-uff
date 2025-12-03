import numpy as np
import gc
import logging
from sklearn.decomposition import IncrementalPCA

# Configuração
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# --- PARÂMETROS ---
NOME_ARQUIVO_ENTRADA = 'features/vgg16_features.npz'
NOME_ARQUIVO_SAIDA = 'dados_vgg16_pca_reduzido.npz'
N_COMPONENTES = 512  # Alvo: Reduzir de 25088 para 1024 features
TAMANHO_LOTE_IPCA = 512  # Tamanho do lote para treinamento e transformação (ex: 1000 amostras)

# =========================================================================

def carregar_dados_npz(caminho_arquivo):
    """Carrega dados do arquivo NPZ."""
    logging.info(f"Carregando dados do arquivo: {caminho_arquivo}")
    try:
        dados = np.load(caminho_arquivo)
        return {k: dados[k] for k in dados.files}
    except FileNotFoundError:
        logging.error(f"Arquivo não encontrado: {caminho_arquivo}")
        return None

def aplicar_ipca(dados, n_componentes, batch_size):
    """Aplica o PCA Incremental (IPCA) nos dados de treino e transforma."""
    
    dados_treino = dados['dados_treino']
    dados_teste = dados['dados_teste']
    
    logging.info(f"Dimensões originais: Treino {dados_treino.shape}, Teste {dados_teste.shape}")
    
    # 1. Inicializa o IPCA
    ipca = IncrementalPCA(n_components=n_componentes)
    
    # 2. Treinamento (fit) com partial_fit em lotes
    logging.info("Iniciando o treinamento do Incremental PCA (partial_fit)...")
    n_amostras_treino = dados_treino.shape[0]
    
    for i in range(0, n_amostras_treino, batch_size):
        # Seleciona o lote de dados de treino
        lote_treino = dados_treino[i : i + batch_size]
        ipca.partial_fit(lote_treino)
        
        logging.info(f"  > Lote {i // batch_size + 1} de {n_amostras_treino // batch_size} treinado.")
        del lote_treino
        gc.collect() # Limpeza de memória
        
    logging.info("Treinamento do IPCA concluído.")
    logging.info(f"Variância explicada total: {np.sum(ipca.explained_variance_ratio_):.4f}")

    # 3. Transformação (transform) dos dados de treino em lotes
    logging.info("Iniciando a transformação dos dados de treino...")
    dados_treino_reduzidos = []
    
    for i in range(0, n_amostras_treino, batch_size):
        lote_treino = dados_treino[i : i + batch_size]
        lote_reduzido = ipca.transform(lote_treino)
        dados_treino_reduzidos.append(lote_reduzido)
        
        del lote_treino, lote_reduzido
        gc.collect() 
        
    dados_treino_reduzidos = np.concatenate(dados_treino_reduzidos, axis=0)

    # 4. Transformação (transform) dos dados de teste
    logging.info("Iniciando a transformação dos dados de teste...")
    # O conjunto de teste é menor (10k), geralmente pode ser transformado de uma vez, 
    # mas transformaremos em lote para consistência e segurança.
    dados_teste_reduzidos = []
    n_amostras_teste = dados_teste.shape[0]
    
    for i in range(0, n_amostras_teste, batch_size):
        lote_teste = dados_teste[i : i + batch_size]
        lote_reduzido = ipca.transform(lote_teste)
        dados_teste_reduzidos.append(lote_reduzido)
        
        del lote_teste, lote_reduzido
        gc.collect()
        
    dados_teste_reduzidos = np.concatenate(dados_teste_reduzidos, axis=0)
    
    logging.info(f"Novas dimensões: Treino {dados_treino_reduzidos.shape}, Teste {dados_teste_reduzidos.shape}")

    # 5. Retorna os novos dados, mantendo as classes originais
    return {
        'dados_treino': dados_treino_reduzidos,
        'dados_teste': dados_teste_reduzidos,
        'classes_treino': dados['classes_treino'],
        'classes_teste': dados['classes_teste'],
    }

# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    dados = carregar_dados_npz(NOME_ARQUIVO_ENTRADA)
    
    if dados:
        # Garante que as classes tenham o formato correto (N, 1) se necessário
        dados['classes_treino'] = dados['classes_treino'].reshape(-1, 1)
        dados['classes_teste'] = dados['classes_teste'].reshape(-1, 1)
        
        # Aplica o IPCA
        dados_reduzidos = aplicar_ipca(dados, N_COMPONENTES, TAMANHO_LOTE_IPCA)
        
        # Salva o novo arquivo NPZ com as features reduzidas
        np.savez_compressed(NOME_ARQUIVO_SAIDA, **dados_reduzidos)
        logging.info(f"Novos dados salvos em: {NOME_ARQUIVO_SAIDA}")

        # Limpeza final
        del dados, dados_reduzidos
        gc.collect()