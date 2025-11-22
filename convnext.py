import torch
from torch import nn, optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset
import time
import random
import numpy as np
import logging

# --- Configurações ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

args = {
    'num_epocas': 2,
    'taxa_aprendizado': 1e-3,
    'tamanho_lote': 128,
    'qtd_classes': 10,
    'debug': True, # Modo de depuração: Usa apenas algumas amostras.
    'dispositivo': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

# --- 1. Funções de Setup do Modelo ---

def iniciar_modelo_convnext(dispositivo, qtd_classes):
    """Inicializa o ConvNeXt Tiny pré-treinado e ajusta o classificador."""
    
    # 1. Carrega o modelo
    model = models.convnext_tiny(pretrained=True).to(dispositivo)
    
    # 2. Ajusta a camada final (classifier[2] é a camada Linear)
    try:
        atributos_de_entrada = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(atributos_de_entrada, qtd_classes).to(dispositivo)
    except Exception as e:
        logging.error(f"Falha ao ajustar o classificador do ConvNeXt: {e}")
        raise

    logging.info(f"Modelo ConvNeXt Tiny carregado e ajustado para {qtd_classes} classes.")
    return model

def iniciar_otimizador_simples(model, taxa_aprendizado):
    """Cria um otimizador simples para todos os parâmetros (sem fine-tuning complexo)."""
    
    # Usamos o método parameters() simples para evitar qualquer erro de grupo de parâmetros
    optimizer = optim.Adam(
        model.parameters(), 
        lr=taxa_aprendizado
    )
    logging.info("Otimizador Adam simples inicializado.")
    return optimizer

# --- 2. Funções de Data Loading ---

def carregar_dados(args):
    """Carrega e prepara os datasets CIFAR-10."""
    
    # Usamos as transformações padrão do ImageNet, pois o ConvNeXt foi pré-treinado no ImageNet
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = datasets.CIFAR10('./datasets', train=True, transform=transform, download=True)
    test_set = datasets.CIFAR10('./datasets', train=False, transform=transform, download=True)

    if args['debug']:
        logging.warning("MODO DEBUG ATIVO: Usando APENAS 1000 amostras de treino e 200 de teste.")
        # Usando um número maior para teste de performance, mas ainda reduzido
        train_indices = random.sample(range(len(train_set)), 1000)
        test_indices = random.sample(range(len(test_set)), 200)
        train_set = Subset(train_set, train_indices)
        test_set = Subset(test_set, test_indices)

    train_loader = DataLoader(train_set, batch_size=args['tamanho_lote'], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args['tamanho_lote'], shuffle=False)
    
    logging.info(f"Dados carregados. Treino: {len(train_set)} amostras, Teste: {len(test_set)} amostras.")
    return train_loader, test_loader

# --- 3. Funções de Treinamento e Teste ---

def treinar(model, optimizer, criterio, train_loader, dispositivo):
    """Executa um ciclo de treinamento."""
    start_time = time.time()
    model.train()
    total_loss = 0
    
    for k, (dado, classe) in enumerate(train_loader):
        # GARANTE o formato NCLH padrão, removendo qualquer otimização 'channels_last'
        dado = dado.to(dispositivo) 
        classe = classe.to(dispositivo)
        
        optimizer.zero_grad()
        
        saida_modelo = model(dado)
        
        # ConvNeXt deve retornar um Tensor simples. Não precisa de lógica AuxLogits
        erro = criterio(saida_modelo, classe)
        
        erro.backward()
        optimizer.step()
        
        total_loss += erro.item()
        print(f'\r[Treino] Batch {k+1}/{len(train_loader)}', end='', flush=True)

    end_time = time.time()
    avg_loss = total_loss / len(train_loader)
    return avg_loss, end_time - start_time

def testar(model, criterio, test_loader, dispositivo):
    """Executa o teste."""
    start_time = time.time()
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for k, (dado, classe) in enumerate(test_loader):
            dado = dado.to(dispositivo)
            classe = classe.to(dispositivo)

            saida_modelo = model(dado)
            erro = criterio(saida_modelo, classe)
            total_loss += erro.item()
            
            print(f'\r[Teste] Batch {k+1}/{len(test_loader)}', end='', flush=True)

    end_time = time.time()
    avg_loss = total_loss / len(test_loader)
    return avg_loss, end_time - start_time

# --- 4. Função Principal ---

def main():
    logging.info(f"Iniciando ConvNeXt no dispositivo: {args['dispositivo']}")
    logging.info("---")
    
    # 1. Carregar Dados
    train_loader, test_loader = carregar_dados(args)
    
    # 2. Setup do Modelo e Otimizador
    model = iniciar_modelo_convnext(args['dispositivo'], args['qtd_classes'])
    optimizer = iniciar_otimizador_simples(model, args['taxa_aprendizado'])
    criterio = nn.CrossEntropyLoss().to(args['dispositivo'])
    
    tempo_total_treino = 0
    
    # 3. Loop de Treinamento
    for epoca in range(args['num_epocas']):
        print()
        logging.info(f"Iniciando Época {epoca + 1}/{args['num_epocas']}")
        
        loss, tempo = treinar(model, optimizer, criterio, train_loader, args['dispositivo'])
        tempo_total_treino += tempo
        
        logging.info(f"\nÉpoca {epoca + 1} concluída. Erro: {loss:.4f}. Tempo: {tempo:.2f}s")

    logging.info("---")
    logging.info(f"Treinamento total concluído. Tempo total: {tempo_total_treino:.2f}s")
    
    # 4. Teste
    logging.info("Iniciando Teste Final")
    test_loss, tempo_teste = testar(model, criterio, test_loader, args['dispositivo'])
    
    logging.info(f"\nTeste concluído. Erro de Teste: {test_loss:.4f}. Tempo de Teste: {tempo_teste:.2f}s")

    print("\n\n--- RESULTADO FINAL ---")
    print(f"Tempo Total de Treino ({args['num_epocas']} épocas): {tempo_total_treino:.2f} segundos")
    print(f"Tempo de Teste: {tempo_teste:.2f} segundos")

if __name__ == "__main__":
    main()