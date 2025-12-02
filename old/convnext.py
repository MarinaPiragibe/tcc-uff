import torch
from torch import nn, optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset
import time
import random
import numpy as np
import logging

from utils.arquivo_utils import ArquivoUtils
from utils.metricas import Metricas

# --- Configurações ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

args = {
	'num_epocas': 2,
	'taxa_aprendizado': 1e-3,
	'tamanho_lote': 128,
	'qtd_classes': 10,
	'debug': True,
	'dispositivo': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
	'modelo_base': 'convnext_tiny'
}

def iniciar_modelo_convnext(dispositivo, qtd_classes):
	model = models.convnext_tiny(pretrained=True).to(dispositivo)
	try:
		atributos_de_entrada = model.classifier[2].in_features
		model.classifier[2] = nn.Linear(atributos_de_entrada, qtd_classes).to(dispositivo)
	except Exception as e:
		logging.error(f"Falha ao ajustar o classificador do ConvNeXt: {e}")
		raise
	logging.info(f"Modelo ConvNeXt Tiny carregado e ajustado para {qtd_classes} classes.")
	return model

def iniciar_otimizador_simples(model, taxa_aprendizado):
	optimizer = optim.Adam(model.parameters(), lr=taxa_aprendizado)
	logging.info("Otimizador Adam simples inicializado.")
	return optimizer

def carregar_dados(args):
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	train_set = datasets.CIFAR10('./datasets', train=True, transform=transform, download=True)
	test_set = datasets.CIFAR10('./datasets', train=False, transform=transform, download=True)
	
	if args['debug']:
		logging.warning("MODO DEBUG ATIVO: Usando APENAS 1000 amostras de treino e 200 de teste.")
		train_indices = random.sample(range(len(train_set)), 1000)
		test_indices = random.sample(range(len(test_set)), 200)
		train_set = Subset(train_set, train_indices)
		test_set = Subset(test_set, test_indices)
	
	train_loader = DataLoader(train_set, batch_size=args['tamanho_lote'], shuffle=True)
	test_loader = DataLoader(test_set, batch_size=args['tamanho_lote'], shuffle=False)
	logging.info(f"Dados carregados. Treino: {len(train_set)} amostras, Teste: {len(test_set)} amostras.")
	return train_loader, test_loader

def forward(lote, classes_preditas, classes_reais, erro_da_epoca):
	dado, classe = lote
	dado = dado.to(args['dispositivo'])
	classe = classe.to(args['dispositivo'])
	saida_modelo = model(dado)
	
	if isinstance(saida_modelo, torch.Tensor):
		classe_predita = saida_modelo
	else:
		classe_predita = saida_modelo[0]
	
	erro = criterio(classe_predita, classe)
	erro_da_epoca.append(erro.item())
	_, pred = torch.max(classe_predita, dim=1)
	classes_preditas.extend(pred.cpu().numpy())
	classes_reais.extend(classe.cpu().numpy())
	return erro

def backpropagation(erro):
	optimizer.zero_grad()
	erro.backward()
	optimizer.step()

def treinar(train_loader, epoca=None):
	start = time.time()
	model.train()
	erro_da_epoca = []
	classes_preditas = []
	classes_reais = []
	
	for k, lote in enumerate(train_loader):
		print(f'\r{k+1}/{len(train_loader)}', end='', flush=True)
		erro = forward(lote, classes_preditas, classes_reais, erro_da_epoca)
		backpropagation(erro)
	
	end = time.time()
	erro_da_epoca_array = np.asarray(erro_da_epoca)
	tempo_execucao = end-start
	return erro_da_epoca_array.mean(), tempo_execucao

def testar(test_loader, num_execucao=None):
	logging.info(f"[EXECUÇÃO {num_execucao}] Iniciando teste do modelo CNN")
	start = time.time()
	model.eval()
	erro_da_epoca = []
	classes_preditas = []
	classes_reais = []
	
	with torch.no_grad():
		for k, lote in enumerate(test_loader):
			print(f'\r{k+1}/{len(test_loader)}', end='', flush=True)
			erro = forward(lote, classes_preditas, classes_reais, erro_da_epoca)
		print()
	
	erro_da_epoca_array = np.asarray(erro_da_epoca)
	classes_reais_array = np.asarray(classes_reais)
	classes_preditas_array = np.asarray(classes_preditas)
	end = time.time()
	tempo_execucao = end-start
	return classes_reais_array, classes_preditas_array, erro_da_epoca_array.mean(), tempo_execucao

def executar_modelo(num_execucao):
	global model, optimizer, criterio
	model = iniciar_modelo_convnext(args['dispositivo'], args['qtd_classes'])
	optimizer = iniciar_otimizador_simples(model, args['taxa_aprendizado'])
	criterio = nn.CrossEntropyLoss().to(args['dispositivo'])
	lista_erro_treino = []
	lista_tempo_execucao_treino = []
	train_loader, test_loader = carregar_dados(args)
	
	for epoca in range(args['num_epocas']):
		logging.info(f"[EXECUÇÃO {num_execucao}][ÉPOCA {epoca+1}] Iniciando o treinamento do modelo")
		erro_do_treino, tempo_execucao_treino = treinar(train_loader, epoca)
		lista_erro_treino.append(erro_do_treino)
		lista_tempo_execucao_treino.append(tempo_execucao_treino)
		logging.info(f"[EXECUÇÃO {num_execucao}][ÉPOCA {epoca+1}] Treinamento concluído em {tempo_execucao_treino}")
		logging.info(f"[EXECUÇÃO {num_execucao}][ÉPOCA {epoca+1}] Erro do treino: {erro_do_treino}")
	
	logging.info(f"[EXECUÇÃO {num_execucao}] Iniciando teste do modelo após todas as épocas")
	
	classes_reais, classes_preditas, erro_do_teste, tempo_execucao_teste = testar(test_loader, num_execucao=num_execucao)
	logging.info(f"[EXECUÇÃO {num_execucao}] Teste concluído em {tempo_execucao_teste}")
	logging.info(f"[EXECUÇÃO {num_execucao}] Erro do teste: {erro_do_teste}")
	
	metricas = Metricas(classes_reais=classes_reais, classes_preditas=classes_preditas)
	logging.info(f"[EXECUÇÃO {num_execucao}] Execução concluída em {sum(lista_tempo_execucao_treino) + tempo_execucao_teste}")
	logging.info(f"[EXECUÇÃO {num_execucao}] Calculando métricas de desempenho")
	metricas.calcular_e_imprimir_metricas()
	
	dados_execucao = {
		"execucao": num_execucao,
		"modelo_base": args['modelo_base'],
		"acuracia": metricas.acc,
		"precisao": metricas.precisao,
		"recall": metricas.recall,
		"f1": metricas.f1,
		"erro_treino": lista_erro_treino[-1],
		"media_erro_treino": np.mean(lista_erro_treino),
		"erro_teste": erro_do_teste,
		"tempo_treino": sum(lista_tempo_execucao_treino),
		"tempo_teste": tempo_execucao_teste,
		"tempo_total": sum(lista_tempo_execucao_treino) + tempo_execucao_teste
	}
	ArquivoUtils.salvar_csv(args, dados_execucao)
	
	ArquivoUtils.salvar_modelo(
		args=args,
		estado_modelo=model.state_dict(),
		estado_otimizador=optimizer.state_dict()
	)
	logging.info(f"Execução do modelo concluída. Tempo total: {sum(lista_tempo_execucao_treino) + tempo_execucao_teste}")
	return

if __name__ == "__main__":
	executar_modelo(1)
