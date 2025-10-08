import logging
import os
from time import time
import numpy as np
import torch
from torchvision import models
from torch import nn, optim

from utils.metricas import Metricas
from utils.modelos_base_enum import ModeloBase
from utils.arquivo_utils import ArquivoUtils


class Modelo:
	def __init__(self, train_loader, test_loader, args, criterio):
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.args = args
		self.criterio=criterio

	def iniciar_modelo(self, modelo_base):
		match modelo_base:
			case ModeloBase.VGG16:
				self.iniciar_modelo_vgg16()
			case ModeloBase.INCEPTION_V3:
				self.iniciar_modelo_inception_v3()
			case ModeloBase.RESNET18:
				self.iniciar_modelo_resnet18()
			case ModeloBase.RESNET50:
				self.iniciar_modelo_resnet50()
			case ModeloBase.CONVNEXT:
				self.iniciar_modelo_convnext()


	def iniciar_modelo_vgg16(self):
		self.model = models.vgg16_bn(pretrained=True).to(self.args['dispositivo'])

		classificador_original = self.model.classifier
		atributos_de_entrada = classificador_original[-1].in_features 

		novo_classificador = list(classificador_original.children())[:-1]
		novo_classificador.append(nn.Linear(atributos_de_entrada, self.args['qtd_classes']))

		self.model.classifier = nn.Sequential(*novo_classificador).to(self.args['dispositivo'])
		self.iniciar_otimizador()

	def iniciar_modelo_resnet50(self):
		self.model = models.resnet50(pretrained=True).to(self.args['dispositivo'])
		atributos_de_entrada = self.model.fc.in_features

		novo_classificador = nn.Linear(atributos_de_entrada, self.args['qtd_classes']).to(self.args['dispositivo'])
		self.model.fc = novo_classificador

		self.iniciar_otimizador_resnet()

	def iniciar_modelo_resnet18(self):
		self.model = models.resnet18(pretrained=True).to(self.args['dispositivo'])
		atributos_de_entrada = self.model.fc.in_features

		novo_classificador =  nn.Linear(atributos_de_entrada, self.args['qtd_classes']).to(self.args['dispositivo'])
		self.model.fc = novo_classificador

		self.iniciar_otimizador_resnet()

	def iniciar_modelo_convnext(self):
		self.model = models.convnext_tiny(pretrained=True).to(self.args['dispositivo'])
		
		atributos_de_entrada = self.model.classifier[2].in_features
		
		self.model.classifier[2] = nn.Linear(atributos_de_entrada, self.args['qtd_classes']).to(self.args['dispositivo'])

		self.iniciar_otimizador()
	
	def iniciar_modelo_inception_v3(self):
		self.model = models.inception_v3(pretrained=True).to(self.args['dispositivo'])
		
		self.model.AuxLogits.fc = nn.Linear(self.model.AuxLogits.fc.in_features, 
											self.args['qtd_classes']).to(self.args['dispositivo'])
		
		self.model.fc = nn.Linear(self.model.fc.in_features, 
								self.args['qtd_classes']).to(self.args['dispositivo'])

		self.iniciar_otimizador_inceptionv3()

	def iniciar_otimizador(self):
		self.optimizer = optim.Adam([
			{'params': self.model.features.parameters(), 'lr':self.args['taxa_aprendizado']*0.2, 'weight_decay': self.args['penalidade']*0.2},
			{'params': self.model.classifier.parameters(), 'lr': self.args['taxa_aprendizado'], 'weight_decay': self.args['penalidade']}
		], lr=0)

	def iniciar_otimizador_resnet(self):
		self.optimizer = optim.Adam([
			{'params': [p for name, p in self.model.named_parameters() if 'fc' not in name], 'lr':self.args['taxa_aprendizado']*0.2, 'weight_decay': self.args['penalidade']*0.2},
			{'params': self.model.fc.parameters(), 'lr': self.args['taxa_aprendizado'], 'weight_decay': self.args['penalidade']}
		], lr=0)
	
	def iniciar_otimizador_inceptionv3(self):
		self.optimizer = optim.Adam([
			{'params': [p for name, p in self.model.named_parameters() if 'fc' not in name and 'AuxLogits' not in name], 
			'lr':self.args['taxa_aprendizado']*0.2, 'weight_decay': self.args['penalidade']*0.2},
			
			{'params': self.model.fc.parameters(), 'lr': self.args['taxa_aprendizado'], 'weight_decay': self.args['penalidade']},
			{'params': self.model.AuxLogits.fc.parameters(), 'lr': self.args['taxa_aprendizado'], 'weight_decay': self.args['penalidade']}
		], lr=0)


	def forward(self, lote, classes_preditas, classes_reais, erro_da_epoca):
		dado, classe = lote

		dado = dado.to(self.args['dispositivo'])
		classe = classe.to(self.args['dispositivo'])

		saida_modelo = self.model(dado)
		classe_predita = None

		if isinstance(saida_modelo, torch.Tensor):
			classe_predita = saida_modelo
		else:
			classe_predita = saida_modelo[0] 

		erro = self.criterio(classe_predita, classe)

		erro_da_epoca.append(erro.item())

		_, pred = torch.max(classe_predita, dim=1)
		classes_preditas.extend(pred.cpu().numpy())
		classes_reais.extend(classe.cpu().numpy())

		return erro

	
	def backpropagation(self, erro):
		self.optimizer.zero_grad()
		erro.backward()
		self.optimizer.step()
		
	def treinar(self, epoca=None):
		logging.info(f"[ÉPOCA {epoca+1}] Iniciando treinamento do modelo CNN")
		start = time()

		self.model.train()

		erro_da_epoca = []
		classes_preditas = []
		classes_reais = []
		erro = None

		for k, lote in enumerate(self.train_loader):
			print(f'\r{k+1}/{len(self.train_loader)}', end='', flush=True) 
			erro = self.forward(lote, classes_preditas, classes_reais, erro_da_epoca)            
			self.backpropagation(erro)

		end = time()

		erro_da_epoca_array = np.asarray(erro_da_epoca)

		tempo_execucao = end-start
		return erro_da_epoca_array.mean(), tempo_execucao
	
	def executar_modelo(self):
		lista_erro_treino, lista_erro_teste = [], []
		lista_tempo_execucao_treino, lista_tempo_execucao_teste = [], []
		for epoca in range(self.args['num_epocas']):
			logging.info(f"[ÉPOCA {epoca+1}] Iniciando a execução do modelo (treino e teste)")

			erro_do_treino, tempo_execucao_treino = self.treinar(epoca)
			lista_erro_treino.append(erro_do_treino)
			lista_tempo_execucao_treino.append(tempo_execucao_treino)

			logging.info(f"[ÉPOCA {epoca+1}] Treinamento concluído em {tempo_execucao_treino}")
			logging.info(f"[ÉPOCA {epoca+1}] Erro do treino: {erro_do_treino}")

			classes_reais, classes_preditas, erro_do_teste, tempo_execucao_teste = self.testar(epoca)
			lista_erro_teste.append(erro_do_teste)
			lista_tempo_execucao_teste.append(tempo_execucao_teste)

			logging.info(f"[ÉPOCA {epoca+1}] Teste concluído em {tempo_execucao_teste}")
			logging.info(f"[ÉPOCA {epoca+1}] Erro do teste: {erro_do_teste}")

			metricas = Metricas(classes_reais=classes_reais, classes_preditas=classes_preditas)
			logging.info(f"[ÉPOCA {epoca+1}] Calculando métricas de desempenho")
			metricas.calcular_e_imprimir_metricas()
		
		tempo_total_treino = sum(tempo for tempo in lista_tempo_execucao_treino)
		tempo_total_teste = sum(tempo for tempo in lista_tempo_execucao_teste)

		logging.info(f"Execução do modelo concluída em {tempo_total_treino + tempo_total_teste}")
		logging.info(f"Tempo de execução do treino: {tempo_total_treino}")
		logging.info(f"Tempo de execução do teste: {tempo_total_teste}")

		ArquivoUtils.salvar_modelo(
			args=self.args,
    		estado_modelo=self.model.state_dict(),
    		estado_otimizador=self.optimizer.state_dict(),
		)

		return


	def testar(self, epoca=None):
		logging.info(f"[ÉPOCA {epoca+1}] Iniciando teste do modelo CNN")
		start = time()

		self.model.eval() 

		erro_da_epoca = []
		classes_preditas = []
		classes_reais = []

		with torch.no_grad(): 
			for k, lote in enumerate(self.test_loader):
				print(f'\r{k+1}/{len(self.test_loader)}', end='', flush=True) 
				erro = self.forward(lote, classes_preditas, classes_reais, erro_da_epoca)            
			print()

		erro_da_epoca_array = np.asarray(erro_da_epoca)
		classes_reais_array = np.asarray(classes_reais)
		classes_preditas_array = np.asarray(classes_preditas)
		end = time()

		tempo_execucao = end-start

		return classes_reais_array, classes_preditas_array, erro_da_epoca_array.mean(), tempo_execucao