import logging
import os
from time import time
import h5py
import numpy as np
import torch
from torchvision import models
from torch import nn, optim

from utils.enums.modelos_base_enum import ModeloBase
from utils.metricas import Metricas
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

	
	def get_feature_extractor(self, modelo_base):
		match(modelo_base):
			case ModeloBase.RESNET18:
				return torch.nn.Sequential(*list(self.model.children())[:-1])
			case ModeloBase.RESNET50:
				return torch.nn.Sequential(*list(self.model.children())[:-1])
			case ModeloBase.VGG16:
				return torch.nn.Sequential(self.model.features, self.model.avgpool)
			case ModeloBase.CONVNEXT:
				return torch.nn.Sequential(self.model.features, self.model.avgpool)
			case ModeloBase.INCEPTION_V3:
				feature_extractor = self.model
				feature_extractor.fc = torch.nn.Identity()  # Remove a última camada
				if hasattr(feature_extractor, "AuxLogits"):
					feature_extractor.AuxLogits = None  # Remove o auxiliar, pra evitar erro
				return feature_extractor
			case _:
				raise ValueError(f"Modelo base '{modelo_base}' não suportado.")
			
	def extrair_features_npz(self, modelo_base, loader):
		feature_extractor = self.get_feature_extractor(modelo_base)
		self.model.eval()
		feature_extractor.eval() 

		all_features = []
		all_labels = []

		with torch.no_grad():
			for k, (dado, rotulo) in enumerate(loader):
				print(f'\r{k+1}/{len(loader)}', end='', flush=True)

				dado = dado.to(self.args['dispositivo'])
				rotulo = rotulo.to(self.args['dispositivo'])

				# Extrai as features e achata
				features = feature_extractor(dado)
				features = features.view(features.size(0), -1).detach().cpu().numpy()
				rotulo = rotulo.cpu().numpy()

				all_features.append(features)
				all_labels.append(rotulo)

		# Concatena todos os batches
		dados = np.concatenate(all_features, axis=0)
		classes = np.concatenate(all_labels, axis=0)

		return dados, classes


	def extrair_features(self, modelo_base, loader):
		feature_extractor = self.get_feature_extractor(modelo_base)
		with h5py.File('features_labels.h5', 'w') as h5f:
			first = True
			self.model.eval()
			with torch.no_grad():
				for k, (dado, rotulo) in enumerate(loader):
					print(f'\r{k+1}/{len(loader)}', end='', flush=True)

					dado = dado.to(self.args['dispositivo'])
					rotulo = rotulo.to(self.args['dispositivo'])

					features = feature_extractor(dado)
					features = features.view(features.size(0), -1).detach().cpu().numpy()
					rotulo = rotulo.cpu().numpy()

					if first:
						features_dset = h5f.create_dataset('features', data=features, maxshape=(None, features.shape[1]))
						labels_dset = h5f.create_dataset('labels', data=rotulo, maxshape=(None,))
						first = False
					else:
						features_dset.resize(features_dset.shape[0] + features.shape[0], axis=0)
						features_dset[-features.shape[0]:] = features

						labels_dset.resize(labels_dset.shape[0] + rotulo.shape[0], axis=0)
						labels_dset[-rotulo.shape[0]:] = rotulo

				print("\nDados salvos em 'features_labels.h5'")
	
	def extrair_e_salvar_features_cnn(self, modelo_base):

		logging.info(f"[ÍNICIO] Extração de características do treino")
		
		start_treino = time()
		dados_treino, classes_treino = self.extrair_features_npz(modelo_base, self.train_loader)
		end_treino = time()

		tempo_total_treino = end_treino - start_treino

		logging.info(f"[FIM] Extração de características do treino demorou {tempo_total_treino}")
		
		logging.info(f"[ÍNICIO] Extração de características do teste")

		start_teste = time()
		dados_teste, classes_teste = self.extrair_features_npz( modelo_base, self.test_loader,)
		end_teste = time()

		tempo_total_teste = end_teste - start_teste
		
		logging.info(f"[FIM] Extração de características do teste demorou {tempo_total_teste}")

		# if modelo_base == ModeloBase.VGG16:
		# 	ArquivoUtils.salvar_features_vgg16_h5(
		# 		nome_dataset=self.args['dataset'],
		# 		dados_treino=dados_treino,
		# 		classes_treino=classes_treino,
		# 		dados_teste=dados_teste,
		# 		classes_teste=classes_teste
		# 	)

		# else:
		ArquivoUtils.salvar_features_imagem(
			nome_tecnica_ext=modelo_base.value,
			nome_dataset=self.args['dataset'],
			dados_treino=dados_treino,  
			classes_treino=classes_treino,
			dados_teste=dados_teste,  
			classes_teste=classes_teste
		)

		dados_execucao = {
			"nome_tecnica": modelo_base.value,
			"shape_treino": dados_treino.shape,
			"shape_teste": dados_teste.shape,
			"tempo_treino": tempo_total_treino,
			"tempo_teste": tempo_total_teste,
			"tempo_total": tempo_total_treino + tempo_total_teste
		}

		ArquivoUtils.salvar_csv(self.args, dados_execucao, self.args['arq_ext_caract'])

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
	
	def executar_modelo(self, num_execucao):
		lista_erro_treino = []
		lista_tempo_execucao_treino = []

		for epoca in range(self.args['num_epocas']):
			logging.info(f"[EXECUÇÃO {num_execucao}][ÉPOCA {epoca+1}] Iniciando o treinamento do modelo")

			erro_do_treino, tempo_execucao_treino = self.treinar(epoca)
			lista_erro_treino.append(erro_do_treino)
			lista_tempo_execucao_treino.append(tempo_execucao_treino)

			logging.info(f"[EXECUÇÃO {num_execucao}][ÉPOCA {epoca+1}] Treinamento concluído em {tempo_execucao_treino}")
			logging.info(f"[EXECUÇÃO {num_execucao}][ÉPOCA {epoca+1}] Erro do treino: {erro_do_treino}")

		logging.info(f"[EXECUÇÃO {num_execucao}] Iniciando teste do modelo após todas as épocas")

		classes_reais, classes_preditas, erro_do_teste, tempo_execucao_teste = self.testar(num_execucao=num_execucao)

		logging.info(f"[EXECUÇÃO {num_execucao}] Teste concluído em {tempo_execucao_teste}")
		logging.info(f"[EXECUÇÃO {num_execucao}] Erro do teste: {erro_do_teste}")

		metricas = Metricas(classes_reais=classes_reais, classes_preditas=classes_preditas)
		logging.info(f"[EXECUÇÃO {num_execucao}] Execução concluída em {sum(lista_tempo_execucao_treino) + tempo_execucao_teste}")
		logging.info(f"[EXECUÇÃO {num_execucao}] Calculando métricas de desempenho")
		metricas.calcular_e_imprimir_metricas()

		dados_execucao = {
			"execucao": num_execucao,
			"modelo_base": self.args['modelo_base'],
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

		ArquivoUtils.salvar_csv(self.args, dados_execucao)

		ArquivoUtils.salvar_modelo(
			args=self.args,
			estado_modelo=self.model.state_dict(),
			estado_otimizador=self.optimizer.state_dict(),
		)

		logging.info(f"Execução do modelo concluída. Tempo total: {sum(lista_tempo_execucao_treino) + tempo_execucao_teste}")
		return


	def testar(self, num_execucao=None):
		logging.info(f"[EXECUÇÃO {num_execucao}] Iniciando teste do modelo CNN")
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