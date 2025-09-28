from time import time
import numpy as np
import torch
from torchvision import models, transforms
from torch import nn, optim

class Modelo:
    def __init__(self, train_loader, test_loader, args, criterio):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        self.criterio=criterio

    def iniciar_modelo_vgg16(self):
        self.model = models.vgg16_bn(pretrained=True).to(self.args['dispositivo'])

        atributos_de_entrada = list(self.model.children())[-1][-1].in_features

        novo_classificador =list(self.model.classifier.children())[:-1]
        novo_classificador.append(nn.Linear(atributos_de_entrada, self.args['qtd_classes']))

        self.model.classifier = nn.Sequential(*novo_classificador).to(self.args['dispositivo'])        
        print(self.model)

        self.iniciar_otimizador_vgg16()

    def iniciar_modelo_resnet18(self):
        self.model = models.resnet18(pretrained=True).to(self.args['dispositivo'])
        atributos_de_entrada = self.model.fc.in_features

        novo_classificador =  nn.Linear(atributos_de_entrada, self.args['qtd_classes']).to(self.args['dispositivo'])
        self.model.fc = novo_classificador

        self.iniciar_otimizador_resnet18()


    def iniciar_otimizador_resnet18(self):
        self.optimizer = optim.Adam([
            {'params': [p for name, p in self.model.named_parameters() if 'fc' not in name], 'lr':self.args['taxa_aprendizado']*0.2, 'weight_decay': self.args['penalidade']*0.2},
            {'params': self.model.fc.parameters(), 'lr': self.args['taxa_aprendizado'], 'weight_decay': self.args['penalidade']}
        ], lr=0)

    def iniciar_otimizador_vgg16(self):
        self.optimizer = optim.Adam([
            {'params': self.model.features.parameters(), 'lr':self.args['taxa_aprendizado']*0.2, 'weight_decay': self.args['penalidade']*0.2},
            {'params': self.model.classifier.parameters(), 'lr': self.args['taxa_aprendizado'], 'weight_decay': self.args['penalidade']}
        ], lr=0)

    def forward(self, lote, classes_preditas, classes_reais, erro_da_epoca):
        dado, classe = lote

        dado = dado.to(self.args['dispositivo'])
        classe = classe.to(self.args['dispositivo'])

        # Forward
        classe_predita = self.model(dado)
        erro = self.criterio(classe_predita, classe)
        erro_da_epoca.append(erro.cpu().data)

        _, pred = torch.max(classe_predita, dim=1)
        classes_preditas.extend(pred.cpu().numpy())
        classes_reais.extend(classe.cpu().numpy())

        return erro
    
    def backpropagation(self, erro):
        self.optimizer.zero_grad()
        erro.backward()
        self.optimizer.step()
        
    def treinar(self):
        self.model.train()

        start = time()

        erro_da_epoca = []
        classes_preditas = []
        classes_reais = []

        for k, lote in enumerate(self.train_loader):
            print(f'\r{k+1}/{len(self.train_loader)}', end='', flush=True) 
            erro = self.forward(lote, classes_preditas, classes_reais, erro_da_epoca)            
            self.backpropagation(erro)

        print()
        end = time.time()
        return


    def testar(self):
        self.model.eval() 

        start = time.time()

        erro_da_epoca = []
        classes_preditas = []
        classes_reais = []

        with torch.no_grad(): 
            for k, lote in enumerate(self.test_loader):
                print(f'\r{k+1}/{len(self.test_loader)}', end='', flush=True) 
                erro = self.forward(lote, classes_preditas, classes_reais, erro_da_epoca)            
            print()

        end = time.time()

        erro_da_epoca_array = np.asarray(erro_da_epoca)
        classes_reais_array = np.asarray(classes_reais)
        classes_preditas_array = np.asarray(classes_preditas)
        
        return classes_reais_array, classes_preditas_array 