import logging
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

class Metricas:

    def __init__(self, classes_reais, classes_preditas):
        self.classes_reais = classes_reais
        self.classes_preditas = classes_preditas

    def imprimir_resultados(self):
        logging.info(f'\n{"-" * 10} Métricas {"-" * 10}')
        logging.info(f"Accuracy: {self.acc}")
        logging.info(f"Precisão: {self.precisao}")
        logging.info(f"Recall: {self.recall}")
        logging.info(f"F1: {self.f1}")
        logging.info(f"Matriz de confusão:\n{self.matriz}")

    def calcular_e_imprimir_metricas(self):
        
        self.acc = accuracy_score(self.classes_reais, self.classes_preditas)
        self.precisao = precision_score(self.classes_reais, self.classes_preditas, average="macro", zero_division=0)
        self.recall = recall_score(self.classes_reais, self.classes_preditas, average="macro", zero_division=0)
        self.f1 = f1_score(self.classes_reais, self.classes_preditas, average="macro", zero_division=0)
        self.matriz = confusion_matrix(self.classes_reais, self.classes_preditas)
        self.imprimir_resultados()