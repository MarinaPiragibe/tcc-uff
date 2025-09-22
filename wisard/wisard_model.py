import random


class WisardModel:
    def __init__(self, tamanho_entrada, tamanho_tupla):
        self.tamanho_entrada = tamanho_entrada
        self.tamanho_tupla = tamanho_tupla

        if self.tamanho_entrada % self.tamanho_tupla != 0:
            raise ValueError("O tamanho da entradad deve ser múltiplo do tamanho da tupla")

        self.num_rams = self.tamanho_entrada // self.tamanho_tupla

        self.descriminadores = [{}] 


    def buscar_enderecos_de_ativacao(self, entrada):
        indices = list(range(self.tamanho_entrada))
        random.shuffle(indices)

        enderecos = []
        for i in range(0, self.tamanho_entrada, self.tamanho_tupla):
            tupla_indices = indices[i:i + self.tamanho_tupla]
            tupla = [entrada[j] for j in tupla_indices]    
            endereco = int("".join(map(str, tupla)), 2)
            enderecos.append(endereco)
        return enderecos

    def fit(self, entrada, classe):
        if classe not in self.descriminadores:
            self.descriminadores[classe] = [set() for _ in range(self.num_rams)]
        
        enderecos = self.buscar_enderecos_de_ativacao(entrada)

        for indice_ram, endereco in enumerate(enderecos):
            self.memories[classe][indice_ram].add(endereco)

    
    def predict(self, entrada):
        enderecos = self.buscar_enderecos_de_ativacao(entrada)

        pontuacoes = {}
        for classe, rams in self.descriminadores.items():
            score = 0
            for indice_ram, endereco in enumerate(enderecos):
                if endereco in rams[indice_ram]:
                    score += 1
            pontuacoes[classe] = score

        return max(pontuacoes, key=pontuacoes.get), pontuacoes

if __name__ == "__main__":
    model = WisardModel(8, 2)
    print(model.buscar_enderecos_de_ativacao([1,1,1,0,0,1,1,0]))
