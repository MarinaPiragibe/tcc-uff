from ucimlrepo import fetch_ucirepo 

class Iris:
    nome = "Iris"
    id = 53
    atributos_categoricos = []
    atributos_numericos = ['sepal length', 'sepal width', 'petal length', 'petal width']

    def carregar_dataset(self):
        iris = fetch_ucirepo(id=self.id) 
        self.atributos = iris.data.features 
        self.classes = iris.data.targets   
             
    def __init__(self):
        self.carregar_dataset()         
