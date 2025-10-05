from torchvision import transforms
import matplotlib.pyplot as plt

class ImagemUtils:
    @staticmethod
    def opcoes_transformacao_imagenet():
        return transforms.Compose([
                                     transforms.Resize(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                  ])
    
    @staticmethod
    def opcoes_transformacao_inceptionv3():
        return transforms.Compose([
            transforms.Resize(340),     
            transforms.CenterCrop(299), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def exibir_imagens(quantidade, dados):
        _, axs = plt.subplots(1,quantidade, figsize=(20, 2))
        for i in range(10):
            data, _ = dados[i]
            axs[i].imshow(data.permute((1,2,0)))
            axs[i].axis('off')