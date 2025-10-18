import torch
import torchvision
from utils.tipos_transformacao_wisard import TiposDeTransformacao


class WisardImageTransform:
    @staticmethod
    def get_image_transformation(transformation_type: TiposDeTransformacao):
        match (transformation_type):

            case(TiposDeTransformacao.ESCALA_DE_CINZA):
                return torchvision.transforms.Compose([
                    torchvision.transforms.Grayscale(num_output_channels=1),  
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Lambda(lambda x: torch.flatten(x))                      
                ])
            
            case(TiposDeTransformacao.BASICA):
                return torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Lambda(lambda x: torch.flatten(x))
                ])