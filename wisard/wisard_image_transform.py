import torch
import torchvision
from utils.tipos_transformacao_wisard import TiposDeTransformacao


class WisardImageTransform:
    @staticmethod
    def get_image_transformation(transformation_type: TiposDeTransformacao):
        match (transformation_type):
            case(TiposDeTransformacao.THRESHOLD_3):
                transforms = lambda x: torch.cat(
                    [(x > (i + 1) / 4).int() for i in range(3)], dim=0
                )
                return torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Lambda(transforms),
                    torchvision.transforms.Lambda(lambda x: torch.flatten(x))
                ])
            
            case(TiposDeTransformacao.THRESHOLD_31):
                transforms = lambda x: torch.cat([(x > (i + 1) / 32).int() for i in range(31)], dim=0)
                return torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Lambda(transforms),
                    torchvision.transforms.Lambda(lambda x: torch.flatten(x))
                ])

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