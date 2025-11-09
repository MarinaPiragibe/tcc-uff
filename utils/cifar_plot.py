import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import Image

# Transformação básica
transform = transforms.ToTensor()

# Carregar dataset (assumindo que já está baixado)
dataset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=False, transform=transform)

# Classes do CIFAR-10
class_names = dataset.classes

# Selecionar uma imagem por classe
images = []
labels = []
seen_classes = set()

for img, label in dataset:
    if label not in seen_classes:
        images.append(img)
        labels.append(label)
        seen_classes.add(label)
    if len(seen_classes) == 10:
        break

# Salvar cada imagem individualmente
for img, label in zip(images, labels):
    # Converter tensor para PIL Image
    img_pil = transforms.ToPILImage()(img)
    filename = f'cifar10_{class_names[label]}.png'
    img_pil.save(filename)
    print(f'Imagem salva: {filename}')

# Plotar grid (opcional)
fig, axes = plt.subplots(2, 5, figsize=(6.4, 3.2))
axes = axes.flatten()

for i, (img, label) in enumerate(zip(images, labels)):
    img_np = img.permute(1, 2, 0).numpy()
    axes[i].imshow(img_np)
    axes[i].set_title(class_names[label], fontsize=8)
    axes[i].axis('off')

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.savefig('cifar10_classes_grid.png', dpi=300, bbox_inches='tight')
plt.show()
