
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder(root='data', transform=transform)
print(dataset.classes)
print(f"Total images: {len(dataset)}")

image, label = dataset[0]
print(type(image))
print(image.shape)
print(image.min(), image.max())