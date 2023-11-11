# Importing Libraries
import torch
import torchvision
from torchvision import transforms


# Images Shape
image_width = 96
image_height = 96

# Image Transforms
scratch_train_transform = torchvision.transforms.Compose([
		transforms.RandomApply([
				transforms.RandomHorizontalFlip(p=1),
				transforms.RandomAffine(degrees=45)
			], p=0.5),
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor()
	]
)

scratch_test_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor()
	]
)

VGG16_train_transform = torchvision.transforms.Compose([
		transforms.RandomApply([
				transforms.RandomHorizontalFlip(p=1),
				transforms.RandomAffine(degrees=45)
			], p=0.5),
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.48235, 0.45882, 0.40784], std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])
	]
)

VGG16_test_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.48235, 0.45882, 0.40784], std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])
	]
)

InceptionNet_train_transform = torchvision.transforms.Compose([
		transforms.RandomApply([
				transforms.RandomHorizontalFlip(p=1),
				transforms.RandomAffine(degrees=45)
			], p=0.5),
		transforms.Resize((299, 299)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	]
)

InceptionNet_test_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	]
)



