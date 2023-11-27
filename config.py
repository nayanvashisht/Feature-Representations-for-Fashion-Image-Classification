# Importing Libraries
import torch
import torchvision
from torchvision import transforms
import models


# Images Shape
image_width = 128
image_height = 128

# Classification Transforms
classification_scratch_train_transform = torchvision.transforms.Compose([
		transforms.RandomApply([
				transforms.RandomHorizontalFlip(p=1),
				transforms.RandomAffine(degrees=45)
			], p=0.5),
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	]
)

classification_scratch_test_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	]
)

classification_VGG13_train_transform = torchvision.transforms.Compose([
		transforms.RandomApply([
				transforms.RandomHorizontalFlip(p=1),
				transforms.RandomAffine(degrees=45)
			], p=0.5),
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	]
)

classification_VGG13_test_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	]
)

classification_ResNet18_train_transform = torchvision.transforms.Compose([
		transforms.RandomApply([
				transforms.RandomHorizontalFlip(p=1),
				transforms.RandomAffine(degrees=45)
			], p=0.5),
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	]
)

classification_ResNet18_test_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	]
)

# Reconstruction Transforms
reconstruction_scratch_train_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	]
)

reconstruction_scratch_test_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	]
)

reconstruction_scratch_target_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	]
)

reconstruction_VGG13_train_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	]
)

reconstruction_VGG13_test_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	]
)

reconstruction_VGG13_target_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	]
)

reconstruction_ResNet18_train_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	]
)

reconstruction_ResNet18_test_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	]
)

reconstruction_ResNet18_target_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	]
)

# Normal Classification and Reconstruction Architectures
Classification_ResNet18_Classifier = {
	"feature_extractor": models.ResNet18(),
	"classifier": models.Classifier(512),
	"name": "ResNet18_Classifier"
}

Classification_VGG13_Classifier = {
	"feature_extractor": models.VGG13(),
	"classifier": models.Classifier(512),
	"name": "VGG13_Classifier"
}

Classification_FeatureExtractor_Classifier = {
	"feature_extractor": models.FeatureExtractor(),
	"classifier": models.Classifier(512),
	"name": "FeatureExtractor_Classifier"
}

Reconstruction_Encoder_Decoder = {
	"encoder": models.ImageEncoder(512),
	"decoder": models.ImageDecoder(512),
	"name": "Encoder_Decoder"
}

# Cross Architectures
Cross_Reconstruction_ResNet18_Decoder = {
	"encoder": models.ResNet18(),
	"decoder": models.ImageDecoder(512),
	"name": "ResNet18_Decoder"
}

Cross_Reconstruction_VGG13_Decoder = {
	"encoder": models.VGG13(),
	"decoder": models.ImageDecoder(512),
	"name": "VGG13_Decoder"
}

Cross_Reconstruction_FeatureExtractor_Decoder = {
	"encoder": models.FeatureExtractor(),
	"decoder": models.ImageDecoder(512),
	"name": "FeatureExtractor_Decoder"
}

Cross_Classification_Encoder_Classifier = {
	"feature_extractor": models.ImageEncoder(512),
	"classifier": models.Classifier(512),
	"name": "Encoder_Classifier"
}