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
		transforms.ToTensor()
	]
)

classification_scratch_test_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor()
	]
)

classification_VGG16_train_transform = torchvision.transforms.Compose([
		transforms.RandomApply([
				transforms.RandomHorizontalFlip(p=1),
				transforms.RandomAffine(degrees=45)
			], p=0.5),
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.48235, 0.45882, 0.40784], std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])
	]
)

classification_VGG16_test_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.48235, 0.45882, 0.40784], std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])
	]
)

classification_ResNet50_train_transform = torchvision.transforms.Compose([
		transforms.RandomApply([
				transforms.RandomHorizontalFlip(p=1),
				transforms.RandomAffine(degrees=45)
			], p=0.5),
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	]
)

classification_ResNet50_test_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	]
)

# Reconstruction Transforms
reconstruction_scratch_train_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor()
	]
)

reconstruction_scratch_test_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor()
	]
)

reconstruction_scratch_target_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor()
	]
)

reconstruction_VGG16_train_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.48235, 0.45882, 0.40784], std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])
	]
)

reconstruction_VGG16_test_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.48235, 0.45882, 0.40784], std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])
	]
)

reconstruction_VGG16_target_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor()
	]
)

reconstruction_ResNet50_train_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	]
)

reconstruction_ResNet50_test_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	]
)

reconstruction_ResNet50_target_transform = torchvision.transforms.Compose([
		transforms.Resize((image_height, image_width)),
		transforms.ToTensor()
	]
)

# Normal Classification and Reconstruction Architectures
Classification_ResNet50_Classifier = {
	"feature_extractor": models.ResNet50(),
	"classifier": models.Classifier(2048),
	"name": "ResNet50_Classifier"
}

Classification_VGG16_Classifier = {
	"feature_extractor": models.VGG16(),
	"classifier": models.Classifier(2048),
	"name": "VGG16_Classifier"
}

Classification_FeatureExtractor_Classifier = {
	"feature_extractor": models.FeatureExtractor(),
	"classifier": models.Classifier(2048),
	"name": "FeatureExtractor_Classifier"
}

Reconstruction_Encoder_Decoder = {
	"encoder": models.ImageEncoder(),
	"decoder": models.ImageDecoder(2048),
	"name": "Encoder_Decoder"
}

# Cross Architectures
Cross_Reconstruction_ResNet50_Decoder = {
	"encoder": models.ResNet50(),
	"decoder": models.ImageDecoder(2048),
	"name": "ResNet50_Decoder"
}

Cross_Reconstruction_VGG16_Decoder = {
	"encoder": models.VGG16(),
	"decoder": models.ImageDecoder(2048),
	"name": "VGG16_Decoder"
}

Cross_Reconstruction_FeatureExtractor_Decoder = {
	"encoder": models.FeatureExtractor(),
	"decoder": models.ImageDecoder(2048),
	"name": "FeatureExtractor_Decoder"
}

Cross_Classification_Encoder_Classifier = {
	"feature_extractor": models.ImageEncoder(),
	"classifier": models.Classifier(2048),
	"name": "Encoder_Classifier"
}