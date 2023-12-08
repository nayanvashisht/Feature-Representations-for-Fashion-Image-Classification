import torch
import torch.nn as nn
import torchvision
from torchinfo import summary


class VGG13(nn.Module):
	def __init__(self) -> None:
		super().__init__()

		self.model = torchvision.models.vgg13(weights='DEFAULT')
		self.model = self.model.features
		for param in self.model.parameters():
			param.requires_grad = False
		
		self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1,1))
		self.flatten = nn.Flatten()
 
	def forward(self, x):
		x = self.model(x)
		x = self.AdaptiveAvgPool2d(x)
		x = self.flatten(x)
		return x
	

class ResNet18(nn.Module):
	def __init__(self) -> None:
		super().__init__()

		self.model = torchvision.models.resnet18(weights='DEFAULT')
		self.model.fc = nn.Identity()
		for param in self.model.parameters():
			param.requires_grad = False
 
	def forward(self, x):
		x = self.model(x)
		return x


class FeatureExtractor(nn.Module):
	def __init__(self) -> None:
		super().__init__()

		self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
		self.act1 = nn.ReLU() 
		self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=2, padding=1)
		self.act2 = nn.ReLU()
		self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding=1)
		self.act3 = nn.ReLU() 
		self.conv4 = nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=1)
		self.act4 = nn.ReLU()
		self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

		self.conv5 = nn.Conv2d(128, 256, kernel_size=(3,3), stride=1, padding=1)
		self.act5 = nn.ReLU() 
		self.conv6 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=2, padding=1)
		self.act6 = nn.ReLU()
		self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

		self.conv7 = nn.Conv2d(256, 512, kernel_size=(3,3), stride=1, padding=1)
		self.act7 = nn.ReLU()
		self.conv8 = nn.Conv2d(512, 512, kernel_size=(3,3), stride=2, padding=1)
		self.act8 = nn.ReLU()
		self.conv9 = nn.Conv2d(512, 512, kernel_size=(3,3), stride=2, padding=1)
		self.act9 = nn.ReLU()
		self.pool3 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

		self.flatten = nn.Flatten()
 
	def forward(self, x):
		x = self.act1(self.conv1(x))
		x = self.act2(self.conv2(x))
		x = self.act3(self.conv3(x))
		x = self.act4(self.conv4(x))
		x = self.pool1(x)
		x = self.act5(self.conv5(x))
		x = self.act6(self.conv6(x))
		x = self.pool2(x)
		x = self.act7(self.conv7(x))
		x = self.act8(self.conv8(x))
		x = self.act9(self.conv9(x))
		x = self.pool3(x)
		x = self.flatten(x)
		return x
	

class ImageEncoder(nn.Module):
	def __init__(self, latent_dims):
		super().__init__()

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
		self.act1 = nn.GELU()

		self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.act2 = nn.GELU()

		self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
		self.act3 = nn.GELU()

		self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.act4 = nn.GELU()

		self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
		self.act5 = nn.GELU()

		self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
		self.act6 = nn.GELU()

		self.conv7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
		self.act7 = nn.GELU()

		self.flatten = nn.Flatten()
		self.linear1 = nn.Linear(in_features=8*8*128, out_features=latent_dims)


	def forward(self, x):
		x = self.act1(self.conv1(x))
		x = self.act2(self.conv2(x))
		x = self.act3(self.conv3(x))
		x = self.act4(self.conv4(x))
		x = self.act5(self.conv5(x))
		x = self.act6(self.conv6(x))
		x = self.act7(self.conv7(x))
		x = self.linear1(self.flatten(x))
		return x	


class ImageDecoder(nn.Module):
	def __init__(self, latent_dims):
		super().__init__()

		self.linear1 = nn.Linear(in_features=latent_dims, out_features=8*8*128)
		self.act1 = nn.GELU()

		self.convtranspose1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.act2 = nn.GELU()

		self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
		self.act3 = nn.GELU()

		self.convtranspose2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.act4 = nn.GELU()

		self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.act5 = nn.GELU()

		self.convtranspose3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.act6 = nn.GELU()

		self.conv3 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.act7 = nn.GELU()

		self.convtranspose4 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.act8 = nn.Tanh()


	def forward(self, x):
		x = self.act1(self.linear1(x))
		x = torch.reshape(x, (-1, 128, 8, 8))
		x = self.act2(self.convtranspose1(x))
		x = self.act3(self.conv1(x))
		x = self.act4(self.convtranspose2(x))
		x = self.act5(self.conv2(x))
		x = self.act6(self.convtranspose3(x))
		x = self.act7(self.conv3(x))
		x = self.act8(self.convtranspose4(x))
		return x


class Classifier(nn.Module):
	def __init__(self, input_features):
		super().__init__()

		# First Classification
		self.classfication_head1 = nn.Sequential(
			nn.Linear(in_features=input_features, out_features=512),
			nn.ReLU(),
			nn.Linear(in_features=512, out_features=256),
			nn.ReLU(),
			nn.Linear(in_features=256, out_features=256),
			nn.ReLU(),
			nn.Linear(in_features=256, out_features=65)
		)

		# Second Classification
		self.classfication_head2 = nn.Sequential(
			nn.Linear(in_features=input_features, out_features=256),
			nn.ReLU(),
			nn.Linear(in_features=256, out_features=128),
			nn.ReLU(),
			nn.Linear(in_features=128, out_features=64),
			nn.ReLU(),
			nn.Linear(in_features=64, out_features=3)
		)

		# Third Classification
		self.classfication_head3 = nn.Sequential(
			nn.Linear(in_features=input_features, out_features=256),
			nn.ReLU(),
			nn.Linear(in_features=256, out_features=128),
			nn.ReLU(),
			nn.Linear(in_features=128, out_features=64),
			nn.ReLU(),
			nn.Linear(in_features=64, out_features=4)
		)

		# Fourth Classification
		self.classfication_head4 = nn.Sequential(
			nn.Linear(in_features=input_features, out_features=512),
			nn.ReLU(),
			nn.Linear(in_features=512, out_features=256),
			nn.ReLU(),
			nn.Linear(in_features=256, out_features=256),
			nn.ReLU(),
			nn.Linear(in_features=256, out_features=27)
		)

	def forward(self, x):
		y1 = self.classfication_head1(x)
		y2 = self.classfication_head2(x)
		y3 = self.classfication_head3(x)
		y4 = self.classfication_head4(x)

		return y1,y2,y3,y4
	

"""
# Plotting Architectures
from torchview import draw_graph

Model = models.FeatureExtractor()
model_graph = draw_graph(Model, input_data=torch.randn(1,3,128,128), device='cuda')
model_graph.visual_graph"""
