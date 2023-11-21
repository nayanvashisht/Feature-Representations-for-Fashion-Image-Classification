import torch
import torch.nn as nn
import torchvision
from torchinfo import summary


class VGG16(nn.Module):
	def __init__(self) -> None:
		super().__init__()

		self.model = torchvision.models.vgg16(weights='DEFAULT')
		self.model = self.model.features
		for param in self.model.parameters():
			param.requires_grad = False
		
		self.AvgPool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
		self.flatten = nn.Flatten()
 
	def forward(self, x):
		x = self.model(x)
		x = self.AvgPool(x)
		x = self.flatten(x)
		return x
	

class ResNet50(nn.Module):
	def __init__(self) -> None:
		super().__init__()

		self.model = torchvision.models.resnet50(weights='DEFAULT')
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

		self.conv3 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)
		self.act3 = nn.ReLU() 
		self.conv4 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=2, padding=1)
		self.act4 = nn.ReLU()
		self.pool1 = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))

		self.conv5 = nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=1)
		self.act5 = nn.ReLU() 
		self.conv6 = nn.Conv2d(128, 256, kernel_size=(3,3), stride=2, padding=1)
		self.act6 = nn.ReLU()
		self.conv7 = nn.Conv2d(256, 512, kernel_size=(3,3), stride=2, padding=1)
		self.act7 = nn.ReLU()
		self.pool2 = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))

		self.flatten = nn.Flatten()
 
	def forward(self, x):
		x = self.act1(self.conv1(x))
		x = self.act2(self.conv2(x))
		x = self.act3(self.conv3(x))
		x = self.act4(self.conv4(x))
		x = self.pool1(x)
		x = self.act5(self.conv5(x))
		x = self.act6(self.conv6(x))
		x = self.act7(self.conv7(x))
		x = self.pool2(x)
		x = self.flatten(x)
		return x
	

class ImageEncoder(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
		self.act1 = nn.ReLU()

		self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
		self.act2 = nn.ReLU()

		self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
		self.act3 = nn.ReLU()

		self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
		self.act4 = nn.ReLU()

		self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
		self.act5 = nn.ReLU()

		self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
		self.act6 = nn.ReLU()

		self.flatten = nn.Flatten()
		self.linear1 = nn.Linear(in_features=4*4*256, out_features=2048)
		self.act7 = nn.ReLU()


	def forward(self, x):
		x = self.act1(self.conv1(x))
		x = self.act2(self.conv2(x))
		x = self.act3(self.conv3(x))
		x = self.act4(self.conv4(x))
		x = self.act5(self.conv5(x))
		x = self.act6(self.conv6(x))
		x = self.act7(self.linear1(self.flatten(x)))
		return x	


class ImageDecoder(nn.Module):
	def __init__(self, input_features):
		super().__init__()

		self.linear1 = nn.Linear(in_features=input_features, out_features=4*4*256)
		self.act1 = nn.ReLU()

		self.convtranspose1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.act2 = nn.ReLU()

		self.convtranspose2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.act3 = nn.ReLU()

		self.convtranspose3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.act4 = nn.ReLU()

		self.convtranspose4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.act5 = nn.ReLU()

		self.convtranspose5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
		self.act6 = nn.ReLU()

		self.convtranspose6 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1)
		self.act7 = nn.ReLU()


	def forward(self, x):
		x = self.act1(self.linear1(x))
		x = torch.reshape(x, (-1, 256, 4, 4))
		x = self.act2(self.convtranspose1(x))
		x = self.act3(self.convtranspose2(x))
		x = self.act4(self.convtranspose3(x))
		x = self.act5(self.convtranspose4(x))
		x = self.act6(self.convtranspose5(x))
		x = self.act7(self.convtranspose6(x))
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
			nn.Linear(in_features=256, out_features=64),
			nn.ReLU(),
			nn.Linear(in_features=64, out_features=32),
			nn.ReLU(),
			nn.Linear(in_features=32, out_features=3)
		)

		# Third Classification
		self.classfication_head3 = nn.Sequential(
			nn.Linear(in_features=input_features, out_features=256),
			nn.ReLU(),
			nn.Linear(in_features=256, out_features=64),
			nn.ReLU(),
			nn.Linear(in_features=64, out_features=32),
			nn.ReLU(),
			nn.Linear(in_features=32, out_features=4)
		)

		# Fourth Classification
		self.classfication_head4 = nn.Sequential(
			nn.Linear(in_features=input_features, out_features=256),
			nn.ReLU(),
			nn.Linear(in_features=256, out_features=128),
			nn.ReLU(),
			nn.Linear(in_features=128, out_features=64),
			nn.ReLU(),
			nn.Linear(in_features=64, out_features=27)
		)

	def forward(self, x):
		y1 = self.classfication_head1(x)
		y2 = self.classfication_head2(x)
		y3 = self.classfication_head3(x)
		y4 = self.classfication_head4(x)

		return y1,y2,y3,y4
