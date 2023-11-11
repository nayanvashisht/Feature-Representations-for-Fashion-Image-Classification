import torch
import torch.nn as nn
import torchvision
from torchinfo import summary

class Feature_Model(nn.Module):
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
		return x
	

class VGG16_Feature_Model(nn.Module):
	def __init__(self) -> None:
		super().__init__()

		self.model = torchvision.models.vgg16(torchvision.models.VGG16_Weights.DEFAULT)
		self.model = self.model.features
		for param in self.model.parameters():
			param.requires_grad = False
 
	def forward(self, x):
		return self.model(x)
	

class Incpetion_Feature_Model(nn.Module):
	def __init__(self) -> None:
		super().__init__()

		self.model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT)
		self.model.fc = nn.Identity()
		for param in self.model.parameters():
			param.requires_grad = False
 
	def forward(self, x):
		return self.model(x)

# A = Feature_Model()
# summary(A, input_data=torch.randn(1,3,224,224))
# print ()

# B = VGG16_Feature_Model()
# summary(B, input_data=torch.randn(1,3,224,224))
# print ()

C= Incpetion_Feature_Model()
summary(C, input_data=torch.randn(1,3,299,299))
print ()