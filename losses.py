# Import Libraries
import torch
import torch.nn as nn
import torchvision
import torchmetrics

# Perceptual Convergence Loss
class PerceptualConvergenceLoss(nn.Module):
	def __init__(self,
		feature_layers=[0, 1, 2, 3],
		style_layers=[]
	) -> None:
		"""
		VGG16 Perceptual Loss with for Real-Time Style Transfer and Super-Resolution.
		Code from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
		Loss for convergence of prediction to both frames to target-frame during supervised learning i.e with target as reference. Loss includes MSE and VGG16 Perceptual-Loss.
		"""
		super().__init__()

		# VGG16 Loss
		blocks = []
		blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
		blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
		blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
		blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
		for bl in blocks:
			for p in bl.parameters():
				p.requires_grad = False
		self.blocks = torch.nn.ModuleList(blocks).requires_grad_(False)
		self.transform = torch.nn.functional.interpolate
		self.resize = True
		self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
		self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
		self.feature_layers = feature_layers
		self.style_layers = style_layers

		# MSE Loss
		self.MSE_Loss = torch.nn.MSELoss()

	def VGG16_Loss(self, input, target):
		if input.shape[1] != 3:
			input = input.repeat(1, 3, 1, 1)
			target = target.repeat(1, 3, 1, 1)
		input = (input-self.mean) / self.std
		target = (target-self.mean) / self.std
		if self.resize:
			input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
			target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
		loss = 0.0
		x = input
		y = target
		for i, block in enumerate(self.blocks):
			x = block(x)
			y = block(y)
			if i in self.feature_layers:
				loss += torch.nn.functional.l1_loss(x, y)
			if i in self.style_layers:
				act_x = x.reshape(x.shape[0], x.shape[1], -1)
				act_y = y.reshape(y.shape[0], y.shape[1], -1)
				gram_x = act_x @ act_x.permute(0, 2, 1)
				gram_y = act_y @ act_y.permute(0, 2, 1)
				loss += torch.nn.functional.l1_loss(gram_x, gram_y)
		return loss
	
	def forward(self, output, target):
		loss = self.MSE_Loss(output, target) + self.VGG16_Loss(output, target)
		return loss
    

# Multi-Classification Loss
class Classification_Loss(nn.Module):
	def __init__(self) -> None:
		super().__init__()

		self.CrossEntropy_Loss = nn.CrossEntropyLoss(label_smoothing=0.1)

	def forward(self, y_pred, y_true):
		y_pred1, y_pred2, y_pred3, y_pred4 = y_pred
		y_true1, y_true2, y_true3, y_true4 = y_true[:,0], y_true[:,1], y_true[:,2], y_true[:,3]

		loss1 = self.CrossEntropy_Loss(y_pred1, y_true1)
		loss2 = self.CrossEntropy_Loss(y_pred2, y_true2)
		loss3 = self.CrossEntropy_Loss(y_pred3, y_true3)
		loss4 = self.CrossEntropy_Loss(y_pred4, y_true4)

		return (65/99)*loss1 + (3/99)*loss2 + (4/99)*loss3 + (27/99)*loss4


# Multi-Classification Loss
class Reconstruction_Loss(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.MSE_Loss = nn.MSELoss()

	def forward(self, y_pred, y_true):
		return self.MSE_Loss(y_pred, y_true)
	

# Multi-Classification F1 Score
def F1_Score(y_pred, y_true):
	y_pred1, y_pred2, y_pred3, y_pred4 = y_pred
	y_true1, y_true2, y_true3, y_true4 = y_true[:,0], y_true[:,1], y_true[:,2], y_true[:,3]

	score1 = torchmetrics.functional.f1_score(torch.argmax(y_pred1, dim=1), y_true1, task="multiclass", num_classes=65)
	score2 = torchmetrics.functional.f1_score(torch.argmax(y_pred2, dim=1), y_true2, task="multiclass", num_classes=3)
	score3 = torchmetrics.functional.f1_score(torch.argmax(y_pred3, dim=1), y_true3, task="multiclass", num_classes=4)
	score4 = torchmetrics.functional.f1_score(torch.argmax(y_pred4, dim=1), y_true4, task="multiclass", num_classes=27)

	return 0.25*(score1 + score2 + score3 + score4)


# Multi-Classification ROC
def AUROC(y_pred, y_true):
	y_pred1, y_pred2, y_pred3, y_pred4 = y_pred
	y_true1, y_true2, y_true3, y_true4 = y_true[:,0], y_true[:,1], y_true[:,2], y_true[:,3]

	score1 = torchmetrics.functional.auroc(y_pred1, y_true1, task="multiclass", num_classes=65)
	score2 = torchmetrics.functional.auroc(y_pred2, y_true2, task="multiclass", num_classes=3)
	score3 = torchmetrics.functional.auroc(y_pred3, y_true3, task="multiclass", num_classes=4)
	score4 = torchmetrics.functional.auroc(y_pred4, y_true4, task="multiclass", num_classes=27)

	return 0.25*(score1 + score2 + score3 + score4)


# Multi-Classification Accuracy
def Accuracy(y_pred, y_true):
	y_pred1, y_pred2, y_pred3, y_pred4 = y_pred
	y_true1, y_true2, y_true3, y_true4 = y_true[:,0], y_true[:,1], y_true[:,2], y_true[:,3]

	score1 = torchmetrics.functional.accuracy(y_pred1, y_true1, task="multiclass", num_classes=65)
	score2 = torchmetrics.functional.accuracy(y_pred2, y_true2, task="multiclass", num_classes=3)
	score3 = torchmetrics.functional.accuracy(y_pred3, y_true3, task="multiclass", num_classes=4)
	score4 = torchmetrics.functional.accuracy(y_pred4, y_true4, task="multiclass", num_classes=27)

	return 0.25*(score1 + score2 + score3 + score4)