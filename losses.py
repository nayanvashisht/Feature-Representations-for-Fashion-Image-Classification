# Import Libraries
import torch
import torch.nn as nn
import torchmetrics

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