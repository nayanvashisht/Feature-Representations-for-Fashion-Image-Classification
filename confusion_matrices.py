# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch

import datasets
import models
import config


# Loading and Preprocessing Image
Info = [
	("Encoder", models.ImageEncoder(512), models.Classifier(512), "scratch"),
	("FeatureExtractor", models.FeatureExtractor(), models.Classifier(512), "scratch"),
	("VGG13", models.VGG13(), models.Classifier(512), "vgg13"),
	("ResNet18", models.ResNet18(), models.Classifier(512), "resnet18"),
]

# Maps
F = datasets.Classification_Fashion_Dataset(
	dataset_path="data/test.csv",
	images_path="data/images",
	transform=None
)
Index2Label = F.Index2Label
labels = list(Index2Label["subCategory"].values())
for i in range(len(labels)):
	labels[i] = labels[i].replace("subCategory_", "")

for name, feature_extractor, classifer, type in Info:
	# Loading Weights
	ckpt_path = "checkpoints/{}_Classifier/best_model.ckpt".format(name)
	checkpoint = torch.load(ckpt_path)
	model_weights = checkpoint["state_dict"]

	# Loading Feature-Extractor Weights
	feature_extractor_weights = model_weights.copy()
	for key in list(feature_extractor_weights):
		if "feature_extractor." in key:
			feature_extractor_weights[key.replace("feature_extractor.", "")] = feature_extractor_weights.pop(key)
		else:
			feature_extractor_weights.pop(key)
	feature_extractor.load_state_dict(feature_extractor_weights)

	# Loading Classifier Weights
	classifer_weights = model_weights.copy()
	for key in list(classifer_weights):
		if "classifer." in key:
			classifer_weights[key.replace("classifer.", "")] = classifer_weights.pop(key)
		else:
			classifer_weights.pop(key)
	classifer.load_state_dict(classifer_weights)

	# Test DataLoader
	Test_DataLoader_Module = datasets.Classification_DataLoader_Module(
		dataset_path="data/test.csv",
		images_path="data/images",
		setting="valid",
		transform_type=type,
		batch_size=64,
		num_workers=32
	)
	Test_Dataloader = Test_DataLoader_Module.dataloader()

	Y_True = []
	Y_Pred = []

	for X, y_true in Test_Dataloader:
		f = feature_extractor(X)
		y_pred = classifer(f)

		y_true_1 = y_true[:, 0].numpy()
		y_pred_1 = torch.argmax(y_pred[0], dim=1).numpy()

		Y_True.append(y_true_1)
		Y_Pred.append(y_pred_1)

	Y_True = np.concatenate(Y_True, axis=0)
	Y_Pred = np.concatenate(Y_Pred, axis=0)

	cf_matrix = confusion_matrix(y_true=Y_True, y_pred=Y_Pred, labels=np.arange(len(labels)))
	cf_matrix = cf_matrix.astype(np.float16)
	for i in range(cf_matrix.shape[0]):
		cf_matrix[i] = cf_matrix[i]/np.sum(cf_matrix[i])
	df_cm = pd.DataFrame(cf_matrix, index = [i for i in labels], columns = [i for i in labels])
	plt.figure(figsize = (24,18))
	plt.title("Target: {}, Model: {}".format("subCategory", name))
	sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='.2f')
	plt.savefig("plots/{}_subCategory_confusion_matrix.png".format(name), bbox_inches='tight')

	print (name, "Done")