# Importing Libraries
import os

import torch, timm
from torch import nn
import torchmetrics
from torchmetrics import PeakSignalNoiseRatio as PSNR
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchinfo import summary
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies.ddp import DDPStrategy

import itertools
import datasets
import models
import utils
import arguments
import losses
import config
import warnings
warnings.filterwarnings("ignore")


# Lightning Module
class Model_LightningModule(pl.LightningModule):
	def __init__(self, args):
		super().__init__()
		self.args = args

		# Model as Manual Arguments
		self.feature_extractor = config.Cross_Classification_Encoder_Classifier["feature_extractor"]
		self.classifer = config.Cross_Classification_Encoder_Classifier["classifier"]
		self.save_hyperparameters()

		# Weights
		ckpt = torch.load("checkpoints/Encoder_Decoder/best_model.ckpt")
		feature_extractor_weights = dict(filter(lambda k: 'encoder' in k[0], ckpt['state_dict'].items()))
		feature_extractor_weights = {k.split('.', 1)[1]: v for k, v in feature_extractor_weights.items()}
		self.feature_extractor.load_state_dict(feature_extractor_weights, strict=True)

		for params in self.feature_extractor.parameters():
			params.requires_grad = False

		# Loss
		self.train_lossfn = losses.Classification_Loss()
		self.val_lossfn = losses.Classification_Loss()
		

	# Training-Step
	def training_step(self, batch, batch_idx):
		X, y_true = batch
		
		f = self.feature_extractor(X)
		y_pred = self.classifer(f)

		self.train_loss = self.train_lossfn(y_pred, y_true)
		self.log('train_loss', self.train_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

		return self.train_loss


	# Validation-Step
	def validation_step(self, batch, batch_idx):
		X, y_true = batch
		
		f = self.feature_extractor(X)
		y_pred = self.classifer(f)

		self.val_loss = self.val_lossfn(y_pred, y_true)
		self.val_acc = losses.Accuracy(y_pred, y_true)
		self.val_f1score = losses.F1_Score(y_pred, y_true)
		self.val_auroc = losses.AUROC(y_pred, y_true)

		self.log('val_loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
		self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
		self.log('val_f1score', self.val_f1score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
		self.log('val_auroc', self.val_auroc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
		
		
	# Configure Optimizers
	def configure_optimizers(self):
		optimizer = torch.optim.Adam(itertools.chain(self.feature_extractor.parameters(), self.classifer.parameters()), lr=1e-4)
		return [optimizer]


# Main Function
def main(args):
	# Manual Arguments
	model_name = config.Cross_Classification_Encoder_Classifier["name"]
	transform_type = "scratch"

	# Get Datasets
	Train_DataLoader_Module = datasets.Classification_DataLoader_Module(
		dataset_path="data/training.csv",
		images_path="data/images",
		setting="train",
		transform_type=transform_type,
		batch_size=args.batch_size,
		num_workers=64
	)
	Train_Dataloader = Train_DataLoader_Module.dataloader()
	
	Valid_DataLoader_Module = datasets.Classification_DataLoader_Module(
		dataset_path="data/validation.csv",
		images_path="data/images",
		setting="valid",
		transform_type=transform_type,
		batch_size=args.batch_size,
		num_workers=64
	)
	Valid_Dataloader = Valid_DataLoader_Module.dataloader()

	Test_DataLoader_Module = datasets.Classification_DataLoader_Module(
		dataset_path="data/test.csv",
		images_path="data/images",
		setting="valid",
		transform_type=transform_type,
		batch_size=args.batch_size,
		num_workers=64
	)
	Test_Dataloader = Test_DataLoader_Module.dataloader()


	# Lightning Module
	Model = Model_LightningModule(args)


	# Checkpoint Callbacks
	best_checkpoint_callback = ModelCheckpoint(
		monitor="val_f1score",
		mode="max",
		dirpath=os.path.join(args.main_path,"checkpoints",model_name),
		filename="best_model",
	)

	# Resume Training from checkpoint.
	if args.resume_ckpt_path is not None:
		if os.path.isfile(args.resume_ckpt_path):
			print ("Found the checkpoint at resume_ckpt_path provided.")
		else:
			args.resume_ckpt_path = None	# The given variable is altered as it is provided as input to ".fit".
			print("Resume checkpoint not found in the resume_ckpt_path provided. Starting training from the begining.")
	else:
		print ("No path is provided for resume checkpoint (resume_ckpt_path) provided. Starting training from the begining.")


	# PyTorch Lightning Trainer
	trainer = pl.Trainer(
		accelerator="gpu",
		devices = args.gpu,
		callbacks=[best_checkpoint_callback, utils.LitProgressBar()],
		num_nodes=args.num_nodes,
		max_epochs=args.epochs,
		logger=pl_loggers.TensorBoardLogger(save_dir=args.main_path)
	)

	# Training the Model
	if args.train:
		print ("-"*25 + " Starting Training " + "-"*25)
		trainer.fit(Model, train_dataloaders=Train_Dataloader, val_dataloaders=Valid_Dataloader, ckpt_path=args.resume_ckpt_path)

		print ("Final Evaluation of Training Dataset")
		trainer.validate(Model, Train_Dataloader, ckpt_path=args.resume_ckpt_path)

		print ("Final Evaluation of Validation Dataset")
		trainer.validate(Model, Valid_Dataloader, ckpt_path=args.resume_ckpt_path)

		print ("Final Evaluation on Test Dataset")
		trainer.validate(Model, Test_Dataloader, ckpt_path=args.resume_ckpt_path)

	# Evaluate the Model
	if args.evaluate:
		print ("-"*25 + " Starting Evaluation on Validation Set " + "-"*25)
		trainer.validate(Model, Valid_Dataloader, ckpt_path=args.resume_ckpt_path)

		print ("-"*25 + " Starting Evaluation on Test Set " + "-"*25)
		trainer.validate(Model, Test_Dataloader, ckpt_path=args.resume_ckpt_path)


# Calling Main function
if __name__ == '__main__':
	root_dir = os.path.dirname(os.path.realpath(__file__))

	# Get Arguments
	args = arguments.Parse_Arguments()

	# Main Function
	main(args)


"""
python3 classification_train.py \
--train \
--main_path "/home/krishna/Applied-ML-Project" \
--epochs 20 \
--batch_size 256
"""