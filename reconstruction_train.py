# Importing Libraries
import os

import torch, timm
from torch import nn
import torchmetrics
from torchmetrics import PeakSignalNoiseRatio as PSNR
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM
from torchmetrics.image import VisualInformationFidelity
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
		self.encoder = config.Cross_Reconstruction_FeatureExtractor_Decoder["encoder"]
		for params in self.encoder.parameters():
			params.requires_grad = False
		self.decoder = config.Cross_Reconstruction_FeatureExtractor_Decoder["decoder"]
		self.save_hyperparameters()

		# Loss
		self.train_lossfn = losses.Reconstruction_Loss()
		self.val_lossfn = losses.Reconstruction_Loss()

		# Metrics
		self.VIF = VisualInformationFidelity(sigma_n_sq=0.1)
		

	# Training-Step
	def training_step(self, batch, batch_idx):
		X, y_true = batch
		
		f = self.encoder(X)
		y_pred = self.decoder(f)

		self.train_loss = self.train_lossfn(y_pred, y_true)
		self.log('train_loss', self.train_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

		return self.train_loss


	# Validation-Step
	def validation_step(self, batch, batch_idx):
		X, y_true = batch
		
		f = self.encoder(X)
		y_pred = self.decoder(f)

		self.val_loss = self.val_lossfn(y_pred, y_true)
		self.val_vif = self.VIF(y_pred, y_true)

		self.log('val_loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
		self.log('val_vif', self.val_vif, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
		
		
	# Configure Optimizers
	def configure_optimizers(self):
		optimizer = torch.optim.Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=5e-3)
		return [optimizer]


# Main Function
def main(args):
	# Manual Arguments
	model_name = config.Cross_Reconstruction_FeatureExtractor_Decoder["name"]
	transform_type = "scratch"


	# Get Datasets
	Train_DataLoader_Module = datasets.Reconstruction_DataLoader_Module(
		images_path="data/images",
		setting="train",
		transform_type=transform_type,
		batch_size=args.batch_size,
		num_workers=64
	)
	Train_Dataloader = Train_DataLoader_Module.dataloader()
	
	Valid_DataLoader_Module = datasets.Reconstruction_DataLoader_Module(
		images_path="data/images",
		setting="valid",
		transform_type=transform_type,
		batch_size=args.batch_size,
		num_workers=64
	)
	Valid_Dataloader = Valid_DataLoader_Module.dataloader()


	# Lightning Module
	Model = Model_LightningModule(args)

	# Loading Weights
	ckpt = torch.load("checkpoints/FeatureExtractor_Classifier/best_model.ckpt")
	feature_extractor_weights = dict(filter(lambda k: 'feature_extractor' in k[0], ckpt['state_dict'].items()))
	feature_extractor_weights = {k.split('.', 1)[1]: v for k, v in feature_extractor_weights.items()}
	Model.encoder.load_state_dict(feature_extractor_weights, strict=True)


	# Checkpoint Callbacks
	best_checkpoint_callback = ModelCheckpoint(
		monitor="val_vif",
		save_top_k=1,
		mode="min",
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


	# Evaluate the Model
	if args.evaluate:
		print ("-"*25 + " Starting Evaluation on Vvalidation Set " + "-"*25)
		trainer.validate(Model, Valid_Dataloader, ckpt_path=args.resume_ckpt_path)

		print ("-"*25 + " Starting Evaluation on Test Set " + "-"*25)
		trainer.validate(Model, Valid_Dataloader, ckpt_path=args.resume_ckpt_path)


# Calling Main function
if __name__ == '__main__':
	root_dir = os.path.dirname(os.path.realpath(__file__))

	# Get Arguments
	args = arguments.Parse_Arguments()

	# Main Function
	main(args)

"""
python3 reconstruction_train.py \
--train \
--main_path "/home/krishna/Applied-ML-Project" \
--epochs 40 \
--batch_size 1024
"""