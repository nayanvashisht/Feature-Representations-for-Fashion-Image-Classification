# Importing Libraries
import numpy as np
import pandas as pd
from PIL import Image
import os

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder, CIFAR10
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import config


class Fashion_Dataset(torch.utils.data.Dataset):
	def __init__(self, 
	    	dataset_path,
		):
		super().__init__()
		images_path = os.path.join(dataset_path, "images")

	def __getitem__(self, idx):
		# X = Image.open(self.image_paths[idx])

		return None


	def __len__(self):
		return len(self.images_sets)
	
dataset = Fashion_Dataset(dataset_path="data")

	


class DataLoader_Module(pl.LightningDataModule):
	def __init__(self,
		batch_size,
		num_workers,
		data,
		transform_type
	) -> None:
		"""
		Args:
			args: Arguments
		"""
		# Selecting Image Transform based on Model Type
		if transform_type == "scratch":
			self.train_transforms = config.scratch_train_transform
			self.train_transforms = config.scratch_train_transform
		elif transform_type == "vgg16":
			self.train_transforms = config.VGG16_train_transform
			self.train_transforms = config.VGG16_train_transform
		elif transform_type == "inceptionnet":
			self.train_transforms = config.InceptionNet_train_transform
			self.train_transforms = config.InceptionNet_test_transform

		super().__init__()
		self.batch_size = batch_size
		self.num_workers = num_workers

		self.dataset = Fashion_Dataset(data)
		train_size = int(0.6 * len(self.dataset))
		valid_size = int(0.5*(len(self.dataset) - train_size))
		test_size = train_size - valid_size - test_size
		self.train, self.val = torch.utils.data.random_split(self.dataset, [train_size, valid_size, test_size],generator=torch.Generator().manual_seed(7))



	def train_dataloader(self):
			return DataLoader(self.train, batch_size=self.batch_size, num_workers = self.num_workers, shuffle=True)

	def val_dataloader(self):
			return DataLoader(self.val, batch_size=self.batch_size, num_workers = self.num_workers)

	def test_dataloader(self):
			return DataLoader(self.val, batch_size=self.batch_size, num_workers = self.num_workers)