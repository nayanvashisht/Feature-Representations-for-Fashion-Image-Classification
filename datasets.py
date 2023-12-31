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


class Classification_Fashion_Dataset(torch.utils.data.Dataset):
	def __init__(self, 
		dataset_path,
		images_path,
		transform
	):
		"""
		Args:
			dataset_path (str): Path to Dataset i.e csv file.
			images_path (str): Path to Images folder.
			transform: Image Transform.
		"""
		super().__init__()

		# Transform
		self.transform = transform

		# Images Path
		self.images_path = images_path

		# Classification Targets
		self.target_columns = ['articleType', 'gender', 'masterCategory', 'subCategory']
		self.target_columns.sort()

		# Loading Dataset
		df = pd.read_csv(dataset_path)
		self.length = df.shape[0]

		# Inputs and Target Info
		self.create_maps(df)
		self.extract_inputs_targets(df)

	def create_maps(self,
		df
	):
		"""
		Creating maps.
		"""
		self.Index2Encode = {}
		self.Index2Label = {}
		self.Label2Index = {}

		for target_column in self.target_columns:
			columns = []
			for column_name in df.columns:
				if target_column in column_name:
					columns.append(column_name)
			columns.sort()

			I = np.eye(len(columns))
			self.Index2Label[target_column] = {k:v for k,v in enumerate(columns)}
			self.Label2Index[target_column] = {v:k for k,v in enumerate(columns)}
			self.Index2Encode[target_column] = {k:list(I[k]) for k,v in enumerate(columns)}

	def extract_inputs_targets(self,
		df:pd.DataFrame
	):
		"""
		Extracting Inputs and Targets of the Dataframe
		"""
		# Input Images Locations
		self.image_filename = df["id"].to_numpy()

		# Targets One-hot Encodings
		self.target_one_hot_labels = []

		for target_column in self.target_columns:
			columns = []
			for column_name in df.columns:
				if target_column in column_name:
					columns.append(column_name)
			columns.sort()

			t = np.argmax(df[columns].to_numpy(), axis=-1)
			self.target_one_hot_labels.append(t)

	def __getitem__(self, idx):
		while os.path.exists(os.path.join(self.images_path, str(self.image_filename[idx])+".jpg")) == False:
			idx = idx+1
			if idx >= self.length:
				idx = idx%self.length
		X = Image.open(os.path.join(self.images_path, str(self.image_filename[idx])+".jpg")).convert("RGB")
		X = self.transform(X)
		y = torch.from_numpy(np.asarray([self.target_one_hot_labels[i][idx] for i in range(len(self.target_columns))]).astype("int"))

		return X,y

	def __len__(self):
		return self.length



class Classification_DataLoader_Module(pl.LightningDataModule):
	def __init__(self,
		dataset_path,
		images_path,
		setting,
		transform_type,
		batch_size,
		num_workers=16,
	) -> None:
		"""
		Args:
			args: Arguments
		"""
		# Selecting Image Transform based on Model Type
		if transform_type == "scratch":
			self.train_transforms = config.classification_scratch_train_transform
			self.test_transforms = config.classification_scratch_test_transform
		elif transform_type == "vgg13":
			self.train_transforms = config.classification_VGG13_train_transform
			self.test_transforms = config.classification_VGG13_test_transform
		elif transform_type == "resnet18":
			self.train_transforms = config.classification_ResNet18_train_transform
			self.test_transforms = config.classification_ResNet18_test_transform
		else:
			assert False, "Invalid transform_type"

		super().__init__()
		self.batch_size = batch_size
		self.num_workers = num_workers

		# Dataset
		if setting == "train":
			self.dataset = Classification_Fashion_Dataset(dataset_path=dataset_path, images_path=images_path, transform=self.train_transforms)
		else:
			self.dataset = Classification_Fashion_Dataset(dataset_path=dataset_path, images_path=images_path, transform=self.test_transforms)

	def dataloader(self):
			return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
	


class Reconstruction_Fashion_Dataset(torch.utils.data.Dataset):
	def __init__(self,
		dataset_path,
		images_path,
		input_transform,
		target_transform
	):
		"""
		Args:
			dataset_path (str): Path to Dataset i.e csv file.
			images_path (str): Path to Images folder.
			transform: Image Transform.
			target_transform: Target Transform.
		"""
		super().__init__()

		# Transform
		self.input_transform = input_transform
		self.target_transform = target_transform

		# Images Path
		self.images_path = images_path

		# Loading Dataset
		df = pd.read_csv(dataset_path)
		self.length = df.shape[0]

		# Inputs and Target Info
		self.extract_inputs_targets(df)

	def extract_inputs_targets(self,
		df:pd.DataFrame
	):
		"""
		Extracting Inputs and Targets of the Dataframe
		"""
		# Input Images Locations
		self.image_filename = df["id"].to_numpy()

	def __getitem__(self, idx):
		while os.path.exists(os.path.join(self.images_path, str(self.image_filename[idx])+".jpg")) == False:
			idx = idx+1
			if idx >= self.length:
				idx = idx%self.length

		X = Image.open(os.path.join(self.images_path, str(self.image_filename[idx])+".jpg")).convert("RGB")
		X = self.input_transform(X)
		X = X + (np.sqrt(0.1) * torch.randn(X.shape))

		y = Image.open(os.path.join(self.images_path, str(self.image_filename[idx])+".jpg")).convert("RGB")
		y = self.target_transform(y)

		return X,y

	def __len__(self):
		return self.length
	


class Reconstruction_DataLoader_Module(pl.LightningDataModule):
	def __init__(self,
		dataset_path,
		images_path,
		setting,
		transform_type,
		batch_size,
		num_workers=16,
	) -> None:
		"""
		Args:
			args: Arguments
		"""
		# Selecting Image Transform based on Model Type
		if transform_type == "scratch":
			self.train_transforms = config.reconstruction_scratch_train_transform
			self.test_transforms = config.reconstruction_scratch_test_transform
			self.target_transforms = config.reconstruction_scratch_target_transform
		elif transform_type == "vgg13":
			self.train_transforms = config.reconstruction_VGG13_train_transform
			self.test_transforms = config.reconstruction_VGG13_test_transform
			self.target_transforms = config.reconstruction_VGG13_target_transform
		elif transform_type == "resnet18":
			self.train_transforms = config.reconstruction_ResNet18_train_transform
			self.test_transforms = config.reconstruction_ResNet18_test_transform
			self.target_transforms = config.reconstruction_ResNet18_target_transform
		else:
			assert False, "Invalid transform_type"

		super().__init__()
		self.batch_size = batch_size
		self.num_workers = num_workers

		# Dataset
		if setting == "train":
			self.dataset = Reconstruction_Fashion_Dataset(dataset_path=dataset_path, images_path=images_path, input_transform=self.train_transforms, target_transform=self.target_transforms)
		else:
			self.dataset = Reconstruction_Fashion_Dataset(dataset_path=dataset_path, images_path=images_path, input_transform=self.test_transforms, target_transform=self.target_transforms)

	def dataloader(self):
			return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)