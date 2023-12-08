# Importing Libraries
import numpy as np
import pandas as pd
from PIL import Image

import torch

import models
import config

# Loading Test Dataset
df = pd.read_csv("data/test.csv")
ids = df["id"].to_numpy()
sample_indices = np.random.randint(0, ids.shape[0], 6)
print (sample_indices)


# Loading and Preprocessing Image
for i in range(6):
	img_path = "data/images/{}.jpg".format(sample_indices[i])
	img = Image.open(img_path)
	img.save("plots/imgs/Original-Sample-{}.png".format(i+1))

	Info = [
		("Encoder", models.ImageEncoder(512), models.ImageDecoder(512)),
		("FeatureExtractor", models.FeatureExtractor(), models.ImageDecoder(512)),
		("VGG13", models.VGG13(), models.ImageDecoder(512)),
		("ResNet18", models.ResNet18(), models.ImageDecoder(512)),
	]
	for name, encoder, decoder in Info:
		# Loading Weights
		ckpt_path = "checkpoints/{}_Decoder/best_model.ckpt".format(name)
		checkpoint = torch.load(ckpt_path)
		model_weights = checkpoint["state_dict"]

		# Loading Encoder Weights
		encoder_weights = model_weights.copy()
		for key in list(encoder_weights):
			if "encoder." in key:
				encoder_weights[key.replace("encoder.", "")] = encoder_weights.pop(key)
			else:
				encoder_weights.pop(key)
		encoder.load_state_dict(encoder_weights)

		# Loading Decoder Weights
		decoder_weights = model_weights.copy()
		for key in list(decoder_weights):
			if "decoder." in key:
				decoder_weights[key.replace("decoder.", "")] = decoder_weights.pop(key)
			else:
				decoder_weights.pop(key)
		decoder.load_state_dict(decoder_weights)

		# Predicting Reconstructed Image
		if name == "Encoder" or name == "FeatureExtractor":
			X = torch.unsqueeze(config.reconstruction_scratch_test_transform(img), dim=0)
		elif name == "VGG13":
			X = torch.unsqueeze(config.reconstruction_VGG13_test_transform(img), dim=0)
		elif name == "ResNet18":
			X = torch.unsqueeze(config.reconstruction_ResNet18_test_transform(img), dim=0)

		y_pred = decoder(encoder(X))[0]
		y_pred = 0.5 + 0.5*y_pred
		y_pred = y_pred.detach().numpy().transpose(1,2,0)
		
		im = Image.fromarray(np.uint8(255.0*y_pred))
		im.save("plots/imgs/{}-Reconstructed-Sample-{}.png".format(name, i+1))