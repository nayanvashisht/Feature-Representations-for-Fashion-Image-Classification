# Importing Libraries
import numpy as np
from PIL import Image

import torch

import models
import config


# Loading Weights
ckpt_path = "checkpoints/VGG16_Decoder/best_model.ckpt"
checkpoint = torch.load(ckpt_path)
model_weights = checkpoint["state_dict"]

# Model
encoder = models.VGG16()
decoder = models.ImageDecoder(2048)

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

# Loading and Preprocessing Image
img_path = "data/images/2480.jpg"
img = Image.open(img_path)
X = torch.unsqueeze(config.reconstruction_VGG16_test_transform(img), dim=0)

# Predicting Reconstructed Image
y_pred = decoder(encoder(X))[0]
y_pred = 0.5 + 0.5*y_pred
y_pred = y_pred.detach().numpy().transpose(1,2,0)

img.save("Original.png")

im = Image.fromarray(np.uint8(255.0*y_pred))
im.save("Reconstructed.png")