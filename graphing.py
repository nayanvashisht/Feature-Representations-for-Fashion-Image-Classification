from models import VGG13, ResNet18, FeatureExtractor, ImageEncoder
import torch
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.layers import Dense
import pandas as pd
from PIL import Image
import config
# from utils import Load_Model, Load_Model_FE, Load_Model_IE, Load_Model_RN50
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded

# REPLACE with your data directory
data_directory = r'C:\Users\surya\Desktop\UT Austin\Classes\ECE 381K Applied Machine Learning\fashion-dataset\images'

FE = FeatureExtractor()
RN50 = ResNet18()
VG = VGG13()
IE = ImageEncoder(512)

# REPLACE with your checkpoint files
vg_cp_path = r'C:\Users\surya\PycharmProjects\UT_Testing_Env\Applied-ML-Project-main\new_checkpoints\VGG13_Classifier\best_model.ckpt'  # Replace with your .ckpt file path
vg_cp = torch.load(vg_cp_path, map_location=torch.device('cpu'))

fe_cp_path = r'C:\Users\surya\PycharmProjects\UT_Testing_Env\Applied-ML-Project-main\new_checkpoints\FeatureExtractor_Classifier\best_model.ckpt'  # Replace with your .ckpt file path
fe_cp = torch.load(fe_cp_path, map_location=torch.device('cpu'))

ie_cp_path = r'C:\Users\surya\PycharmProjects\UT_Testing_Env\Applied-ML-Project-main\new_checkpoints\Encoder_Classifier\best_model.ckpt'  # Replace with your .ckpt file path
ie_cp = torch.load(ie_cp_path, map_location=torch.device('cpu'))

rn50_cp_path = r'C:\Users\surya\PycharmProjects\UT_Testing_Env\Applied-ML-Project-main\new_checkpoints\ResNet18_Classifier\best_model.ckpt'  # Replace with your .ckpt file path
rn50_cp = torch.load(rn50_cp_path, map_location=torch.device('cpu'))

# If the .ckpt file contains a state_dict
# Model Weights
model_weights = vg_cp["state_dict"].copy()
for key in list(model_weights):
    if "feature_extractor." in key:
        model_weights[key.replace("feature_extractor.", "")] = model_weights.pop(key)
    else:
        model_weights.pop(key)
VG.load_state_dict(model_weights)


model_weights = fe_cp["state_dict"].copy()
for key in list(model_weights):
    if "feature_extractor." in key:
        model_weights[key.replace("feature_extractor.", "")] = model_weights.pop(key)
    else:
        model_weights.pop(key)
FE.load_state_dict(model_weights)

model_weights = ie_cp["state_dict"].copy()
for key in list(model_weights):
    if "feature_extractor." in key:
        model_weights[key.replace("feature_extractor.", "")] = model_weights.pop(key)
    else:
        model_weights.pop(key)

IE.load_state_dict(model_weights)

model_weights = rn50_cp["state_dict"].copy()
for key in list(model_weights):
    if "feature_extractor." in key:
        model_weights[key.replace("feature_extractor.", "")] = model_weights.pop(key)
    else:
        model_weights.pop(key)

RN50.load_state_dict(model_weights)

# Replace with your .csv file
df = pd.read_csv(r'C:\Users\surya\PycharmProjects\UT_Testing_Env\Applied-ML-Project-main\new_checkpoints\test.csv')
# Change to match columns of interest
target = 'articleType'
cols_of_interest = [col for col in df.columns if target in col]
print(cols_of_interest)
id_set = {}
for col in cols_of_interest:
    cur_rows = df[df[col] == 1]
    id_set[col] = cur_rows.head(20)['id']

# female_rows = df[df['gender_Female'] == 1]
# female_ids = female_rows.head(100)['id']
#
# unisex_rows = df[df['gender_Unisex'] == 1]
# unisex_ids = unisex_rows.head(100)['id']
print("Generating Features")
vgg_features= {}
rn50_features = {}
encoder_features = {}
feature_extractor_features = {}
for col in id_set.keys():
    vgg_features[col] = []
    rn50_features[col] = []
    encoder_features[col] = []
    feature_extractor_features[col] = []
    for id in id_set[col]:
        test_image_directory = data_directory + r'\\' + str(id) + '.jpg'
        image_rgb = Image.open(test_image_directory).convert("RGB")
        vgg_image_rgb = config.reconstruction_VGG13_test_transform(image_rgb)
        rn50_image_rgb = config.reconstruction_ResNet18_test_transform(image_rgb)
        encoder_image_rgb = config.reconstruction_scratch_test_transform(image_rgb)
        feature_extractor_image_rgb = config.reconstruction_scratch_test_transform(image_rgb)

        vgg_features[col].append(VG.forward(torch.unsqueeze(vgg_image_rgb, dim=0)))
        rn50_features[col].append(RN50.forward(torch.unsqueeze(rn50_image_rgb, dim=0)))
        encoder_features[col].append(IE.forward(torch.unsqueeze(encoder_image_rgb, dim=0)))
        feature_extractor_features[col].append(FE.forward(torch.unsqueeze(feature_extractor_image_rgb, dim=0)))

all_features = []
vgg_features_list = []
rn50_features_list = []
encoder_features_list = []
feature_extractor_features_list = []
# Append features from each model and image to a single list
for col in id_set.keys():
    vgg_features_list.extend(vgg_features[col])
    rn50_features_list.extend(rn50_features[col])
    encoder_features_list.extend(encoder_features[col])
    feature_extractor_features_list.extend(feature_extractor_features[col])
vgg_tensor = torch.stack(vgg_features_list)
vgg_tensor = vgg_tensor.view(-1, 512)

rn50_tensor = torch.stack(rn50_features_list)
rn50_tensor = rn50_tensor.view(-1, 512)

encoder_tensor = torch.stack(encoder_features_list)
encoder_tensor = encoder_tensor.view(-1, 512)

feature_extractor_tensor = torch.stack(feature_extractor_features_list)
feature_extractor_tensor = feature_extractor_tensor.view(-1, 512)


# Apply PCA to reduce the dimensionality
num_dimensions = 2  # Choose the desired number of dimensions

pca_vgg = umap.UMAP(n_components=num_dimensions)
vgg_np = vgg_tensor.detach().numpy()
# pca_vgg.fit(vgg_np)

pca_rn50 = umap.UMAP(n_components=num_dimensions)
rn50_np = rn50_tensor.detach().numpy()
# pca_rn50.fit(rn50_np)

pca_encoder = umap.UMAP(n_components=num_dimensions)
encoder_np = encoder_tensor.detach().numpy()
# pca_encoder.fit(encoder_np)

pca_fe = umap.UMAP(n_components=num_dimensions)
feature_extractor_np = feature_extractor_tensor.detach().numpy()
# pca_fe.fit(feature_extractor_np)
# Transform the data using each fitted PCA model
transformed_vgg = {}
transformed_rn50 = {}
transformed_encoder = {}
transformed_feature_extractor = {}
print("Transforming Features")
for col in id_set.keys():
    vgg_features_temp = np.vstack([tensor.detach().numpy() for tensor in vgg_features[col]])
    vgg_features_temp = vgg_features_temp.reshape(len(vgg_features_temp), -1)
    transformed_vgg[col] = pca_vgg.fit_transform(vgg_features_temp)

    rn50_features_temp = np.vstack([tensor.detach().numpy() for tensor in rn50_features[col]])
    rn50_features_temp = rn50_features_temp.reshape(len(rn50_features_temp), -1)
    transformed_rn50[col] = pca_rn50.fit_transform(rn50_features_temp)

    encoder_features_temp = np.vstack([tensor.detach().numpy() for tensor in encoder_features[col]])
    encoder_features_temp = encoder_features_temp.reshape(len(encoder_features_temp), -1)
    transformed_encoder[col] = pca_encoder.fit_transform(encoder_features_temp)

    fe_features_temp = np.vstack([tensor.detach().numpy() for tensor in feature_extractor_features[col]])
    fe_features_temp = fe_features_temp.reshape(len(fe_features_temp), -1)
    transformed_feature_extractor[col] = pca_fe.fit_transform(fe_features_temp)


# Plotting for VGG
# figure(figsize=(12, 8))
for col in id_set.keys():
    plt.scatter(transformed_vgg[col][:, 0], transformed_vgg[col][:, 1], label=col)

plt.title('VGG - Categorization')
# plt.legend()
plt.show()

for col in id_set.keys():
    plt.scatter(transformed_rn50[col][:, 0], transformed_rn50[col][:, 1], label=col)
plt.title('RN18 - Categorization')
# plt.legend()
plt.show()

for col in id_set.keys():
    plt.scatter(transformed_encoder[col][:, 0], transformed_encoder[col][:, 1], label=col)
plt.title('Encoder - Categorization')
# plt.legend()
plt.show()

for col in id_set.keys():
    plt.scatter(transformed_feature_extractor[col][:, 0], transformed_feature_extractor[col][:, 1], label=col)
plt.title('Feature Extractor - Categorization')
# plt.legend()
plt.show()



# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
#
# # Extracting X, Y, Z coordinates from PCA result
# X = transformed_vgg_male[:, 0]
# Y = transformed_vgg_male[:, 1]
# Z = transformed_vgg_male[:, 2]
# ax.scatter(X, Y, Z, label='Male')
#
# X = transformed_vgg_female[:, 0]
# Y = transformed_vgg_female[:, 1]
# Z = transformed_vgg_female[:, 2]
# ax.scatter(X, Y, Z, label='Female')
#
# X = transformed_vgg_uni[:, 0]
# Y = transformed_vgg_uni[:, 1]
# Z = transformed_vgg_uni[:, 2]
# ax.scatter(X, Y, Z, label='Unisex')
#
#
# ax.set_xlabel('Component 1')
# ax.set_ylabel('Component 2')
# ax.set_zlabel('Component 3')
# ax.set_title('VGG - Gender Categorization 3D')
