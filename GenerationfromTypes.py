import sys

import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
import pandas as pd
from model.VAE import VAE
from model.SGE2Hist import SGE2Hist
from Dataset import DataSet
from model.Unet import UNetModel
import os



BATCHSIZE = 1
EPOCHS = 600
NUM_CLASSES = 6
N_FEAT = 256
N_T = 1000
LR = 1e-4
Z_DIM = 128
DEVICE = "cuda:0"
TRAINED_PATH = ''
CLUSTERDIM = 16

vae = VAE(input_dim=19059, latent_dim=Z_DIM, hidden_dims=[2048, 512])
unet = UNetModel(in_channels=3, model_channels=64, out_channels=3, num_res_blocks=3,
                     attention_resolutions=[128, 64, 32, 16, 8], channel_mult=(1, 2, 4, 8, 16), num_head_channels=32,
                     cond_dim=Z_DIM)
model = SGE2Hist(nn_model=unet, vae=vae, betas=(1e-4, 0.02), n_T=N_T, device=DEVICE,
                                      drop_prob=0.1, nClusters=NUM_CLASSES, latent_dim=Z_DIM, Cluster_dim=CLUSTERDIM)

model.load_state_dict(torch.load(TRAINED_PATH))
transform = transforms.Compose([
        transforms.Resize((64, 64))
    ])

dataset = DataSet(datadir='', transform=transform)
    # dataloader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True)', transform=transform)
train_dataset, test_dataset = dataset.split_dataset()
train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=False)

with torch.no_grad():
    for images, genes, types in train_loader:
        genes = genes.to(DEVICE)
        for i in range(NUM_CLASSES):
            images_diff = model.SampleFromType(size=(3, 64, 64), n_sample=500, type=i, genes=genes)
            images_diff_list = [transforms.ToPILImage()(img.cpu()) for img in images_diff]
            for j, img in enumerate(images_diff_list):
                if not os.path.exists(f'TypeGeneration_brain/Type{i}'):
                    os.makedirs(f'TypeGeneration_brain/Type{i}')
                img.save(f'TypeGeneration_brain/Type{i}/image_{j}.png')
        sys.exit()