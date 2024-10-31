import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
import pandas as pd
from model.VAE import VAE
from model.SGE2Hist import SGE2Hist
from Dataset import DataSet
from model.Unet import UNetModel
from PIL import Image

BATCHSIZE = 10
EPOCHS = 600
NUM_CLASSES = 6
N_FEAT = 256
N_T = 1000
LR = 1e-4
Z_DIM = 128
DEVICE = "cuda:0"
TRAINED_PATH = ''
SuperResolutionPATH = ''
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

dataset = DataSet(datadir='', transform=None)
train_dataset, test_dataset = dataset.split_dataset()
# train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False)

with torch.no_grad():
    for images, genes, types in test_loader:
        genes = genes.to(DEVICE)
        z_mu, z_logvar = model.vae.encoder(genes)
        images_diff, images_decode = model.SamplefromGenes(genes, (3, 64, 64), BATCHSIZE)
        images_diff = images_diff.cpu()
        images_list = [transforms.ToPILImage()(img.cpu()) for img in images]
        images_diff_list = [transforms.ToPILImage()(img.cpu()) for img in images_diff]
        genes = genes.cpu()
        non_zero_counts = (genes != 0).sum(dim=1, keepdim=True)
        print(non_zero_counts)
        genes_array = genes.numpy()
        df = pd.DataFrame(genes_array)
        image_large = Image.new('RGB', (BATCHSIZE * 64, 2 * 64))

        xoffset = 0
        yoffset = 0
        for img in images_list:
            image_large.paste(img, (xoffset, yoffset))
            xoffset += 64
        yoffset += 64
        xoffset = 0
        for img in images_diff_list:
            image_large.paste(img, (xoffset, yoffset))
            xoffset += 64
        break


