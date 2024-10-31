from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from model.VAE import VAE
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch
from Dataset import DataSet
from model.Unet import UNetModel
from torch.utils.data import DataLoader
from model.SGE2Hist import SGE2Hist


BATCHSIZE = 128
PRETRAIN_EPOCHS = 100
NUM_CLASSES = 6
N_FEAT = 256
N_T = 1000
LR = 1e-4
Z_DIM = 128
DEVICE = "cuda:0"
SAVE_PATH = ''
save_model = True
LOG_DIR = ''
CLUSTER_DIM = 24

def cluster_acc(Y_pred, Y):
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1

    # 使用 linear_sum_assignment 替代旧的 linear_assignment
    ind_row, ind_col = linear_sum_assignment(w.max() - w)

    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / Y_pred.size, w

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = DataSet(datadir='',transform=None) #Data can get from ""
    train_dataset, test_dataset = dataset.split_dataset()
    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False)
    # print(train_dataset.__len__())
    vae = VAE(input_dim=19059, latent_dim=Z_DIM, hidden_dims=[2048, 512])
    # unet = ContextUnet(in_channels=3, n_feat=N_FEAT, z_dim=Z_DIM)
    unet = UNetModel(in_channels=3, model_channels=64, out_channels=3, num_res_blocks=3,
                     attention_resolutions=[128, 64, 32, 16, 8], channel_mult=(1, 2, 4, 8, 16), num_head_channels=32, cond_dim=Z_DIM)
    model = SGE2Hist(nn_model=unet, vae=vae, betas=(1e-4, 0.02), n_T=N_T, device=DEVICE,
                                       drop_prob=0.1, nClusters=NUM_CLASSES, latent_dim=Z_DIM,
                                       Cluster_dim=CLUSTER_DIM)
    model.to(DEVICE)

    writer = SummaryWriter(LOG_DIR)

    epoch_bar = tqdm(range(PRETRAIN_EPOCHS))
    opti = Adam(model.parameters(), lr=0.001)
    lr_s = StepLR(opti, step_size=10, gamma=0.95)
    for epoch in epoch_bar:
        model.train()
        L = 0
        for images, genes, types in train_loader:
            # images = images.to(DEVICE)
            genes = genes + 1e-6
            genes = genes.to(DEVICE)
            # assert torch.all(images >= 0.0) and torch.all(images <= 1.0), "Data values are not in [0, 1] range."
            loss = model.pretrain_loss(genes)
            opti.zero_grad()
            loss.backward()
            opti.step()
            epoch_bar.set_description("Loss: {:.4f}".format(loss.item()))
            L += loss.detach().cpu().numpy()
        lr_s.step()
        writer.add_scalar("Pretrain Loss", L/len(train_loader), epoch)

    torch.save(model.state_dict(), SAVE_PATH)