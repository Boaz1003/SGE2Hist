import sys
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from model.VAE import VAE
from model.SGE2Hist import SGE2Hist
from Dataset import DataSet
from model.Unet import UNetModel

def cluster_acc_old(Y_pred, Y):
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w


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

def train ():
    # hardcoding these here

    BATCHSIZE = 128
    EPOCHS = 600
    NUM_CLASSES = 6
    N_FEAT = 256
    N_T = 1000
    LR = 1e-4
    Z_DIM = 128
    DEVICE = "cuda:0"
    SAVE_PATH = ''
    PRETRAINED_PATH = ''
    LOG_DIR = ''
    CLUSTERDIM = 16
    save_model = True
    t_NSE = True

    vae = VAE(input_dim=19059, latent_dim=Z_DIM, hidden_dims=[2048, 512])
    # unet = ContextUnet(in_channels=3, n_feat=N_FEAT, z_dim=Z_DIM)
    unet = UNetModel(in_channels=3, model_channels=64, out_channels=3, num_res_blocks=3,
                     attention_resolutions=[64, 32, 16, 8], channel_mult=(1, 2, 4, 8), num_head_channels=32,
                     cond_dim=Z_DIM)
    model = SGE2Hist(nn_model=unet, vae=vae, betas=(1e-4, 0.02), n_T=N_T, device=DEVICE,
                                      drop_prob=0.1, nClusters=NUM_CLASSES, latent_dim=Z_DIM, Cluster_dim=CLUSTERDIM)

    model.load_state_dict(torch.load(PRETRAINED_PATH))
    model.to(DEVICE)

    transform = transforms.Compose([
        transforms.Resize((64, 64))
    ])

    dataset = DataSet(datadir='', transform=transform)
    # data canbe got from "https://pan.baidu.com/s/1gZsRJmA4ChITDF-1_bAAFQ?pwd=1oob" for Mouse Brain and "https://pan.baidu.com/s/13ippRbnTgow9eVtZcTwZNg?pwd=r233" for Mouse Embyro
    train_dataset, test_dataset = dataset.split_dataset()
    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, pin_memory=True)

    optim = torch.optim.Adam(model.parameters(), lr=LR)
    lr_s = StepLR(optim, step_size=10, gamma=0.95)

    writer = SummaryWriter(LOG_DIR)

    for ep in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader)
        Loss = 0
        for images, genes, types in pbar:
            optim.zero_grad()
            images = images.to(DEVICE)
            genes = genes + 1e-6
            genes = genes.to(DEVICE)
            loss, l_rec, l_gmm, l_diff, kld_z2, kld_z1z2 = model.ELBO_Loss_disentangled(images, genes)
            if torch.isnan(l_rec).any():
                print(f"l_rec is not a number{l_rec}")
                sys.exit()
            if torch.isnan(l_gmm).any():
                print(f"l_gmm is not a number{l_gmm}")
                sys.exit()
            if torch.isnan(l_diff).any():
                print(f"l_diff is not a number{l_diff}")
                sys.exit()

            optim.zero_grad()
            loss.backward()
            pbar.set_description(f"Epoch: {ep + 1}, loss: {loss.item():.4f}")
            optim.step()
            Loss += loss.detach().cpu().numpy()
        lr_s.step()
        writer.add_scalar("Loss", Loss/len(train_loader), ep)

        model.eval()

        # calculate ACC
        if ep % 2 == 0 or ep == 0:
            tru = []
            pre = []
            with torch.no_grad():
                for _, x, y in train_loader:
                    x = x.to(DEVICE)
                    tru.append(y.numpy())
                    pre.append(model.predict_disentangled(x))
            tru = np.concatenate(tru, 0)
            pre = np.concatenate(pre, 0)

            acc = cluster_acc(pre,tru)[0] * 100
        # optionally save model
        if save_model and ep % 40 == 0:
            torch.save(model.state_dict(), SAVE_PATH + f"model_{ep + 1}.pth")
            print('saved model at ' + SAVE_PATH + f"model_with_encoder{ep + 1}.pth")

    writer.close()


if __name__ == '__main__':
    train()
