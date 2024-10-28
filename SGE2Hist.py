import math
import sys
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps

def zero_tensor_with_probability(Z, p):
    if np.random.rand() < p:
        return Z * 0.
    else:
        return Z
def invert_transform(data):
    # 1. 逆归一化：将数据缩放回 [0, 255]
    data = data * 255.0

    # 3. 重塑为原始图像尺寸：将数据重塑为 28x28 的图像
    data = data.view(data.size(0), 1, 28, 28)

    return data
class SGE2Hist(nn.Module):
    def __init__(self, nn_model, vae, betas, n_T, device, drop_prob=0.1, nClusters=10, latent_dim=10, Cluster_dim=16):
        super(SGE2Hist, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()
        self.pi = Parameter(torch.zeros(nClusters))
        self.mu = Parameter(torch.randn(nClusters, Cluster_dim))
        self.logvar = Parameter(torch.randn(nClusters, Cluster_dim))
        self.vae = vae.to(device)
        self.nClusters = nClusters
        self.latent_dim = latent_dim
        self.Cluster_dim = Cluster_dim


    def weights(self):
        return torch.softmax(self.pi, dim=0)

    def sample_batch_from_clusters(self, cluster_ids, mu, sigma):
        batch_size = cluster_ids.shape[0]  # 获取批次大小
        num_features = len(mu[0])  # 特征维度
        batch_samples = torch.zeros((batch_size, num_features))  # 存储批次样本的张量

        # 逐个采样每个样本
        for i in range(batch_size):
            sample_cluster_id = cluster_ids[i]  # 获取当前样本对应的聚类标识符
            # 从指定聚类的参数中采样一个样本
            sample = reparameterize(mu[sample_cluster_id], sigma[sample_cluster_id])
            batch_samples[i] = sample

        return batch_samples

    def gaussian_pdfs_log(self, x, mus, log_sigma2s):
        G = []
        for c in range(self.nClusters):
            G.append(self.gaussian_pdf_log(x, mus[c:c + 1, :], log_sigma2s[c:c + 1, :]).view(-1, 1))
        return torch.cat(G, 1)

    @staticmethod
    def gaussian_pdf_log(x, mu, log_sigma2):
        return -0.5 * torch.sum(
            torch.log(torch.tensor(2 * math.pi)) + log_sigma2 + (x - mu).pow(2) / torch.exp(log_sigma2), dim=1)

    def Loss(self, images, genes):
        beta_z2 = 1
        beta_z1z2 = 1
        det = 1e-10
        z_mu, z_logvar = self.vae.encoder(genes)
        z_dim = self.latent_dim
        z1_mu, z2_mu = torch.split(z_mu, [self.Cluster_dim, z_dim - self.Cluster_dim], dim=1)
        z1_logvar, z2_logvar = torch.split(z_logvar, [self.Cluster_dim, z_dim - self.Cluster_dim], dim=1)
        z1 = reparameterize(z1_mu, z1_logvar)
        z2 = reparameterize(z2_mu, z2_logvar)

        if torch.isnan(z1).any():
            print("z1 is not a number")
            print(z1)
            sys.exit()
        if torch.isnan(z2).any():
            print("z2 is not a number")
            sys.exit()
        z = torch.cat((z1, z2), dim=1)
        genes_rec = self.vae.decoder(z)

        LossFucntion = nn.MSELoss()
        L_rec = LossFucntion(genes_rec, genes)

        _ts = torch.randint(1, self.n_T + 1, (images.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(images)  # eps ~ N(0, 1)
        x_t = (
                self.sqrtab[_ts, None, None, None] * images
                + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.
        z_dropped = zero_tensor_with_probability(z, 0.8)
        L_diff = self.loss_mse(noise, self.nn_model(x_t, z_dropped, _ts / self.n_T))

        gamma_c = torch.exp(torch.log(self.pi.unsqueeze(0)) + self.gaussian_pdfs_log(z1, self.mu, self.logvar)) + det

        gamma_c = gamma_c / (gamma_c.sum(1).view(-1, 1))
        # gamma_c = gamma_c / (gamma_c.sum(1).view(-1, 1))  # batch_size*Clusters
        # print(gamma_c.sum(1))

        L_GMM = 0.5 * torch.mean(torch.sum(gamma_c * torch.sum(self.logvar.unsqueeze(0) +
                                                               torch.exp(z1_logvar.unsqueeze(1) - self.logvar.unsqueeze(0)) +
                                                               (z1_mu.unsqueeze(1) - self.mu.unsqueeze(0)).pow(
                                                                   2) / torch.exp(self.logvar.unsqueeze(0)), 2), 1))

        L_GMM -= torch.mean(torch.sum(gamma_c * torch.log(self.pi.unsqueeze(0) / (gamma_c)), 1)) + 0.5 * torch.mean(
            torch.sum(1 + z1_logvar, 1))

        #z1,z2 disentangle
        mu1 = torch.sum(z1_mu, dim=1, keepdim=True)
        mu2 = torch.sum(z2_mu, dim=1, keepdim=True)
        mu_z1z2 = torch.cat((mu1, mu2), dim=1)

        logvar1 = torch.sum(z1_logvar, dim=1, keepdim=True)
        logvar2 = torch.sum(z2_logvar, dim=1, keepdim=True)
        logvar_z1z2 = torch.cat((logvar1, logvar2), dim=1)
        KLD_element_z1z2 = mu_z1z2.pow(2).add_(logvar_z1z2.exp()).mul_(-1).add_(1).add_(logvar_z1z2)
        KLD_z1z2 = torch.mean(KLD_element_z1z2).mul_(-0.5)

        KLD_element_z2 = z2_mu.pow(2).add_(z2_logvar.exp()).mul_(-1).add_(1).add_(z2_logvar)
        KLD_z2 = torch.sum(KLD_element_z2,).mul_(-0.5)

        return L_rec * 1 + L_GMM * 1 + L_diff * 10 + KLD_z2 * 1 + beta_z1z2 * 0.01

    def pretrain_loss(self, genes):
        z_mu, z_logvar = self.vae.encoder(genes)
        # z = reparameterize(z_mu, z_logvar)
        # print(f"z: {z_mu}")
        # print(f"z: {z}")
        genes_rec = self.vae.decoder(z_mu)
        # print(f"tensor_rec: {tensor_rec}")
        # assert torch.all(z_mu >= 0.0) and torch.all(z_mu <= 1.0), "x values are not in [0, 1] range."
        # assert torch.all(tensor_rec >= 0.0) and torch.all(tensor_rec <= 1.0), "x_rec values are not in [0, 1] range."
        # L_rec = F.binary_cross_entropy(x_rec, x, reduction='sum')
        LossFucntion = nn.MSELoss()
        L_rec = LossFucntion(genes_rec, genes)

