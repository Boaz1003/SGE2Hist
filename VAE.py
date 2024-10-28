import torch.nn as nn
import torch.nn.functional as F
import torch


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(Encoder, self).__init__()
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i == 0:
                self.hidden_layers.append(nn.Linear(input_dim, hidden_dims[0]))
            else:
                self.hidden_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        self.fc_mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)
            # print(x)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super(Decoder, self).__init__()
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i == 0:
                self.hidden_layers.append(nn.Linear(latent_dim, hidden_dims[0]))
            else:
                self.hidden_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        self.fc_output = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, z):
        for layer in self.hidden_layers:
            z = layer(z)
            z = F.relu(z)
        reconstruction = self.fc_output(z)
        return reconstruction


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        decoder_hidden_dims = hidden_dims[::-1]
        self.decoder = Decoder(latent_dim, decoder_hidden_dims, input_dim)  # 输入和输出维度相同

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decoder(z)
        return reconstruction
