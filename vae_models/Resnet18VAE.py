import torch
import torch.nn as nn
from functools import reduce
from operator import mul
from typing import Optional, Tuple
from torch import Tensor
from torchvision import  models
encoder_resnet = models.resnet18(pretrained=True)
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Unflatten(nn.Module):
    def __init__(self, channel, height, width):
        super(Unflatten, self).__init__()
        self.channel = channel
        self.height = height
        self.width = width

    def forward(self, input):
        return input.view(input.size(0), self.channel, self.height, self.width)


class Resnet18VAE(nn.Module):
    def __init__(self, latent_size: int):
        super(Resnet18VAE, self).__init__()
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
           encoder_resnet
        )
        Module_1 = nn.ReLU
        # hidden => mu
        self.fc1 = nn.Linear(1000, self.latent_size)

        # hidden => logvar
        self.fc2 = nn.Linear(1000, self.latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 6272),
            nn.ReLU(),
            Unflatten(128, 7, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        return mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def reparameterize(self, mu: Optional[Tensor] = None, logvar: Optional[Tensor] = None) -> Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def reparameterize_eval(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
# x = Resnet18VAE(3)
# # print(x.eval())
#
#
#
#
#
# submodule_dict = dict(x.named_modules())
# # print(submodule_dict.keys())
# conv = []
# for i in submodule_dict:
#     if 'conv' in i:
#         conv.append(i)
# print(conv)





# print(x.named_modules())
# for idx, m in enumerate(x.named_modules()):
#         print(idx, '->', m)
#         print(m[0],m[1])
#         print("_______________++++++++++++++++_________________________")