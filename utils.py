
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
import numpy as np
from main_decode import AIEM

class SELayer(torch.nn.Module):
    def __init__(self, num_filter):
        super(SELayer, self).__init__()
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv_double = torch.nn.Sequential(
            nn.Conv2d(num_filter, num_filter // 16, 1, 1, 0, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(num_filter // 16, num_filter, 1, 1, 0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        mask = self.global_pool(x)
        mask = self.conv_double(mask)
        x = x * mask
        return x


class ResBlock(nn.Module):
    def __init__(self, num_filter):
        super(ResBlock, self).__init__()
        body = []
        for i in range(2):
            body.append(nn.ReflectionPad2d(1))
            body.append(nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=0))
            if i == 0:
                body.append(nn.LeakyReLU())
        body.append(SELayer(num_filter))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        x = res + x
        return x


class Up(nn.Module):
    def __init__(self):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=0),
            nn.LeakyReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=0),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Compute_z(nn.Module):
    def __init__(self, latent_dim):
        super(Compute_z, self).__init__()
        self.latent_dim = latent_dim
        self.u_conv_layer = nn.Conv2d(128, 2 * self.latent_dim, kernel_size=1, padding=0)
        self.s_conv_layer = nn.Conv2d(128, 2 * self.latent_dim, kernel_size=1, padding=0)

    def forward(self, x):
        u_encoding = torch.mean(x, dim=2, keepdim=True)
        u_encoding = torch.mean(u_encoding, dim=3, keepdim=True)
        u_mu_log_sigma = self.u_conv_layer(u_encoding)
        u_mu_log_sigma = torch.squeeze(u_mu_log_sigma, dim=2)
        u_mu_log_sigma = torch.squeeze(u_mu_log_sigma, dim=2)
        u_mu = u_mu_log_sigma[:, :self.latent_dim]
        u_log_sigma = u_mu_log_sigma[:, self.latent_dim:]
        u_dist = Independent(Normal(loc=u_mu, scale=torch.exp(u_log_sigma)), 1) 

        s_encoding = torch.std(x, dim=2, keepdim=True)
        s_encoding = torch.std(s_encoding, dim=3, keepdim=True)
        s_mu_log_sigma = self.s_conv_layer(s_encoding)
        s_mu_log_sigma = torch.squeeze(s_mu_log_sigma, dim=2)
        s_mu_log_sigma = torch.squeeze(s_mu_log_sigma, dim=2)
        s_mu = s_mu_log_sigma[:, :self.latent_dim]
        s_log_sigma = s_mu_log_sigma[:, self.latent_dim:]
        s_dist = Independent(Normal(loc=s_mu, scale=torch.exp(s_log_sigma)), 1) 
        return u_dist, s_dist, u_mu, s_mu, torch.exp(u_log_sigma), torch.exp(s_log_sigma)
#模仿Ucolor
class Channel_Attention_Block(nn.Module):
    def __init__(self, input_dim):
        super(Channel_Attention_Block, self).__init__()
        self.avg=nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Conv2d(input_dim, input_dim//16,kernel_size=1)
        self.relu=nn.ReLU()
        self.fc2=nn.Conv2d(input_dim//16, input_dim,kernel_size=1)
        self.sig=nn.Sigmoid()

    def forward(self, x):
       #print(x.size())
        x1=self.avg(x)
        #print(x1.size())
        x1=self.fc1(x1)
        #print(x1.size())
        x1=self.relu(x1)
        #print(x1.size())
        x1=self.fc2(x1)
        #print(x1.size())
        x1=self.sig(x1)
        #print(x1.size())
        x1=x*x1
        out=x+x1
        return out


class EnhanceLayers(nn.Module):
    def __init__(self, embed_dim=64, split_group=4, n_curve=3):
        super().__init__()
        self.blks = nn.ModuleList()
        for i in range(12):
            layer = AIEM(embed_dim, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., split_group=split_group,
                         n_curve=n_curve)
            self.blks.append(layer)

    def forward(self, x):
        for m in self.blks:
            x = m(x)
        return x
