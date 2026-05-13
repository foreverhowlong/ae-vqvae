# VAE implementation — to be written later

import torch
import torch.nn as nn
import torch.nn.functional as F

latent_dim = 2

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 输入1通道，输出32通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 输入32通道，输出64通道
        # 定义全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 展平后输入到全连接层
        self.fc_mu     = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 第一层卷积 + ReLU
        x = F.max_pool2d(x, 2)     # 最大池化
        x = F.relu(self.conv2(x))  # 第二层卷积 + ReLU
        x = F.max_pool2d(x, 2)     # 最大池化
        x = x.view(-1, 64 * 7 * 7) # 展平
        x = F.relu(self.fc1(x))    # 全连接层 + ReLU
        mu = self.fc_mu(x)            # 最后一层输出
        logvar = self.fc_logvar(x)
        return mu,logvar
    
class Reparam(nn.Module):
    def init(self):
        super().init()
        
    def forward(self,mu,logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)   # 从 N(0,I) 采样
        return mu + eps * std
        
        
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.invconv1 = nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)  # 输入32通道，输出1通道,长宽翻倍
        self.invconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 输入64通道，输出128通道,长宽翻倍
        # 定义全连接层
        self.invfc1 = nn.Linear(128, 64 * 7 * 7)  # 
        self.invfc2 = nn.Linear(latent_dim,128)  # 32个latent dimensions
        
    def forward(self, x):
        x = F.relu(self.invfc2(x))
        x = F.relu(self.invfc1(x))
        x = x.view(-1, 64 , 7 , 7)
        x = F.relu(self.invconv2(x))
        x = self.invconv1(x)
        x = torch.tanh(x)
        return x
    
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.reparam = Reparam()
        self.decoder = Decoder()
        
    def forward(self, x):
        mu,logvar = self.encoder(x)
        x = self.reparam(mu,logvar)
        x = self.decoder(x)
        return x,mu,logvar