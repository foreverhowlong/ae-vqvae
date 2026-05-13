import torch
import torch.nn as nn
import torch.nn.functional as F



class Encoder(nn.Module):
    def __init__(self,latent_dim):
        super(Encoder, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 输入1通道，输出32通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 输入32通道，输出64通道
        # 定义全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 展平后输入到全连接层
        self.fc_2     = nn.Linear(128, latent_dim)



    def forward(self, x):
        x = F.relu(self.conv1(x))  # 第一层卷积 + ReLU
        x = F.max_pool2d(x, 2)     # 最大池化
        x = F.relu(self.conv2(x))  # 第二层卷积 + ReLU
        x = F.max_pool2d(x, 2)     # 最大池化
        x = x.view(-1, 64 * 7 * 7) # 展平
        x = F.relu(self.fc1(x))    # 全连接层 + ReLU
        x = self.fc_2(x)            # 最后一层输出
        return x
    
class CodeBook(nn.Module):
    def __init__(self,latent_dim,codebook_K):
        super().__init__()
        self.codebook = nn.Embedding(codebook_K,latent_dim)
        
    def forward(self,z_e):
        # z_e: (B, D)  codebook.weight: (K, D)
        #因为cdist一定要按照batch算距离，所以这里要先扩展成三个维度，再收缩回两个维度
        distances = torch.cdist(z_e.unsqueeze(0), self.codebook.weight.unsqueeze(0)).squeeze(0)
        # distances: (B, K)
        indices = distances.argmin(dim=-1)  # (B,)
        z_q_raw = self.codebook(indices)  # (B, D) — 量化后的向量
        z_q_st = z_e + (z_q_raw - z_e).detach()
        return z_q_raw, z_q_st, indices 
        
        
        
class Decoder(nn.Module):
    def __init__(self,latent_dim):
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
    
class VQVAE(nn.Module):
    def __init__(self,latent_dim=2,codebook_K=256):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.codebook = CodeBook(latent_dim,codebook_K)
        self.decoder = Decoder(latent_dim)
        
    def forward(self, x):
        z_e= self.encoder(x)
        z_q_raw,z_q_st,indices=self.codebook(z_e)
        x_recon = self.decoder(z_q_st)
        return z_e, z_q_raw,z_q_st, x_recon, indices