"""MNIST 数据加载：统一的 transform、数据集/DataLoader 构建、GPU 预加载。"""

import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from common import ROOT

# 统一归一化到 [-1, 1]（与所有模型的 tanh 输出对应）
MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def get_mnist(train: bool = True, download: bool = True) -> datasets.MNIST:
    """加载 MNIST 数据集（使用统一 transform）。"""
    return datasets.MNIST(
        root=ROOT / "data", train=train, download=download, transform=MNIST_TRANSFORM
    )


def get_train_loader(batch_size: int = 64) -> DataLoader:
    return DataLoader(get_mnist(train=True), batch_size=batch_size, shuffle=True)


def get_test_loader(batch_size: int = 256) -> DataLoader:
    return DataLoader(get_mnist(train=False), batch_size=batch_size, shuffle=False)


def get_subsampled_mnist(size: int = 10000, seed: int = 42, train: bool = True) -> Subset:
    """用固定 seed 从 MNIST 中确定性地子采样 size 张图。"""
    full_dataset = get_mnist(train=train, download=False)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(full_dataset), size=size, replace=False)
    return Subset(full_dataset, indices)


class GPUDataLoader:
    """把整个数据集预加载到 GPU/MPS 上的快速 DataLoader。

    消除训练时 CPU→GPU 拷贝和 dataloader worker 的开销。
    """

    def __init__(self, dataset, batch_size=64, shuffle=True, device=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device if device is not None else torch.device("cpu")

        print(f"[GPU Preloader]: Preloading dataset of size {len(dataset)} onto {self.device}...")
        start_time = time.time()

        # 用一个临时的标准 DataLoader 一次性读入全部数据
        temp_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
        all_images, all_labels = next(iter(temp_loader))

        self.images = all_images.to(self.device)
        self.labels = all_labels.to(self.device)

        self.num_samples = len(dataset)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        print(f"[GPU Preloader]: Preloaded successfully in {time.time() - start_time:.2f} seconds.")

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.num_samples, device=self.device)
        else:
            indices = torch.arange(self.num_samples, device=self.device)

        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            yield self.images[batch_indices], self.labels[batch_indices]

    def __len__(self):
        return self.num_batches
