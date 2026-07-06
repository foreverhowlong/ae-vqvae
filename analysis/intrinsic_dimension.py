"""使用 MLE 和 TwoNN 估计 MNIST 的本征维度，互相验证。"""

# ── Monkey-patch: 修复 skdim MLE 在 Python 3.14 上的兼容性问题 ──
# skdim 0.3.4 的 MLE.__init__ 中使用了 inspect.getargvalues().pop("self")，
# 但 Python 3.14 的 FrameLocalsProxy 不支持 .pop()。
# 这里在 skdim 导入前替换 inspect.getargvalues 来规避。
import inspect as _inspect

_original_getargvalues = _inspect.getargvalues


def _patched_getargvalues(frame):
    args, varargs, varkw, locals_dict = _original_getargvalues(frame)
    # 将 FrameLocalsProxy 转为普通 dict，使其支持 .pop()
    return args, varargs, varkw, dict(locals_dict)


_inspect.getargvalues = _patched_getargvalues

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from skdim.id import MLE, TwoNN

from common import ROOT, get_device


def main():
    device = get_device()

    # ── 加载 MNIST 测试集 ──────────────────────────
    # 只做 ToTensor()，保留 [0,1] 原始像素值
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(
        root=ROOT / "data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # ── 展平为 784 维向量 ──────────────────────────
    all_images = []
    for images, _ in test_loader:
        images = images.to(device)
        all_images.append(images.view(images.size(0), -1).cpu())  # (B, 784)
    all_images = torch.cat(all_images, dim=0).numpy()  # (10000, 784)

    print(f"MNIST 测试集形状: {all_images.shape}")  # (10000, 784)

    # ── 随机子采样 5000 张加速 ─────────────────────
    rng = np.random.default_rng(42)
    idx = rng.choice(all_images.shape[0], size=5000, replace=False)
    X = all_images[idx]
    print(f"子采样形状: {X.shape}")

    # ── MLE 估计 ───────────────────────────────────
    print("\n=== MLE (Maximum Likelihood Estimation) ===")
    mle = MLE()
    d_mle = mle.fit_transform(X)
    print(f"MLE 估计的本征维度: {d_mle:.4f}")

    # ── TwoNN 估计 ─────────────────────────────────
    print("\n=== TwoNN (Two Nearest Neighbors) ===")
    twonn = TwoNN()
    d_twonn = twonn.fit_transform(X)
    print(f"TwoNN 估计的本征维度: {d_twonn:.4f}")

    # ── 对比 ───────────────────────────────────────
    print("\n" + "=" * 40)
    print(f"  MLE  : {d_mle:.4f}")
    print(f"  TwoNN: {d_twonn:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
