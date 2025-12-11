# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import numpy as np
import random
from dynamic_vit_viz import vit_register_dynamic_viz
import utils
from vit_tome import apply_patch
from cifar_train import train_model

from custom_summary import custom_summary

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

# Initialize the model
model = vit_register_dynamic_viz(img_size=224, patch_size=16, in_chans=3, num_classes=10, embed_dim=384, depth=12,
                                 num_heads=6, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
                                 drop_path_rate=0., init_scale=1e-4,
                                 mlp_ratio_clstk=4.0, num_register_tokens=4, cls_pos=6, reg_pos=0)

input_size = (3, 224, 224)
# custom_summary(model, input_size)

runs = 50
batch_size = 64  # Lower this if you don't have that much memory



# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Baseline benchmark
baseline_throughput = utils.benchmark(
    model,
    device=device,
    verbose=True,
    runs=runs,
    batch_size=batch_size,
    input_size=input_size
)
apply_patch(model)

model.r = 16
tome_throughput = utils.benchmark(
    model,
    device=device,
    verbose=True,
    runs=runs,
    batch_size=batch_size,
    input_size=input_size
)
print(f"Throughput improvement: {tome_throughput / baseline_throughput:.2f}x")

