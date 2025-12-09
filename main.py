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

from custom_summary import custom_summary
from tqdm import tqdm
import utils
import vit_tome

# Initialize the model
model = vit_register_dynamic_viz(img_size=224, patch_size=16, in_chans=3, num_classes=10, embed_dim=384, depth=12,
                                 num_heads=6, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
                                 drop_path_rate=0., init_scale=1e-4,
                                 mlp_ratio_clstk=4.0, num_register_tokens=4, cls_pos=6, reg_pos=0)

custom_summary(model, (3, 224, 224))

best_model_path = 'best_model.pth'
model.load_state_dict(torch.load(best_model_path))

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Load the best model for evaluation
runs = 50
batch_size = 64  # Lower this if you don't have that much memory
input_size = model.default_cfg["input_size"]



# Baseline benchmark
baseline_throughput = utils.benchmark(
    model,
    device=device,
    verbose=True,
    runs=runs,
    batch_size=batch_size,
    input_size=input_size
)

vit_tome.apply_patch(model)
# ToMe with r=16
model.r = 17
tome_throughput = utils.benchmark(
    model,
    device=device,
    verbose=True,
    runs=runs,
    batch_size=batch_size,
    input_size=input_size
)
print(f"Throughput improvement: {tome_throughput / baseline_throughput:.2f}x")
