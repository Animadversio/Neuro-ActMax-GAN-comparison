import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm, trange
import sys
sys.path.append("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/biggan-pytorch-ecoset/code")
# from BigGANdeep_nodist import Generator
from BigGAN_nodist import Generator


class Distribution(torch.Tensor):
  # Init the params of the distribution
  def init_distribution(self, dist_type, **kwargs):
    # cwd = os.getcwd()
    filepath = os.path.join("/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/biggan-pytorch-ecoset/code", 'ecoset_class_counts.pt')
    class_weights = torch.load(filepath).tolist()
    self.dist_type = dist_type
    self.dist_kwargs = kwargs
    if self.dist_type == 'normal':
      self.mean, self.var = kwargs['mean'], kwargs['var']
    elif self.dist_type == 'categorical':
      self.num_categories = kwargs['num_categories']
      self.sampler = WeightedRandomSampler(class_weights, len(self), replacement=True)

  def sample_(self):
    if self.dist_type == 'normal':
      self.normal_(self.mean, self.var)
    elif self.dist_type == 'categorical':
      self.random_(0, self.num_categories)
      self[:] = torch.tensor(list(self.sampler), dtype=torch.int)
    # return self.variable

  # Silly hack: overwrite the to() method to wrap the new object
  # in a distribution as well
  def to(self, *args, **kwargs):
    new_obj = Distribution(self)
    new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
    new_obj.data = super().to(*args, **kwargs)
    return new_obj


def prepare_z_y(G_batch_size, dim_z, nclasses, device='cuda',
                fp16=False,z_var=1.0):
  z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
  z_.init_distribution('normal', mean=0, var=z_var)
  z_ = z_.to(device,torch.float16 if fp16 else torch.float32)

  if fp16:
    z_ = z_.half()

  y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
  y_.init_distribution('categorical',num_categories=nclasses)
  y_ = y_.to(device, torch.int64)
  return z_, y_

from collections import defaultdict
suffix = "best2"
batch_size = 100
BGEco_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/biggan-pytorch-ecoset"

device = "cuda"
config = torch.load(join(BGEco_root, "weights", f"state_dict_{suffix}.pth"))['config']
weights_dict = torch.load(join(BGEco_root, "weights", f"G_{suffix}.pth"))
G = Generator(**config)
G.load_state_dict(weights_dict, strict=True)
G.to(device).eval()
G.requires_grad_(False);

batch_num = 50000 // batch_size
global_cnt = 0
from collections import defaultdict

# Initialize counter - defaultdict(int) automatically starts at 0 for new keys
class_counters = defaultdict(int)

for batch_idx in trange(batch_num):
    z_, y_ = prepare_z_y(batch_size, G.dim_z, config['n_classes'],
                        device=device, fp16=config['G_fp16'], 
                        z_var=config['z_var'])
    z_.sample_()
    y_.sample_()
    with torch.no_grad():
        imgs = G.forward(z_, G.shared(y_)).cpu()

    imgs = torch.tensor((imgs + 1) / 2)  # denormalize
    # save the images one by one
    for i in range(batch_size):
        cls_idx = int(y_[i])
        save_image(imgs[i], join(BGEco_root, "samples_50k", 
                 f"img_cls{cls_idx:03d}_{class_counters[cls_idx]:05d}.jpg"))
        class_counters[cls_idx] += 1
        global_cnt += 1

if False:
    grid = make_grid(imgs, nrow=5)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.show()