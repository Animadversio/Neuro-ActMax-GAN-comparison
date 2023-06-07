import torch
from tqdm.autonotebook import trange, tqdm
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR


def GAN_invert(G, target_img, z_init=None, lr=2e-3, weight_decay=0e-4, max_iter=5000, print_progress=True):
    if z_init is None:
        z_opt = torch.randn(5, 4096, requires_grad=True, device="cuda")
    else:
        z_opt = z_init.clone().detach().requires_grad_(True).to("cuda")
    if target_img.device != "cuda":
        target_img = target_img.cuda()
    opt = Adam([z_opt], lr=lr, weight_decay=weight_decay)
    pbar = trange(max_iter)
    for i in pbar:
        img_opt = G.visualize(z_opt)
        losses = ((img_opt - target_img) ** 2).mean(dim=(1, 2, 3))  # changed from mean to sum
        loss = losses.sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        pbar.set_description(f"loss: {losses.mean().item():.3f}")
        if print_progress:
            print(i, losses.mean().item())
    img_opt = G.visualize(z_opt.detach())
    return z_opt, img_opt, losses.detach().cpu()


def GAN_invert_with_scheduler(G, target_img, z_init=None, scheduler=None, lr=1e-2, weight_decay=0e-4, max_iter=5000, print_progress=True):
    if z_init is None:
        z_opt = torch.randn(5, 4096, requires_grad=True, device="cuda")
    else:
        z_opt = z_init.clone().detach().requires_grad_(True).to("cuda")
    if target_img.device != "cuda":
        target_img = target_img.cuda()
    opt = Adam([z_opt], lr=lr, weight_decay=weight_decay)
    if scheduler is None:
        scheduler = ExponentialLR(opt, gamma=0.999)
    pbar = trange(max_iter)
    for i in pbar:
        img_opt = G.visualize(z_opt)
        loss = ((img_opt - target_img) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        scheduler.step()
        pbar.set_description(f"loss: {loss.item():.3f}, lr: {scheduler.get_last_lr()[0]:.3e}")
        if print_progress:
            print(i, loss.item(), "lr", scheduler.get_last_lr()[0])
    img_opt = G.visualize(z_opt.detach())
    return z_opt, img_opt
