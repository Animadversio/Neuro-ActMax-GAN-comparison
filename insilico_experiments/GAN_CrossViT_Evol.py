import torch
import torch.nn as nn
from timm import create_model
from core.utils import show_imgrid
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from core.utils import upconvGAN, featureFetcher, featureFetcher_module, CholeskyCMAES

# model = create_model(configs.TRAIN.arch,pretrained = configs.TRAIN.pretrained_arch)
model = create_model('crossvit_18_dagger_408', pretrained=True)
model.cuda()
#%%
G = upconvGAN("fc6")
G.cuda().eval()

#%%
# model.patch_embed[1].patch_size
# model.patch_embed[0].patch_size


fetcher = featureFetcher_module()
fetcher.record_module(model.blocks[1].revert_projs[0], "B1_revproj0")
# fetcher.record_module(model.blocks[1].revert_projs[1], "B1_revproj1")
optim = CholeskyCMAES(4096, init_code=np.random.randn(1, 4096), Aupdate_freq=100,)
codes = optim.get_init_pop()
for i in range(100):
    # codes = np.random.randn(40, 4096)
    with torch.no_grad():
        imgs = G.visualize(torch.tensor(codes).float().cuda())
        model(imgs)
    scores = fetcher["B1_revproj0"][:, 0, 3]
    codes_new = optim.step_simple(scores, codes)
    codes = codes_new
    print(f"scores {scores.mean():.3f} {scores.std():.3f}")

fetcher.cleanup()
show_imgrid(imgs, )
#%%
for param in model.parameters():
    param.requires_grad = False
G.requires_grad_(False)
#%%
fetcher = featureFetcher_module()
fetcher.record_module(model.blocks[1].revert_projs[0], "B1_revproj0", ingraph=True)
z = torch.randn(1, 4096).cuda()
z.requires_grad = True
optim_grad = Adam([z], lr=0.1)
for i in range(50):
    model(G.visualize(z))
    loss = -fetcher["B1_revproj0"][:, 0, 3].mean()
    loss.backward()
    optim_grad.step()
    optim_grad.zero_grad()
    print(f"loss {-loss.item():.3f}")
fetcher.cleanup()
show_imgrid(G.visualize(z), )
#%%
def grad_evol_scorer(get_score_fun, steps=50, lr=0.1, return_zs=False):
    z = torch.randn(1, 4096).cuda()
    z.requires_grad = True
    optim_grad = Adam([z], lr=lr)
    score_traj = []
    zs = []
    for i in range(steps):
        zs.append(z.detach().cpu().clone())
        model(G.visualize(z))
        loss = -get_score_fun() # fetcher["B1_revproj0"][:, 0, 3].mean()
        loss.backward()
        optim_grad.step()
        optim_grad.zero_grad()
        print(f"loss {-loss.item():.3f}")
        score_traj.append(-loss.item())
    zs = torch.cat(zs, dim=0)
    img = G.visualize(z)
    show_imgrid(img, )
    if return_zs:
        return zs, img, np.array(score_traj)
    else:
        return z, img, np.array(score_traj)
#%%
fetcher = featureFetcher_module()
# fetcher.record_module(model.blocks[1].revert_projs[0], "B1_revproj0", ingraph=True)
# get_score_fun = lambda: fetcher["B1_revproj0"][:, 0, 10].mean()

keystr = "B1_mlp_fc2_1_0"
module = model.blocks[1].blocks[1][1].mlp.fc2
ch_unit = 300, 10
fetcher.record_module(module, keystr, ingraph=True)
get_score_fun = lambda: fetcher[keystr][:, ch_unit[0], ch_unit[1]].mean()
z, img, score_traj = grad_evol_scorer(get_score_fun, steps=100)
fetcher.cleanup()
#%%
from core.utils import grad_RF_estimate, show_gradmap
grad_RF_estimate()
#%%
keystr = "B1_B1_0_norm1"
module = model.blocks[1].blocks[1][0].norm1
ch_unit = 200, 10
fetcher.record_module(module, keystr, ingraph=True)
get_score_fun = lambda: fetcher[keystr][:, ch_unit[0], ch_unit[1]].mean()
z, img, score_traj = grad_evol_scorer(get_score_fun, steps=100)
fetcher.cleanup()
# %%
keystr = "B1_B1_0_norm1"
module = model.blocks[1].blocks[1][0].norm1
ch_unit = 200, 10
fetcher.record_module(module, keystr, ingraph=True)
get_score_fun = lambda: fetcher[keystr][:, ch_unit[0], ch_unit[1]].mean()
get_score_all_fun = lambda: fetcher[keystr][:, ch_unit[0], ch_unit[1]]
zs, img, score_traj = grad_evol_scorer(get_score_fun, steps=100, lr=0.5, return_zs=True)
#%%
zs = zs.cpu().numpy()
#%%
z_peak = zs[-1]


#%%
def get_manifold(peak_vec, subspace_list=("RND",), interval=9, print_manifold=True):
    '''Generate examples on manifold and run'''
    score_sum = []
    Perturb_vec = []
    # T0 = time()
    sphere_norm = np.linalg.norm(peak_vec)
    code_length = np.size(peak_vec)
    unit_peak_vec = peak_vec / sphere_norm
    # figsum = plt.figure(figsize=[16.7, 4])
    for spi, subspace in enumerate(subspace_list):
        code_list = []
        if subspace == "RND":
            title = "Norm%dRND%dRND%d" % (sphere_norm, 0 + 1, 1 + 1)
            print("Generating images on PC1, Random vector1, Random vector2 sphere (rad = %d) " % sphere_norm)
            rand_vec2 = np.random.randn(2, code_length)
            rand_vec2 = rand_vec2 - (rand_vec2 @ unit_peak_vec.T) @ unit_peak_vec
            rand_vec2 = rand_vec2 / np.sqrt((rand_vec2 ** 2).sum(axis=1))[:, np.newaxis]
            rand_vec2[1, :] = rand_vec2[1, :] - (rand_vec2[1, :] @ rand_vec2[0, :].T) * rand_vec2[0, :]
            rand_vec2[1, :] = rand_vec2[1, :] / np.linalg.norm(rand_vec2[1, :])
            vectors = np.concatenate((unit_peak_vec, rand_vec2), axis=0)
            Perturb_vec.append(vectors)
            # img_list = []
            interv_n = int(90 / interval)
            for j in range(-interv_n, interv_n + 1):
                for k in range(-interv_n, interv_n + 1):
                    theta = interval * j / 180 * np.pi
                    phi = interval * k / 180 * np.pi
                    code_vec = np.array([[np.cos(theta) * np.cos(phi),
                                          np.sin(theta) * np.cos(phi),
                                          np.sin(phi)]]) @ vectors
                    code_vec = code_vec / np.sqrt((code_vec ** 2).sum()) * sphere_norm
                    code_list.append(code_vec)
                    # img = self.G.visualize(code_vec)
                    # img_list.append(img.copy())
        else:
            raise NotImplementedError
            # PCi, PCj = subspace
            # title = "Norm%dPC%dPC%d" % (sphere_norm, PCi + 1, PCj + 1)
            # print("Generating images on PC1, PC%d, PC%d sphere (rad = %d)" % (PCi + 1, PCj + 1, sphere_norm,))
            # # img_list = []
            # interv_n = int(90 / interval)
            # self.Perturb_vec.append(self.PC_vectors[[0, PCi, PCj], :])
            # for j in range(-interv_n, interv_n + 1):
            #     for k in range(-interv_n, interv_n + 1):
            #         theta = interval * j / 180 * np.pi
            #         phi = interval * k / 180 * np.pi
            #         code_vec = np.array([[np.cos(theta) * np.cos(phi),
            #                               np.sin(theta) * np.cos(phi),
            #                               np.sin(phi)]]) @ self.PC_vectors[[0, PCi, PCj], :]
            #         code_vec = code_vec / np.sqrt((code_vec ** 2).sum()) * self.sphere_norm
            #         code_list.append(code_vec)
                    # img = self.G.visualize(code_vec)
                    # img_list.append(img.copy())
                    # plt.imsave(os.path.join(newimg_dir, "norm_%d_PC2_%d_PC3_%d.jpg" % (
                    # self.sphere_norm, interval * j, interval * k)), img)

        # pad_img_list = resize_and_pad(img_list, self.imgsize, self.corner) # Show image as given size at given location
        # scores = self.CNNmodel.score(pad_img_list)
        # print("Latent vectors ready, rendering. (%.3f sec passed)" % (time() - T0))
        code_arr = np.concatenate(code_list, axis=0)
        return code_arr, Perturb_vec
        # scores = self.CNNmodel.score(code_arr)
        # img_tsr = self.render_tsr(code_arr)
        # pad_img_tsr = resize_and_pad_tsr(img_tsr, self.imgsize,
        #                                  self.corner)  # Show image as given size at given location
        # # scores = self.CNNmodel.score_tsr(pad_img_tsr)
        # img_arr = img_tsr.permute([0, 2, 3, 1])
        # print("Image and score ready! Figure printing (%.3f sec passed)" % (time() - T0))
        # fig = utils.visualize_img_list(img_list, scores=scores, ncol=2*interv_n+1, nrow=2*interv_n+1, )
        # subsample images for better visualization
    #     msk, idx_lin = subsample_mask(factor=2, orig_size=(21, 21))
    #     img_subsp_list = [img_arr[i] for i in range(len(img_arr)) if i in idx_lin]
    #     if print_manifold:
    #         fig = visualize_img_list(img_subsp_list, scores=scores[idx_lin], ncol=interv_n + 1, nrow=interv_n + 1, )
    #         fig.savefig(join(self.savedir, "%s_%s.png" % (title, self.explabel)))
    #         plt.close(fig)
    #     scores = np.array(scores).reshape((2 * interv_n + 1, 2 * interv_n + 1))  # Reshape score as heatmap.
    #     self.score_sum.append(scores)
    #     ax = figsum.add_subplot(1, len(subspace_list), spi + 1)
    #     im = ax.imshow(scores)
    #     plt.colorbar(im, ax=ax)
    #     ax.set_xticks([0, interv_n / 2, interv_n, 1.5 * interv_n, 2 * interv_n]);
    #     ax.set_xticklabels([-90, 45, 0, 45, 90])
    #     ax.set_yticks([0, interv_n / 2, interv_n, 1.5 * interv_n, 2 * interv_n]);
    #     ax.set_yticklabels([-90, 45, 0, 45, 90])
    #     ax.set_title(title + "_Hemisphere")
    # figsum.suptitle("%s-%s-unit%03d  %s" % (self.pref_unit[0], self.pref_unit[1], self.pref_unit[2], self.explabel))
    # # figsum.savefig(join(self.savedir, "Manifold_summary_%s_norm%d.png" % (self.explabel, self.sphere_norm)))
    # # figsum.savefig(join(self.savedir, "Manifold_summary_%s_norm%d.pdf" % (self.explabel, self.sphere_norm)))
    # self.Perturb_vec = np.concatenate(tuple(self.Perturb_vec), axis=0)
    # np.save(join(self.savedir, "Manifold_score_%s" % (self.explabel)), self.score_sum)
    # np.savez(join(self.savedir, "Manifold_set_%s.npz" % (self.explabel)),
    #          Perturb_vec=self.Perturb_vec, imgsize=self.imgsize, corner=self.corner,
    #          evol_score=self.scores_all, evol_gen=self.generations, sphere_norm=self.sphere_norm)

def score_manifold(code_arr, model, G, get_score_all_fun, B=40):
    nrow = int(np.sqrt(code_arr.shape[0]))
    scores_all = []
    for i in range(0, code_arr.shape[0], B):
        with torch.no_grad():
            img_tsr = G.visualize_batch_np(code_arr[i:i + B])
            model(img_tsr.cuda())
        scores_batch = get_score_all_fun()
        scores_all.append(scores_batch)
    print(scores_all)
    scores_vec = torch.cat(scores_all, dim=0)
    score_map = scores_vec.reshape((nrow, nrow))
    return score_map
#%%
code_arr, Perturb_vec = get_manifold(zs[-1:,:].numpy(), subspace_list=("RND",), interval=18, )
score_map = score_manifold(code_arr, G, get_score_all_fun, B=40)
#%%
plt.figure()
plt.imshow(score_map)
plt.colorbar()
plt.show()
#%%
import os
from os.path import join
from core.utils import saveallforms, save_imgrid
import matplotlib
matplotlib.use('Agg')

savedir = r"F:\insilico_exps\GAN_crossViT\crossvit_18_dagger_384"
# keystr = "B1_B1_0_norm1"
# module = model.blocks[1].blocks[1][0].norm1
keystr = "B1_B1_4_norm2"
module = model.blocks[1].blocks[1][4].norm2
os.makedirs(join(savedir, keystr), exist_ok=True)
fetcher = featureFetcher_module()
for chan in [50,100,150,200,250,300]:
    for unit in range(20):
        ch_unit = chan, unit
        for repi in range(10):
            fetcher.record_module(module, keystr, ingraph=True)
            get_score_all_fun = lambda: fetcher[keystr][:, ch_unit[0], ch_unit[1]]
            z, img, score_traj = grad_evol_scorer(get_score_all_fun, steps=100, lr=0.25, return_zs=False)
            code_arr, Perturb_vec = get_manifold(z.detach().cpu().numpy(), subspace_list=("RND",), interval=18, )
            score_map = score_manifold(code_arr, model, G, get_score_all_fun, B=41)
            save_imgrid(img, join(savedir, keystr, "proto_%s_%d_%d_rep%d.png" % (keystr, ch_unit[0], ch_unit[1], repi)))

            # plt.imsave(join(savedir, keystr, "proto_%s_%d_%d_rep%d.npz" % (keystr, ch_unit[0], ch_unit[1], repi)), img)
            np.savez(join(savedir, keystr, "Manifold_evol_%s_%d_%d_rep%d.npz" % (keystr, ch_unit[0], ch_unit[1], repi)),
                        Perturb_vec=Perturb_vec, score_traj=score_traj, z=z.detach().cpu(), score_map=score_map)
            fetcher.cleanup()
            plt.figure()
            plt.imshow(score_map)
            plt.colorbar()
            plt.title("CrossViT %s ch %d unit %d" % (keystr, ch_unit[0], ch_unit[1]))
            saveallforms(join(savedir, keystr),
                         "Manifold_map_%s_%d_%d_rep%d" % (keystr, ch_unit[0], ch_unit[1], repi),)
            # plt.show()
#%%

#%%
fig = visualize_img_list(img_arr, ncol=5, nrow=8, )
fig.savefig(join(savedir, "Manifold_%s_norm%d_%d.png" % (explabel, sphere_norm, i)))
plt.close(fig)
#%%
fetcher.cleanup()

#%% Summarize figures
savedir = r"F:\insilico_exps\GAN_crossViT\crossvit_18_dagger_384"
sumdir  = r"F:\insilico_exps\GAN_crossViT\crossvit_18_dagger_384\proto_summary"
keystr = "B1_B1_4_norm2" #"B1_B1_0_norm1"
ch_unit = 300, 10
for chan in [50,100,150,200,250,300]:
    for unit in range(20):
        ch_unit = chan, unit
        figh, axs = plt.subplots(6, 5, figsize=(10, 12))
        for repi in range(10):
            irow_plus, icol = divmod(repi, 5)
            proto = plt.imread(join(savedir, keystr, "proto_%s_%d_%d_rep%d.png" % (keystr, ch_unit[0], ch_unit[1], repi)))
            savedict = np.load(join(savedir, keystr, "Manifold_evol_%s_%d_%d_rep%d.npz" % (keystr, ch_unit[0], ch_unit[1], repi)))
            axs[irow_plus, icol].imshow(proto)
            axs[irow_plus, icol].axis("off")
            axs[2+irow_plus, icol].plot(savedict["score_traj"])
            axs[4+irow_plus, icol].imshow(savedict["score_map"])
            axs[4+irow_plus, icol].set_xticks([0, 5, 10], [-90, 0, 90])
            axs[4+irow_plus, icol].set_yticks([0, 5, 10], [-90, 0, 90])
        figh.suptitle("CrossViT %s ch %d unit %d" % (keystr, ch_unit[0], ch_unit[1]))
        plt.tight_layout()
        saveallforms(sumdir, "%s_%d_%d_summary"%(keystr, chan, unit), figh)
        plt.show()
        # raise Exception
#%%
#%% Develop zone
img = torch.randn(1,3,224,224).cuda()
fetcher = featureFetcher_module()
# fetcher.record_module(model.patch_embed, "patch_embed")
# fetcher.record_module(model.blocks, "blocks")
fetcher.record_module(model.patch_embed[0], "patembed0")
fetcher.record_module(model.patch_embed[1], "patembed1")
fetcher.record_module(model.blocks[1].blocks[0][0].mlp.fc2, "B1_mlp_fc2_0_0")
fetcher.record_module(model.blocks[1].blocks[1][0].mlp.fc2, "B1_mlp_fc2_1_0")
fetcher.record_module(model.blocks[1].projs[0], "B1_proj0")
fetcher.record_module(model.blocks[1].projs[1], "B1_proj1")
fetcher.record_module(model.blocks[1].fusion[0], "B1_fusion0_0")
fetcher.record_module(model.blocks[1].fusion[1], "B1_fusion1_0")
fetcher.record_module(model.blocks[1].revert_projs[0], "B1_revproj0")
fetcher.record_module(model.blocks[1].revert_projs[1], "B1_revproj1")
fetcher.record_module(model.blocks[2].blocks[0][0].mlp.fc2, "B2_mlp_fc2_0_0")
fetcher.record_module(model.blocks[2].blocks[1][0].mlp.fc2, "B2_mlp_fc2_0_1")
fetcher.record_module(model.blocks[2].blocks[1][5].mlp.fc2, "B2_mlp_fc2_5_1")
fetcher.record_module(model.blocks[2].projs[0], "B2_proj0")
fetcher.record_module(model.blocks[2].projs[1], "B2_proj1")
fetcher.record_module(model.blocks[2].fusion[0], "B2_fusion0_0")
fetcher.record_module(model.blocks[2].fusion[1], "B2_fusion1_0")
fetcher.record_module(model.blocks[2].revert_projs[0], "B2_revproj0")
fetcher.record_module(model.blocks[2].revert_projs[1], "B2_revproj1")
fetcher.record_module(model.head[0], "head0")
fetcher.record_module(model.head[1], "head1")
model(img)
fetcher.cleanup()

for key in fetcher.activations:
    print(key, fetcher[key].shape)