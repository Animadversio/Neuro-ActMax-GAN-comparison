import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from os.path import join
import torch
from torch.nn.functional import interpolate
import matplotlib.pyplot as plt
import numpy as np
#%%
def show_anns(anns, autoscale=False):
    if isinstance(anns, dict):
        anns = [anns]
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(autoscale)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


def show_ann(ann):
    ax = plt.gca()
    m = ann['segmentation']
    img = np.ones((m.shape[0], m.shape[1], 3))
    color_mask = np.random.random((1, 3)).tolist()[0]
    for i in range(3):
        img[:,:,i] = color_mask[i]
    ax.imshow(np.dstack((img, m*0.35)))


def show_bbox(ann):
    ax = plt.gca()
    bbox = ann['bbox']
    rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                         fill=False, edgecolor='g', linewidth=3)
    ax.add_patch(rect)


def crop_with_bbox(img, ann):
    bbox = ann['bbox']
    return img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :]
#%%
ckpt_root = r"D:\Github\segment-anything\ckpts"
sam_checkpoint = join(ckpt_root, "sam_vit_h_4b8939.pth") #"sam_vit_l_0b3195.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)
predictor = SamPredictor(sam, )
#%%
mtg_path = r"F:\insilico_exps\GAN_Evol_cmp\protoimgs_merge\resnet50_linf8_.layer4.Bottleneck2_44_4_4_optim_pool.jpg"
mtg_path = r"F:\insilico_exps\GAN_Evol_cmp\protoimgs_merge\resnet50_linf8_.layer4.Bottleneck2_27_4_4_optim_pool.jpg"
mtg = plt.imread(mtg_path)
#%%
image = mtg[:, :776, :]
predictor.set_image(image)
masks = mask_generator.generate(image)
img_embed = predictor.get_image_embedding()
#%%
def filter_masks_point(masks, coord, show=False):
    match_anns = []
    for imsk, ann in enumerate(masks):
        if ann['segmentation'][coord[1], coord[0]]:
            print(imsk, ann['stability_score'], ann['area'])
            match_anns.append(ann)
            if show:
                plt.figure(figsize=(6,10))
                # plt.imshow(image)
                show_ann(ann)
                show_bbox(ann)
                plt.scatter(coord[0], coord[1], s=100, c='r')
                plt.title(f"mask {imsk} score {ann['stability_score']:.2f} area {ann['area']}")
                plt.axis('off')
                plt.tight_layout()
                plt.show()
    return match_anns


"""extract the embedding of the mask"""
def get_mask_embed(mask, img_embed):
    """Extract mask embedding

    :param mask: H x W, binary mask in numpy
    :param img_embed: 1 x c x h x w, image embedding tensor [256, 64, 64]
    :return:
        query_embed: 1 x c, mask embedding tensor
        mask_resize: 1 x 1 x h x w, the mask that is used to extract the embedding
    """
    orig_H, orig_W = mask.shape[:2]
    embed_H, embed_W = img_embed.shape[-2:]
    if orig_H >= orig_W:
        resize_W = int(embed_H * orig_W / orig_H)
        resize_H = embed_H
    else:
        resize_H = int(embed_W * orig_H / orig_W)
        resize_W = embed_W
    mask_resize = interpolate(torch.tensor(mask)[None, None].float(),
                size=(resize_H, resize_W), mode='nearest')
    query_embed = (img_embed[:, :, :resize_H, :resize_W] * mask_resize.cuda()).sum(dim=(-2, -1)) / mask_resize.sum()
    return query_embed, mask_resize
#%%
"""Find the mask intersecting with the coordinate"""
coord = [672, 160]# [416, 416]#[160, 672] # [416, 416] # [416, 160] #[160, 160]
match_anns = filter_masks_point(masks, coord, show=True)
#%%
ann = match_anns[2]
query_embed, mask_embed = get_mask_embed(ann['segmentation'], img_embed)
#%%
mask_embs = []
for ann in masks:
    query_embed, _ = get_mask_embed(ann['segmentation'], img_embed)
    mask_embs.append(query_embed)

mask_embs = torch.cat(mask_embs, dim=0)
#%%
query_id = 25
mask_sim = torch.cosine_similarity(mask_embs, mask_embs[query_id:query_id+1], dim=1)
#%%
topk_simval, topk_simid = torch.topk(mask_sim, 10)
#%%
plt.figure(figsize=(10, 10))
plt.imshow(image)
# show_anns([masks[i] for i in topk_simid.data], autoscale=True)
plt.tight_layout()
plt.show()
#%%
plt.figure(figsize=(10, 10))
# plt.imshow(image)
show_anns([masks[i] for i in topk_simid.data], autoscale=True)
plt.tight_layout()
plt.show()

#%%
"""find the nearest neighbor"""
simmap = torch.cosine_similarity(query_embed[:, :, None, None], img_embed, dim=1)
#%%
# img_embed #
with torch.no_grad():
    neck_pos_emb = sam.image_encoder.neck(sam.image_encoder.pos_embed.data.permute([0,3,1,2]))
#%%
query_pos_embed, mask_embed = get_mask_embed(ann['segmentation'], neck_pos_emb)
#%%
simmap2 = torch.cosine_similarity(query_embed[:, :, None, None] - query_pos_embed[:, :, None, None], img_embed - neck_pos_emb, dim=1)

#%%
plt.figure(figsize=(10, 10))
plt.imshow((simmap2[0].cpu() / 0.2).exp())
plt.axis('off')
plt.tight_layout()
plt.show()

#%%
from core.utils.layer_hook_utils import recursive_print
recursive_print(sam, deepest=2)
#%%
recursive_print(sam.image_encoder, deepest=1)
#%%
recursive_print(sam.image_encoder.blocks[0], deepest=1)

#%% Crop the image and mask to extract the embedding
img_crop = crop_with_bbox(image, match_anns[0])
bin_mask_crop = crop_with_bbox(match_anns[0]['segmentation'][:, :, None], match_anns[0])
#%%
predictor.set_image(img_crop)
crop_emb = predictor.get_image_embedding()
crop_query_embed, _ = get_mask_embed(bin_mask_crop[0], crop_emb)
#%%
predictor.set_image(np.fliplr(img_crop))
crop_emb2 = predictor.get_image_embedding()
crop_query_embed2, _ = get_mask_embed(np.fliplr(bin_mask_crop[0]), crop_emb2)
#%%
plt.figure(figsize=(10,10))
plt.imshow((bin_mask_crop))
plt.axis('off')
plt.tight_layout()
plt.colorbar()
plt.show()
#%%
"""find the nearest neighbor"""
# simmap = torch.cosine_similarity(crop_query_embed[:, :, None, None], img_embed, dim=1)
simmap = torch.cosine_similarity(((crop_query_embed + crop_query_embed2) / 2)[:, :, None, None], img_embed, dim=1)
#%%
plt.figure(figsize=(10,10))
plt.imshow((simmap[0].cpu() / 0.25).exp())
plt.axis('off')
plt.tight_layout()
plt.show()


#%%
list(masks[0])
#%%
plt.imshow(image)
coord = plt.ginput(1)
# Print the pixel coordinate
print("Pixel coordinate:", coord)
# Show the plot
plt.show()
#%%
plt.figure(figsize=(10,10))
plt.imshow(img_embed[0, 0].cpu())
plt.axis('off')
plt.tight_layout()
plt.show()

#%%
plt.figure(figsize=(10,10))
plt.imshow(img_embed.norm(dim=1)[0].cpu())
plt.axis('off')
plt.tight_layout()
plt.show()

#%%
plt.figure(figsize=(5,15))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.tight_layout()
plt.show()

#%% Scratch zone
coord = [160, 160]
match_anns = []
for imsk, ann in enumerate(masks):
    if ann['segmentation'][coord[1], coord[0]]:
        print(imsk, ann['stability_score'], ann['area'])
        match_anns.append(ann)
        plt.figure(figsize=(6,10))
        # plt.imshow(image)
        show_ann(ann)
        show_bbox(ann)
        plt.scatter(coord[0], coord[1], s=100, c='r')
        plt.title(f"mask {imsk} score {ann['stability_score']:.2f} area {ann['area']}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

#%% Scratch zone
# resize the segmentation mask to the embedding size and extract the embedding of the mask as query
ann = match_anns[0]
embed_H, embed_W = img_embed.shape[-2:]
orig_H, orig_W = ann['segmentation'].shape[:2]
if orig_H >= orig_W:
    resize_W = int(embed_H * orig_W / orig_H)
    resize_H = embed_H
else:
    resize_H = int(embed_W * orig_H / orig_W)
    resize_W = embed_W
mask_embed = interpolate(torch.tensor(ann['segmentation'])[None, None].float(),
            size=(resize_H, resize_W), mode='nearest')
#%%
query_embed = (img_embed[:, :, :resize_H, :resize_W] * mask_embed.cuda()).mean(dim=(-2, -1))