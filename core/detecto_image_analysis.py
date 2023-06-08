from detecto.core import Model
from detecto import utils, visualize
# https://medium.com/pytorch/detecto-build-and-train-object-detection-models-with-pytorch-5f31b68a8109
model = Model()
def print_detecto_results(labels, scores, boxes=None):
    for i, label in enumerate(labels):
        print(f"{label}: {scores[i]}","" if boxes is None else boxes[i] )


# image = utils.read_image('image.jpg')  # Helper function to read in images
#%%
import os
from os.path import join
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import trange, tqdm
from neuro_data_analysis.neural_data_lib import parse_montage
montage_dir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoSummary"
#%%
# _, BFEStats = load_neural_data()
# outdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Proto_covtsr_similarity"
# Animal = "Both"
# Expi = 111
all_detecto_results = {}
for Expi in trange(1, 191):  # range(160, 191):
    if not os.path.exists(join(montage_dir, "Exp%d_proto_attr_montage.jpg" % Expi)):
        continue
    mtg_S = parse_montage(plt.imread(join(montage_dir, "Exp%d_proto_attr_montage.jpg" % Expi)))
    results = {}
    for key in mtg_S.keys():
        # labels, boxes, scores = model.predict_top(mtg_S[key])
        labels, boxes, scores = model.predict(mtg_S[key])  # Get all predictions on an image
        print("\n", key)
        print_detecto_results(labels, scores, boxes)
        # visualize.show_labeled_image(mtg_S[key], boxes, labels)
        results[key] = (labels, boxes, scores)
    all_detecto_results[Expi] = results
    # labels, boxes, scores = model.predict(mtg_S["FC_maxblk"])  # Get all predictions on an image
    # print("FC_maxblk")
    # print_detecto_results(labels, scores)
    # labels, boxes, scores = model.predict(mtg_S["BG_maxblk"])  # Get all predictions on an image
    # print("BG_maxblk")
    # print_detecto_results(labels, scores)
    # predictions = model.predict_top(mtg_S["BG_reevol_pix"])  # Same as above, but returns only the top predictions
    # raise NotImplementedError

#%%
with open(r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoSummary\all_detecto_results.pkl", "wb") as f:
    pickle.dump(all_detecto_results, f)
#%%
detect_col = {}
for Expi, results in all_detecto_results.items():
    row = {}
    for key, (labels, boxes, scores) in results.items():
        if scores.shape[0] == 0:
            row[key] = 0.0
        else:
            row[key] = max(scores).item()
        # detect_col[(Expi, key)] = np.array([labels, boxes, scores]).T
    detect_col[Expi] = row
detect_df = pd.DataFrame(detect_col).T
#%%
from core.utils.stats_utils import ttest_rel_print_df
ttest_rel_print_df(detect_df, None, "BG_maxblk", "FC_maxblk")
# ttest_rel_print_df(detect_df, None, "BG_maxblk", "FC_maxblk")
#%%
statdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\Evol_activation_cmp"
meta_df = pd.read_csv(join(statdir, "meta_stats.csv"), index_col=0)
#%%
meta_detect_df = meta_df.merge(detect_df, left_index=True, right_index=True)
#%%

meta_detect_df.groupby("visual_area")[["BG_maxblk", "FC_maxblk"]].mean()
meta_detect_df.groupby("visual_area")[["BG_reevol_pix", "FC_reevol_pix"]].mean()
meta_detect_df.groupby("visual_area")[["BG_reevol_G", "FC_reevol_G"]].mean()

#%%
import torch
import matplotlib.pyplot as plt
# Model
yolomodel = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
plt.switch_backend('module://backend_interagg')
# #%%
# # Images
#%%
imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images
# Inference
results = yolomodel(imgs)
#%%
plt.figure()
plt.imshow(results.ims[0])
plt.show()
#%%
plt.figure()
plt.imshow(results.render()[0])
plt.show()
#%%

plt.figure()
plt.imshow(torch.rand(3, 640, 640).numpy().transpose(1,2,0))
plt.show()
#%%

plt.figure()
plt.imshow(torch.rand(100,100))
plt.show()
#%%
#%%
all_yolo_results = {}
for Expi in trange(1, 191):  # range(160, 191):
    if not os.path.exists(join(montage_dir, "Exp%d_proto_attr_montage.jpg" % Expi)):
        continue
    mtg_S = parse_montage(plt.imread(join(montage_dir, "Exp%d_proto_attr_montage.jpg" % Expi)))
    # turn the dict of images into uint8 format
    for key in mtg_S.keys():
        mtg_S[key] = (mtg_S[key]*255).astype("uint8")
    results = yolomodel([*mtg_S.values()], size=224)
    print(results)
    all_yolo_results[Expi] = results

#%%
imgkeys = list(mtg_S.keys())
#%%
yolo_stats = {}
for Expi, results in all_yolo_results.items():
    full_df = results.pandas().xyxy
    stats = {}
    for imgi, key in enumerate(imgkeys):
        # if
        stats[key] = full_df[imgi].confidence.max()
    yolo_stats[Expi] = stats

yolo_stats_df = pd.DataFrame(yolo_stats).T

#%%
yolo_stats_df.fillna(0.0).mean()
#%%
yolo_stats_df.count() / yolo_stats_df.shape[0]
#%%
result = yolomodel((mtg_S["FC_maxblk"] * 255).astype("uint8"), size=224)
plt.imshow(result.render()[0])
plt.show()
#%%
from pathlib import Path
from core.utils.dataset_utils import ImageDataset_filter, DataLoader
savedir = Path("/n/scratch3/users/b/biw905/GAN_sample_fid/BigGAN_std_008")
dataset = ImageDataset_filter(savedir)
dataloader = DataLoader(dataset, batch_size=50, shuffle=False, num_workers=4)
#%%
# use yolo model with data loader


#%%
from core.utils.GAN_utils import upconvGAN, BigGAN_wrapper, loadBigGAN
G = upconvGAN("fc6")
G.cuda().eval().requires_grad_(False)
#%%
BG = BigGAN_wrapper(loadBigGAN())
#%%
gen_img = 255*G.visualize(4*torch.randn(1, 4096).cuda()).cpu().permute(0,2,3,1).numpy()
deepsim_result = yolomodel(gen_img[0], size=256)
plt.figure(figsize=[4.5, 4.5])
plt.imshow(deepsim_result.render()[0])
plt.axis("off")
plt.tight_layout()
plt.show()
#%%
gen_img = BG.visualize(BG.sample_vector(1)).cpu().permute(0,2,3,1).numpy()*255
deepsim_result = yolomodel(gen_img[0], size=256)
plt.figure(figsize=[4.5, 4.5])
plt.imshow(deepsim_result.render()[0])
plt.axis("off")
plt.tight_layout()
plt.show()
#%%
deepsim_result = yolomodel(gen_img, size=224)
plt.figure(figsize=[4.5, 4.5])
plt.imshow(deepsim_result.render()[0])
plt.axis("off")
plt.show()
#%%
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
#%%
inputs = [utils.prepare_input(uri) for uri in uris]
tensor = utils.prepare_tensor(inputs)
#%%
