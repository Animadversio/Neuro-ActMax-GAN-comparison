import re
from pathlib import Path
import pickle as pkl
import pandas as pd
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import torch
from timm.models import create_model, list_models
from core.utils.CNN_scorers import load_featnet
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from core.utils.dataset_utils import ImagePathDataset, ImagePathDataset_pure
from torchvision import transforms
from core.utils.plot_utils import saveallforms
from neuro_data_analysis.neural_data_utils import get_all_masks
from neuro_data_analysis.neural_data_lib import load_img_resp_pairs, load_neural_data, get_expstr
# from ultralytics import YOLO
# model_new = YOLO("yolov8x.pt")
tabdir = Path(r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables")
meta_df = pd.read_csv(tabdir / "meta_activation_stats_w_optimizer.csv", index_col=0)
#%%
_, BFEStats = load_neural_data()
#%%
saveroot = Path(r"E:\Network_Data_Sync\BigGAN_Evol_feat_extract")

figdir = saveroot / "figsummary"
figdir.mkdir(exist_ok=True, parents=True)
#%%
input_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    transforms.ToTensor(),
])
def cnn_feat_process(imgpathlist, cnn_feat, key, batch_size=100, size=256, savename=None, sumdir=""):
    """Process images with yolo model and return results in a list of dataframes"""
    dataset = ImagePathDataset_pure(imgpathlist, img_dim=(size, size))
    feature_col = []
    for i in trange(0, len(imgpathlist), batch_size):
        imgtsr = dataset[i:i+batch_size]
        with torch.no_grad():
            results = cnn_feat(imgtsr.cuda())
        if key is not None:
            feature_col.append(results[key].cpu())
        else:
            feature_col.append(results.cpu())
        # yolo_results[i] = results
    feature_col = torch.cat(feature_col, dim=0)
    if feature_col.ndim == 4:
        feature_col = feature_col.squeeze(-1).squeeze(-1)
    meta_data = {"img_path": imgpathlist}
    meta_data_df = pd.DataFrame(meta_data)
    if savename is not None:
        torch.save(feature_col, sumdir / f"{savename}_features.pt")
        meta_data_df.to_csv(sumdir / f"{savename}_meta.csv")
        print(f"Saved {savename} features to {sumdir}")
    return feature_col, meta_data_df

#%%
sumdir = (saveroot / "resnet50_linf8")
sumdir.mkdir(exist_ok=True, parents=True)
cnn, _ = load_featnet("resnet50_linf8")
cnn_feat = create_feature_extractor(cnn, ["avgpool"])
# Model
# plt.switch_backend('module://backend_interagg')
for Expi in trange(1, 190+1):
    if BFEStats[Expi-1]["evol"] is None:
        continue
    expdir = saveroot / f"Both_Exp{Expi}"
    expdir.mkdir(exist_ok=True)
    imgfps_col0, resp_vec0, bsl_vec0, gen_vec0 = \
        load_img_resp_pairs(BFEStats, Expi, "Evol", thread=0, output_fmt="vec")
    imgfps_col1, resp_vec1, bsl_vec1, gen_vec1 = \
        load_img_resp_pairs(BFEStats, Expi, "Evol", thread=1, output_fmt="vec")
    feature_arr0, meta_data_df0 = cnn_feat_process(imgfps_col0, cnn_feat,"avgpool",
         batch_size=100, size=224, savename=f"Evol_Exp{Expi:03d}_thread0", sumdir=sumdir)
    feature_arr1, meta_data_df1 = cnn_feat_process(imgfps_col1, cnn_feat,"avgpool",
            batch_size=100, size=224, savename=f"Evol_Exp{Expi:03d}_thread1", sumdir=sumdir)


#%%
list_models(filter="*dino*")
#%%
saveroot = Path(r"E:\Network_Data_Sync\BigGAN_Evol_feat_extract")
sumdir = (saveroot / "vit_base_patch8_224_dino")
sumdir.mkdir(exist_ok=True, parents=True)

dino_feat = create_model("vit_base_patch8_224_dino", pretrained=True,).cuda()
for Expi in trange(1, 190+1):
    if BFEStats[Expi-1]["evol"] is None:
        continue
    expdir = saveroot / f"Both_Exp{Expi}"
    expdir.mkdir(exist_ok=True)
    imgfps_col0, resp_vec0, bsl_vec0, gen_vec0 = \
        load_img_resp_pairs(BFEStats, Expi, "Evol", thread=0, output_fmt="vec")
    imgfps_col1, resp_vec1, bsl_vec1, gen_vec1 = \
        load_img_resp_pairs(BFEStats, Expi, "Evol", thread=1, output_fmt="vec")
    feature_arr0, meta_data_df0 = cnn_feat_process(imgfps_col0, dino_feat, None,
         batch_size=50, size=224, savename=f"Evol_Exp{Expi:03d}_thread0", sumdir=sumdir)
    feature_arr1, meta_data_df1 = cnn_feat_process(imgfps_col1, dino_feat, None,
         batch_size=50, size=224, savename=f"Evol_Exp{Expi:03d}_thread1", sumdir=sumdir)

#%%
