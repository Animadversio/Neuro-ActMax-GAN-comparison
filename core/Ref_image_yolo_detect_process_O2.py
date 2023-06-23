import torch
from pathlib import Path
import pickle as pkl
import pandas as pd
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
# from ultralytics import YOLO
# model_new = YOLO("yolov8x.pt")
from core.yolo_lib import yolo_process_objconf, yolo_process
#%%
# Model
yolomodel = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
# plt.switch_backend('module://backend_interagg')
saveroot = Path(r"/n/scratch3/users/b/biw905/GAN_sample_fid")
sumdir = (saveroot / "yolo_summary")
sumdir.mkdir(exist_ok=True)
#%%
def yolo_process(imgpathlist, batch_size=100, size=256, savename=None, sumdir=sumdir):
    """Process images with yolo model and return results in a list of dataframes"""
    results_dfs = []
    for i in trange(0, len(imgpathlist), batch_size):
        results = yolomodel(imgpathlist[i:i+batch_size], size=size)
        results_dfs.extend(results.pandas().xyxy)
        # yolo_results[i] = results

    yolo_stats = {}
    for i, single_df in tqdm(enumerate(results_dfs)):
        yolo_stats[i] = {"confidence": single_df.confidence.max(),
                        "class": single_df["class"][single_df.confidence.argmax()] if len(single_df) > 0 else None,
                        "n_objs": len(single_df),
                        "img_path": imgpathlist[i]}
    yolo_stats_df = pd.DataFrame(yolo_stats).T
    if savename is not None:
        yolo_stats_df.to_csv(sumdir / f"{savename}_yolo_stats.csv")
        pkl.dump(results_dfs, open(sumdir / f"{savename}_dfs.pkl", "wb"))
        print(f"Saved to {sumdir / f'{savename}_dfs.pkl'}")
        print(f"Saved to {sumdir / f'{savename}_yolo_stats.csv'}")
    print("Fraction of images with objects", (yolo_stats_df.n_objs > 0).mean())
    print("confidence", yolo_stats_df.confidence.mean(), "confidence with 0 filled",
          yolo_stats_df.confidence.fillna(0).mean())
    print("most common class", yolo_stats_df["class"].value_counts().index[0])
    print("n_objs", yolo_stats_df.n_objs.mean(), )
    return results_dfs, yolo_stats_df

#%%

imgdir = saveroot / "pink_noise"
imgpathlist = sorted(list(Path(imgdir).glob("sample*.png")))
results_dfs, yolo_stats_df = yolo_process(imgpathlist,
                  batch_size=100, size=256, savename="pink_noise")
#%%
imgdir_name = "resnet50_linf8_gradevol_avgpool"
for imgdir_name in [
    "resnet50_linf8_gradevol",
    "resnet50_linf8_gradevol_avgpool",
    "resnet50_linf8_gradevol_layer4",
    "resnet50_linf8_gradevol_layer3",
]:
    imgdir = saveroot / imgdir_name
    imgpathlist = sorted(list(Path(imgdir).glob("class*")))
    results_dfs, yolo_stats_df = yolo_process(imgpathlist, batch_size=100, size=256,
                                              savename=imgdir_name)

#%%
imgdir = saveroot / "BigGAN_1000cls_std07_invert"
imgpathlist = sorted(list(Path(imgdir).glob("FC_invert*.png")))
results_dfs, yolo_stats_df = yolo_process(imgpathlist, batch_size=100, size=256,
                                          savename="BigGAN_1000cls_std07_FC_invert")
imgpathlist = sorted(list(Path(imgdir).glob("BG*.png")))
results_dfs, yolo_stats_df = yolo_process(imgpathlist, batch_size=100, size=256,
                                          savename="BigGAN_1000cls_std07")

#%%
for imgdir_name in [
        "DeePSim_4std",
        "BigGAN_std_008",
        "BigGAN_trunc07",
        ]:
    imgdir = saveroot / imgdir_name
    imgpathlist = sorted(list(Path(imgdir).glob("sample*.png")))
    results_dfs, yolo_stats_df = yolo_process(imgpathlist, batch_size=100, size=256,
                                              savename=imgdir_name)
    print("Fraction of images with objects", (yolo_stats_df.n_objs > 0).mean())
    print("confidence", yolo_stats_df.confidence.mean(), "confidence with 0 filled", yolo_stats_df.confidence.fillna(0).mean())
    print("most common class", yolo_stats_df["class"].value_counts().index[0])
    print("n_objs", yolo_stats_df.n_objs.mean(), )

#%%

imgdir = r"/home/biw905/Datasets/imagenet-valid/valid"
imgpathlist = sorted(list(Path(imgdir).glob("*.JPEG")))
results_dfs, yolo_stats_df = yolo_process(imgpathlist, batch_size=100, size=256,
                                            savename="imagenet_valid")
print("Fraction of images with objects", (yolo_stats_df.n_objs > 0).mean())
print("confidence", yolo_stats_df.confidence.mean(), "confidence with 0 filled", yolo_stats_df.confidence.fillna(0).mean())
print("most common class", yolo_stats_df["class"].value_counts().index[0])
print("n_objs", yolo_stats_df.n_objs.mean(), )


#%%
from core.yolo_lib import yolo_process_objconf, yolo_process

yolomodel = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
# plt.switch_backend('module://backend_interagg')
saveroot = Path(r"/n/scratch3/users/b/biw905/GAN_sample_fid")
sumdir = (saveroot / "yolo_objconf_summary")
sumdir.mkdir(exist_ok=True)
#%%
imgdir = r"/home/biw905/Datasets/imagenet-valid/valid"
imgpathlist = sorted(list(Path(imgdir).glob("*.JPEG")))
results_dfs, yolo_stats_df = yolo_process_objconf(yolomodel, imgpathlist, batch_size=100, size=256,
                                savename="imagenet_valid", sumdir=sumdir, use_letterbox_resize=True)
# results_dfs, yolo_stats_df = yolo_process_objconf(yolomodel, imgpathlist, batch_size=100, size=256,
#                                             savename="imagenet_valid", sumdir=sumdir)
#%%
imgdir = saveroot / "pink_noise"
imgpathlist = sorted(list(Path(imgdir).glob("sample*.png")))
results_dfs, yolo_stats_df = yolo_process_objconf(yolomodel, imgpathlist,
                  batch_size=100, size=256, savename="pink_noise", sumdir=sumdir)
#%%
for imgdir_name in [
        "DeePSim_4std",
        "BigGAN_std_008",
        "BigGAN_trunc07",
        ]:
    imgdir = saveroot / imgdir_name
    imgpathlist = sorted(list(Path(imgdir).glob("sample*.png")))
    results_dfs, yolo_stats_df = yolo_process_objconf(yolomodel, imgpathlist, batch_size=100, size=256,
                                                      savename=imgdir_name, sumdir=sumdir)
#%%
for imgdir_name in [
    "resnet50_linf8_gradevol",
    "resnet50_linf8_gradevol_avgpool",
    "resnet50_linf8_gradevol_layer4",
    "resnet50_linf8_gradevol_layer3",
]:
    imgdir = saveroot / imgdir_name
    imgpathlist = sorted(list(Path(imgdir).glob("class*")))
    results_dfs, yolo_stats_df = yolo_process_objconf(yolomodel, imgpathlist, batch_size=100, size=256,
                                              savename=imgdir_name, sumdir=sumdir)
#%%
imgdir = saveroot / "BigGAN_1000cls_std07_invert"
imgpathlist = sorted(list(Path(imgdir).glob("FC_invert*.png")))
results_dfs, yolo_stats_df = yolo_process_objconf(yolomodel, imgpathlist, batch_size=100, size=256,
                                          savename="BigGAN_1000cls_std07_FC_invert", sumdir=sumdir)
imgpathlist = sorted(list(Path(imgdir).glob("BG*.png")))
results_dfs, yolo_stats_df = yolo_process_objconf(yolomodel, imgpathlist, batch_size=100, size=256,
                                          savename="BigGAN_1000cls_std07", sumdir=sumdir)
#%%




# imgdir = saveroot / "BigGAN_std_008"
# imgpathlist = sorted(list(Path(imgdir).glob("sample*.png")))
# results_dfs, yolo_stats_df = yolo_process_objconf(yolomodel, imgpathlist[:50], batch_size=100, size=256,
#                                                       savename=None, sumdir=None)

