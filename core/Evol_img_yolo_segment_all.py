import torch
import re
from pathlib import Path
import pickle as pkl
import pandas as pd
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from core.utils.plot_utils import saveallforms
from neuro_data_analysis.neural_data_utils import get_all_masks
from neuro_data_analysis.neural_data_lib import load_img_resp_pairs, load_neural_data, get_expstr
# from ultralytics import YOLO
# model_new = YOLO("yolov8x.pt")
#%%
_, BFEStats = load_neural_data()
#%%
saveroot = Path(r"E:\Network_Data_Sync\BigGAN_Evol_yolo")
sumdir = (saveroot / "yolo_v8_seg_summary")
sumdir.mkdir(exist_ok=True)

figdir = saveroot / "figsummary"
figdir.mkdir(exist_ok=True)
#%%
def _masks_to_np(masks):
    if masks is None:
        return None
    else:
        return masks.data.to(bool).cpu().numpy()

def _get_segments(masks,):
    if masks is None:
        return []
    else:
        return masks.xy


def result2xyxy_df(result):
    tab_data = []
    for box in result.boxes:
        tab_data.append(box.data[0, :5].tolist() + [int(box.cls.item())] + [result.names[int(box.cls.item())]])
    result_df = pd.DataFrame(tab_data,
                             columns=["xmin", "ymin", "xmax", "ymax", 'confidence', "class", "class_name"])
    return result_df

def yolov8_segment_process(yolomodel, imgpathlist, batch_size=100, size=256, savename=None, sumdir=sumdir):
    """Process images with yolo model and return results in a list of dataframes"""
    results_dfs = []
    boxes_all = []
    masks_all = []
    segments_all = []
    for i in trange(0, len(imgpathlist), batch_size):
        results = yolomodel.predict(imgpathlist[i:i+batch_size], imgsz=size)
        boxes_all.extend([r.boxes for r in results])
        masks_all.extend([_masks_to_np(r.masks) for r in results])
        segments_all.extend([_get_segments(r.masks) for r in results])
        results_dfs.extend([result2xyxy_df(r) for r in results])
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
        pkl.dump({"boxes": boxes_all, "masks": masks_all, "segments": segments_all},
                 open(sumdir / f"{savename}_results.pkl", "wb"))
        print(f"Saved to {sumdir / f'{savename}_dfs.pkl'}")
        print(f"Saved to {sumdir / f'{savename}_yolo_stats.csv'}")
    print("Fraction of images with objects", (yolo_stats_df.n_objs > 0).mean())
    print("confidence", yolo_stats_df.confidence.mean(), "confidence with 0 filled",
          yolo_stats_df.confidence.fillna(0).mean())
    print("most common class", yolo_stats_df["class"].value_counts().index[0])
    print("n_objs", yolo_stats_df.n_objs.mean(), )
    return results_dfs, yolo_stats_df, boxes_all, masks_all, segments_all


# Model
# yolomodel = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
# plt.switch_backend('module://backend_interagg')
#%%
from ultralytics import YOLO
from ultralytics.nn.autoshape import AutoShape
model_seg = YOLO('yolov8x-seg.pt')
# model_seg = AutoShape(model_seg)
#%%
for Expi in trange(1, 190+1):
    if BFEStats[Expi-1]["evol"] is None:
        continue
    expdir = saveroot / f"Both_Exp{Expi}"
    expdir.mkdir(exist_ok=True)
    imgfps_col0, resp_vec0, bsl_vec0, gen_vec0 = \
        load_img_resp_pairs(BFEStats, Expi, "Evol", thread=0, output_fmt="vec")
    imgfps_col1, resp_vec1, bsl_vec1, gen_vec1 = \
        load_img_resp_pairs(BFEStats, Expi, "Evol", thread=1, output_fmt="vec")
    results_dfs0, yolo_stats_df0, boxes_0, masks_0, segments_0 = yolov8_segment_process(model_seg, imgfps_col0,
                batch_size=100, size=256, savename=f"Exp{Expi:03d}_thread0", sumdir=sumdir)
    results_dfs1, yolo_stats_df1, boxes_1, masks_1, segments_1 = yolov8_segment_process(model_seg, imgfps_col1,
                 batch_size=100, size=256, savename=f"Exp{Expi:03d}_thread1", sumdir=sumdir)
    # raise Exception("Stop here")

#%%
results_dfs0, yolo_stats_df0, boxes_0, masks_0, segments_0 = yolov8_segment_process(model_seg, imgfps_col0,
                batch_size=100, size=256, savename=f"Exp{Expi:03d}_thread0", sumdir=sumdir)
results_dfs1, yolo_stats_df1, boxes_1, masks_1, segments_1 = yolov8_segment_process(model_seg, imgfps_col1,
                 batch_size=100, size=256, savename=f"Exp{Expi:03d}_thread1", sumdir=sumdir)
#%%
boxes_0, masks_0 = yolov8_segment_process(model_seg, imgfps_col0, batch_size=100, size=256,)
#%%
pkl.dump({"boxes": boxes_0, "masks": masks_0}, open(saveroot / "yolo_v8_seg_summary" / f"Exp{Expi:03d}_thread{0}_results.pkl", "wb"))
#%%
# results = model_uly(imgfps_col0[-40:])
# results = model_seg.predict(imgfps_col1[-30:])
# results = model_seg.predict(imgfps_col0[-40:], imgsz=256)
results = model_seg.predict(imgfps_col0, imgsz=256)
#%%
plt.imshow(results[36].plot(masks=True))
plt.show()
#%%

for box, mask in zip(boxes_0, masks_0):
    if mask is None:
        continue
    assert len(box) == len(mask)
#%%



result2xyxy_df(results[0])
#%%
result = results[10]
#%%
tab_data = torch.concat([result.boxes.xyxy, result.boxes.conf, result.boxes.cls, ], dim=1).cpu().numpy()
#%%
