import torch
import torchvision
import numpy as np
# # from ultralytics.yolo.utils.general import xywh2xyxy
# try:
#     from ultralytics.yolo.utils.metrics import box_iou
#     from ultralytics.yolo.utils.ops import xywh2xyxy
#     from ultralytics.nn.autoshape import scale_boxes, Detections
# except ImportError:
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def non_max_suppression_obj(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=1000,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,7) tensor per image [xyxy, conf, cls, obj_conf]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    # t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 8 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        obj_conf = x[:, 4:5]
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), obj_conf[i], x[i, 5 + j, None] / obj_conf[i], mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), obj_conf, conf / obj_conf, mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        # if (time.time() - t) > time_limit:
        #     LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
        #     break  # time limit exceeded

    return output


from pathlib import Path
import pickle as pkl
import pandas as pd
from tqdm import tqdm, trange
def yolo_process(yolomodel, imgpathlist, batch_size=100, size=256, savename=None, sumdir=Path("")):
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
    if len(yolo_stats_df) > 0:
        print("most common class", yolo_stats_df["class"].value_counts().index[0])
    print("n_objs", yolo_stats_df.n_objs.mean(), )
    return results_dfs, yolo_stats_df


def nms_output2df(nms_results, class_names=None):
    """
    :param nms_results: list of torch.tensor shape (n, 8)
            x1, y1, x2, y2, conf, cls, obj_conf, cls_conf
    :return:
        list of pandas.DataFrame
    """
    df_list = []
    for nms_result in nms_results:
        df = pd.DataFrame(nms_result.numpy(), columns=\
                        ['xmin', 'ymin', 'xmax', 'ymax',
                         'confidence', 'class', 'obj_conf', 'cls_conf'])
        df['class'] = df['class'].astype(int)
        if class_names is not None:
            df['name'] = df['class'].apply(lambda x: class_names[x]) if not df.empty else None
        df_list.append(df)
    return df_list


def load_batch_imgpaths(imgpaths, size=256, ):
    # https://github.com/ultralytics/yolov5/blob/2334aa733872bc4bb3e1a1ba90e5fd319399596f/models/common.py#LL679C9-L679C9
    # transforms.ToTensor()  # normalizes to 0-1
    from torchvision import transforms
    import matplotlib.pyplot as plt
    from PIL import Image
    import torch.nn.functional as F
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    imglist = []
    for imgpath in imgpaths:
        im_pil = Image.open(imgpath).convert('RGB')
        imtsr = transform(im_pil)
        imglist.append(imtsr)
        # im_np = plt.imread(imgpath)
        # imtsr = torch.from_numpy(im_np.transpose(2, 0, 1)).unsqueeze(0)
        # # imtsr = F.interpolate(imtsr, size=size, mode='bilinear',)
        # imglist.append(imtsr)
        # n, ims = (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (1, [ims])  # number, list of images
        # shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        # for i, im in enumerate(ims):
        #     f = f'image{i}'  # filename
        #     if isinstance(im, (str, Path)):  # filename or uri
        #         im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
        #         im = np.asarray(exif_transpose(im))
        #     elif isinstance(im, Image.Image):  # PIL Image
        #         im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
        #     files.append(Path(f).with_suffix('.jpg').name)
        #     if im.shape[0] < 5:  # image in CHW
        #         im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
        #     im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # enforce 3ch input
        #     s = im.shape[:2]  # HWC
        #     shape0.append(s)  # image shape
        #     g = max(size) / max(s)  # gain
        #     shape1.append([int(y * g) for y in s])
        #     ims[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        # shape1 = [make_divisible(x, self.stride) for x in np.array(shape1).max(0)]  # inf shape
        # x = [letterbox(im, shape1, auto=False)[0] for im in ims]  # pad
        # x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
        # x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32

        # imglist.append(x)
    imgtsrs = torch.stack(imglist, 0)
    return imgtsrs


def yolo_process_objconf(yolomodel, imgpathlist, batch_size=100, size=256, savename=None, sumdir=Path("")):
    """Process images with yolo model and return results in a list of dataframes"""
    results_dfs = []
    for i in trange(0, len(imgpathlist), batch_size):
        # results = yolomodel(imgpathlist[i:i+batch_size], size=size)
        imtsrs = load_batch_imgpaths(imgpathlist[i:i+batch_size], size=size)
        outtsr = yolomodel(imtsrs.cuda()).cpu()
        nms_result = non_max_suppression_obj(outtsr,
                                             yolomodel.conf, yolomodel.iou,
                                             yolomodel.classes, yolomodel.agnostic,
                                             yolomodel.multi_label,
                                             max_det=yolomodel.max_det)  # NMS
        result_df = nms_output2df(nms_result, yolomodel.names)
        results_dfs.extend(result_df)
        # yolo_results[i] = results

    yolo_stats = {}
    for i, single_df in tqdm(enumerate(results_dfs)):
        if len(single_df) > 0:
            max_row = single_df.loc[single_df.confidence.argmax()]
            yolo_stats[i] = {"xmin": max_row.xmin,
                             "ymin": max_row.ymin,
                             "xmax": max_row.xmax,
                             "ymax": max_row.ymax,
                             "confidence": max_row.confidence,
                             'obj_confidence': max_row.obj_conf,
                             'cls_confidence': max_row.cls_conf,
                             "class": max_row["class"],
                             "n_objs": len(single_df),
                             "img_path": imgpathlist[i]}
        else:
            yolo_stats[i] = { "xmin": None, "ymin": None,
                              "xmax": None, "ymax": None,
                              "confidence": None,
                              'obj_confidence': None,
                              'cls_confidence': None,
                              "class": None,
                              "n_objs": len(single_df),
                              "img_path": imgpathlist[i]}

    yolo_stats_df = pd.DataFrame(yolo_stats).T
    if savename is not None:
        yolo_stats_df.to_csv(sumdir / f"{savename}_yolo_objconf_stats.csv")
        pkl.dump(results_dfs, open(sumdir / f"{savename}_objconf_dfs.pkl", "wb"))
        print(f"Saved to {sumdir / f'{savename}_objconf_dfs.pkl'}")
        print(f"Saved to {sumdir / f'{savename}_yolo_objconf_stats.csv'}")
    print("Fraction of images with objects", (yolo_stats_df.n_objs > 0).mean())
    print("confidence", yolo_stats_df.confidence.mean(), "confidence with 0 filled",
          yolo_stats_df.confidence.fillna(0).mean())
    print("obj_confidence", yolo_stats_df.obj_confidence.mean(), "obj_confidence with 0 filled",
          yolo_stats_df.obj_confidence.fillna(0).mean())
    print("cls_confidence", yolo_stats_df.cls_confidence.mean(), "cls_confidence with 0 filled",
          yolo_stats_df.cls_confidence.fillna(0).mean())
    print("n_objs", yolo_stats_df.n_objs.mean(), )
    if len(yolo_stats_df) > 0:
        most_common_class = yolo_stats_df["class"].value_counts().index[0]
        print("most common class", most_common_class, " name:", yolomodel.names[most_common_class])
    return results_dfs, yolo_stats_df
# im_np = plt.imread(r"F:\insilico_exps\GAN_sample_fid\BigGAN_std_008\sample0310.png")
# im_np = plt.imread(r"F:\insilico_exps\GAN_sample_fid\BigGAN_std_008\sample0398.png")
# im_np = plt.imread(r"F:\insilico_exps\GAN_sample_fid\DeePSim_4std\sample0039.png")
# im_np = plt.imread(r"F:\insilico_exps\GAN_sample_fid\BigGAN_std_008\sample0709.png")
#
# # im_np = plt.imread(r"F:\insilico_exps\GAN_sample_fid\DeePSim_4std\sample0390.png")
# imtsr = torch.from_numpy(im_np.transpose(2, 0, 1)).unsqueeze(0).cuda()
# outtsr = yolomodel(imtsr).cpu()
# nms_result = non_max_suppression_obj(outtsr,
#                          yolomodel.conf, yolomodel.iou,
#                          yolomodel.classes, yolomodel.agnostic,
#                          yolomodel.multi_label,
#                          max_det=yolomodel.max_det)  # NMS
# Detections(imtsr, nms_result, None, None, None, None)
