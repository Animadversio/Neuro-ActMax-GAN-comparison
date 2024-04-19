"""
little library to load LRM models from local repo and weights, or from Torch hub repo and weights.
Handling the issue of multiple compute node trying to download from hub at the same time, 
and has collision error at the TORCH HOME.

Partially source from: 
    https://github.com/harvard-visionlab/lrm-steering/blob/main/lrm_models/lrms_neurips2023.py
"""
import re
import os
import torch
from os.path import join

TORCH_HOME = os.environ["TORCH_HOME"]

weight_urls = {
    "alexnet_lrm1": "https://s3.us-east-1.wasabisys.com/visionlab-projects/dnn_feedback_dev/logs/set14/set14_alexnet_lrm_cls6to8_9to0_2steps_stepwise/df5a9767-1046-4d0f-9f82-649c0e5c7881/set14_alexnet_lrm_cls6to8_9to0_2steps_stepwise_final_weights-40b29a3427.pth",
    "alexnet_lrm2": "https://s3.us-east-1.wasabisys.com/visionlab-projects/dnn_feedback_dev/logs/set15/set15_alexnet_torchvision_imagenet1k_lrm_2back_2steps/84bdc4f4-1de0-4438-941b-43e574298694/set15_alexnet_torchvision_imagenet1k_lrm_2back_2steps_final_weights-17b4229a30.pth",
    "alexnet_lrm3": "https://s3.us-east-1.wasabisys.com/visionlab-projects/dnn_feedback_dev/logs/set15/set15_alexnet_torchvision_imagenet1k_lrm_3back_2steps/28453e80-c5e5-4d76-bc81-99c5fade39ff/set15_alexnet_torchvision_imagenet1k_lrm_3back_2steps_final_weights-63ab1b3b06.pth",
}
weight_paths = {
    "alexnet_lrm1": join(TORCH_HOME, "hub", "set14_alexnet_lrm_cls6to8_9to0_2steps_stepwise_final_weights-40b29a3427.pth",),
    "alexnet_lrm2": join(TORCH_HOME, "hub", "set15_alexnet_torchvision_imagenet1k_lrm_2back_2steps_final_weights-17b4229a30.pth",),
    "alexnet_lrm3": join(TORCH_HOME, "hub", "set15_alexnet_torchvision_imagenet1k_lrm_3back_2steps_final_weights-63ab1b3b06.pth",),
}

layername_map = {".feedforward.features.Conv2d0": "conv1",
                ".feedforward.features.ReLU1": "conv1_relu",
                ".feedforward.features.MaxPool2d2": "pool1",
                ".feedforward.features.Conv2d3": "conv2",
                ".feedforward.features.ReLU4": "conv2_relu",
                ".feedforward.features.MaxPool2d5": "pool2",
                ".feedforward.features.Conv2d6": "conv3",
                ".feedforward.features.ReLU7": "conv3_relu",
                ".feedforward.features.Conv2d8": "conv4",
                ".feedforward.features.ReLU9": "conv4_relu",
                ".feedforward.features.Conv2d10": "conv5",
                ".feedforward.features.ReLU11": "conv5_relu",
                ".feedforward.features.MaxPool2d12": "pool5",
                ".feedforward.AdaptiveAvgPool2davgpool": "avgpool",
                ".feedforward.classifier.Dropout0": "avgpool_dropout",
                ".feedforward.classifier.Linear1": "fc6",
                ".feedforward.classifier.ReLU2": "fc6_relu",
                ".feedforward.classifier.Dropout3": 'fc6_dropout',
                ".feedforward.classifier.Linear4": "fc7",
                ".feedforward.classifier.ReLU5": "fc7_relu",
                ".feedforward.classifier.Linear6": "fc8",}
layername_inv_map = {v: k for k, v in layername_map.items()}

lrm_repo_path = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/torch_cache/hub/harvard-visionlab_lrm-steering_main"

def format_state_dict(checkpoint):
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    state_dict = {k.replace("module.",""):v for k,v in state_dict.items()}
    # we renamed "backbone" to be "feedforward"
    state_dict = {k.replace("backbone.", "feedforward."):v for k,v in state_dict.items()}
    # we adjusted the lrm module nameing to replace "." with "_" in layer names (instead of just trimming the .)
    pattern = re.compile(r"features(\d+)")
    state_dict = {pattern.sub(lambda m: f"features_{m.group(1)}", k):v for k,v in state_dict.items()}
    return state_dict


def load_LRM_models(modelname="alexnet_lrm3", source="local"):
    if source == "local":
        print("Loading model from local repo and weights.")
        model, transforms = torch.hub.load(lrm_repo_path, modelname, source='local',
                        pretrained=False, steering=True, force_reload=True)
        model.load_state_dict(format_state_dict(torch.load(weight_paths[modelname])))
    elif source == "localweight":
        print("Loading model from Github repo and local weights.")
        model, transforms = torch.hub.load('harvard-visionlab/lrm-steering', modelname, 
                                        pretrained=False, steering=True, force_reload=True)
        model.load_state_dict(format_state_dict(torch.load(weight_paths[modelname])))
    elif source == "hub":
        print("Loading model from Torch hub repo and weights.")
        model, transforms = torch.hub.load('harvard-visionlab/lrm-steering', modelname, 
                                           pretrained=True, steering=True, force_reload=True)
    else:
        raise ValueError("source must be one of 'local', 'localweight', 'hub'")
    
    return model, transforms


