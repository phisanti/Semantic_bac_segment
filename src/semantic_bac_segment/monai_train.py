import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
import traceback
import json

import torch
from torch.nn import CrossEntropyLoss
from monai.transforms import (Compose, 
                              RandRotate90d, 
                              Rand2DElasticd, 
                              RandGaussianNoised,
                              RandGaussianSmoothd,
                              ScaleIntensityd,
                              RandZoomd,
                              ToTensord
                              )
from monai.data import DataLoader, Dataset, PatchDataset
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, compute_iou
from monai.networks.nets import UNet as MonaiUnet
from semantic_bac_segment.utils import get_device, tensor_debugger
from semantic_bac_segment.data_loader import TrainSplit
from semantic_bac_segment.monai_trainer import MonaiTrainer
from semantic_bac_segment.loss_functions import MultiClassDiceLoss, MultiClassWeightedBinaryCrossEntropy
from semantic_bac_segment.monay_utils import (TIFFLoader, 
                                              BacSegmentDatasetCreator, 
                                              ClearBackgroundTransform,
                                              NormalizePercentileTransform,
                                              Ensure4D,
                                              ComposeInspect)
from semantic_bac_segment.models.pytorch_cnnunet import Unet as atomai_unet
from semantic_bac_segment.models.pytorch_altmodel import UNET as base_unet
from semantic_bac_segment.model_loader import model_loader

# Full image transform (read and remove background)
img_transforms = Compose([
    TIFFLoader(keys=["image"]),
    TIFFLoader(keys=["label"], add_channel_dim=False), # If running on 2D images, change to True
    ClearBackgroundTransform(keys=["image"], sigma_r=151, method='divide', convert_32=True)
])


# Patch transfroms
patch_train_trans = Compose([
    RandZoomd(keys=["image", "label"], prob=0.5, min_zoom=0.8, max_zoom=1.2),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    #RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.1),
    RandGaussianSmoothd(keys=["image"], prob=0.5, sigma_x=(0.1, 1.1), sigma_y=(0.1, 1.1)),
    NormalizePercentileTransform(keys=["image"], pmax=95),
    ScaleIntensityd(keys=["label"]),
    #Ensure4D(keys=["image", "label"]),
    ToTensord(keys=["image", "label"])
])
patch_val_trans = Compose([
    NormalizePercentileTransform(keys=["image"], pmax=95),
    ScaleIntensityd(keys=["label"]),
    #Ensure4D(keys=["image", "label"]),
    ToTensord(keys=["image", "label"])

])




# Get datasets
source_folder='./data/source/'
mask_folder='./data/multiclass_masks/'
val_ratio = 0.3
num_samples = 2
dataset_creator = BacSegmentDatasetCreator(source_folder, mask_folder, val_ratio)
train_dataset, val_dataset= dataset_creator.create_datasets(img_transforms , img_transforms)
train_patch_dataset, val_patch_dataset = dataset_creator.create_patches(num_samples=num_samples, 
                                                                        roi_size=(256,256), 
                                                                        train_transforms=patch_train_trans, 
                                                                        val_transforms=patch_val_trans)

# Get loss and metrics
loss_function = MultiClassDiceLoss(is_sigmoid=True)
metrics = {
    'Dice': MultiClassDiceLoss(is_sigmoid=True),
    'Monai_diceloss' : DiceLoss(to_onehot_y=False, sigmoid=False),
    'CrossEntropy': MultiClassWeightedBinaryCrossEntropy(is_sigmoid=True, class_weights=[1, 1, 1, 1]),
    'Cross_entropy_pytorch' : CrossEntropyLoss()
}

device=get_device()
debugging=True

# Get list of architechtures and run Training loop
num_epochs = 2

with open('/Users/santiago/switchdrive/boeck_lab_projects/Semantic_bac_segment/train_models.json') as file:
    network_arch=json.load(file)

for model_i in network_arch:
    try:

        m=model_loader(model_i, device)
        torch.compile(m)
        optimizer = torch.optim.Adam(m.parameters(), lr=0.005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        trainer=MonaiTrainer(m, train_patch_dataset, val_patch_dataset, optimizer, scheduler, device, sigmoid_transform=True, debugging=True)
        trainer.train(loss_function, metrics, num_epochs, './results', model_i['model_name'], model_i['model_args'])

    except Exception as e:
        error_message = f"An error occurred while training model {model_i['model_name']}: {str(e)}\n"
        error_message += f"Traceback: {traceback.format_exc()}\n"
        trainer.logger.log(error_message, level='ERROR')