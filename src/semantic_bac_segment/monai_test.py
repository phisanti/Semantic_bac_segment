# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from glob import glob

import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImage,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
    NormalizeIntensityd
)
from monai.visualize import plot_2d_or_3d_image
from semantic_bac_segment.utils import get_device, tensor_debugger
from semantic_bac_segment.data_loader import TrainSplit, BacSegmentDataset
from semantic_bac_segment.models.pytorch_cnnunet import Unet
from monai.transforms import LoadImageD
from monai.data import DataLoader, Dataset
import tifffile
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import PILReader, pad_list_data_collate
import numpy as np
from typing import Tuple, List
import tifffile
from semantic_bac_segment.models.pytorch_altmodel import UNET as Unet_base

class TIFFLoader(object):
    def __init__(self, keys=("image", "label")):
        self.keys = keys

    def __call__(self, data, add_channel_dim=True):
        for key in self.keys:
            if isinstance(data[key], str):
                data[key] = tifffile.imread(data[key])
                if add_channel_dim:
                # Add an extra channel dimension
                    data[key] = np.expand_dims(data[key], axis=0)

            elif isinstance(data[key], np.ndarray):
                # The data is already an array, so no need to load it
                pass
            else:
                print(f'Loading key: {data[key]}')        
                raise ValueError(f"Unsupported data type for key '{key}': {type(data[key])}")
                        
        return data

class FitImaged(object):
    """
    Fit the image to ensure it is divisible by the effective patch size.
    """
    def __init__(self, keys, step_size):
        self.keys = keys
        self.step_size = step_size

    def __call__(self, data):
        for key in self.keys:
            data[key] = self.fit_image(data[key], self.step_size)
        return data
    
    def fit_image(self, img: np.ndarray, step_size: Tuple[int, int]) -> np.ndarray:
        """
        Pad the image to ensure it is divisible by the effective patch size.

        Args:
            img (np.ndarray): Input image array.
            step_size (Tuple[int, int]): Step size for the sliding window.

        Returns:
            np.ndarray: Padded image array.
        """
        # Check if the image has multiple channels
        #print(f'shape image: {img.shape}')
        padded_channels = []
        if img.shape[0] == 3:
            for channel in range(img.shape[0]):

                channel_img = img[channel, :,:]
                padded_channel = self.pad_single_channel(channel_img, step_size)
                padded_channels.append(padded_channel)
            padded_img = np.stack(padded_channels, axis=0)
        else:
            for channel in range(img.shape[0]):
                channel_img = img[channel, :, :]
                padded_channel = self.pad_single_channel(channel_img, step_size)
                padded_channels.append(padded_channel)
            padded_img = np.stack(padded_channels, axis=0)

        return padded_img
    
    def pad_single_channel(self, img: np.ndarray, step_size: Tuple[int, int]) -> np.ndarray:
        """
        Pad a single channel image to ensure it is divisible by the effective patch size.

        Args:
            img (np.ndarray): Input image array.
            step_size (Tuple[int, int]): Step size for the sliding window.

        Returns:
            np.ndarray: Padded image array.
        """
        # Calculate the padding dims
        remainder_h = img.shape[0] % step_size[0]
        remainder_w = img.shape[1] % step_size[1]

        pad_h = step_size[0] - remainder_h if remainder_h else 0
        pad_w = step_size[1] - remainder_w if remainder_w else 0

        # Pad the image
        img = np.pad(img, ((0, pad_h), (0, pad_w)), 
                     mode='symmetric')

        return img
    
class EnsureChannelFirstForLabel(object):
    """
    Ensure that the channel dimension is in the first position for the label images.
    """
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key != "image":
                data[key] = self.ensure_channel_first(data[key])
        return data

    def ensure_channel_first(self, arr: np.ndarray) -> np.ndarray:
        """
        Ensure that the channel dimension is in the first position.
        """
        if arr.ndim == 3:
            return np.transpose(arr, (2, 0, 1))
        else:
            return arr


configs=[
    {
    'modelid' : 'config1',
    'model_fam' : UNet,
    'spatial_dims': 2,
    'in_channels': 1,
    'out_channels': 1,  # Number of output channels (classes)
    'channels': (32, 64, 128, 256),
    'strides': (2, 2, 2, 2),
    'num_res_units': 2,
    },
    {
    'modelid' : 'config2',
    'spatial_dims': 2,
    'in_channels': 1,
    'out_channels': 1,  # Number of output channels (classes)
    'channels': (32, 64, 128, 256, 512),
    'strides': (2, 2, 2, 2),
    'num_res_units': 2,
    },


]

def train(model, modelid, source_folder, mask_folder, saving_folder='.', debugging=False):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    device=get_device()
    # create a temporary directory and 40 random image, mask pairs

    splitter = TrainSplit(source_folder, mask_folder, val_ratio=0.3)
    splitter.get_samplepairs()
    train_pairs, val_pairs = splitter.split_samples()

    # Define the transformations
    transforms_train = Compose([
        #TIFFLoader(keys=["image", "label"]),
        LoadImaged(keys=["image", "label"], reader=PILReader),  # Load both image and metadata
        EnsureChannelFirstd(keys=["image"]),  # Ensure the image has a channel dimension
        EnsureChannelFirstForLabel(keys=["label"]), 
        ScaleIntensityd(keys=["image", "label"], ),  # Scale the intensity of the image to the range [0, 1]
        FitImaged(keys=["image", "label"], step_size=(2400, 2400)),
        ToTensord(keys=["image", "label"])  # Convert the image and label to PyTorch tensors
    ])

    # Create the datasets and data loaders
    train_data = [{"image": image, "label": label} for image, label in train_pairs]
    val_data = [{"image": image, "label": label} for image, label in val_pairs]
    train_dataset = Dataset(data=train_data, transform=transforms_train)
    val_dataset = Dataset(data=val_data, transform=transforms_train)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=pad_list_data_collate)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=pad_list_data_collate)

    # Define the U-Net model

    # Define the loss function and metric
    loss_function = DiceLoss(to_onehot_y=False, sigmoid=False)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Define the optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # Training loop
    num_epochs = 50
    best_val_loss=10000
    print(f'training model {modelid}')
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch}')
        model.train()
        train_loss = 0
        train_dice = 0
        for batch_data in train_loader:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()

            if debugging:
                tensor_debugger(inputs, 'inputs')

            outputs = model(inputs)
            outputs=torch.sigmoid(outputs)

            if debugging:
                tensor_debugger(outputs, 'outputs')
                tensor_debugger(labels, 'labels')

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_dice += dice_metric(outputs, labels)
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            for batch_data in val_loader:
                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                outputs = model(inputs)
                
                outputs=torch.sigmoid(outputs)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()
                val_dice += dice_metric(outputs, labels)
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        scheduler.step()

        # Save the model with the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(saving_folder, f"unet_model_best-binary2-{modelid}.pth"))
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(saving_folder, f"unet_model_last-{modelid}.pth"))


if __name__ == "__main__":

#    for config_i in configs:
    #model=atomunet(layers=[2, 2, 2, 3]).to(device)
    #model=Unet_base(features=[32, 64, 128, 256]).to(device)

    device=get_device()

    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=3,  # Number of output channels (classes)
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)



    torch.compile(model)
    source_folder='/Users/santiago/switchdrive/boeck_lab_projects/Semantic_bac_segment/data/source_norm2/'
    mask_folder='/Users/santiago/switchdrive/boeck_lab_projects/Semantic_bac_segment/data/multiclass_masks/'
    
    print(f'masks folder exist: {os.path.exists(mask_folder)}')
    train(model, source_folder, mask_folder,'channel1-basemodel', debugging=False)

    