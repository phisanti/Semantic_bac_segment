import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
import cv2
import torch
import tifffile
import numpy as np
from typing import Tuple
from monai.data import Dataset, PatchDataset, DataLoader
from monai.transforms import RandSpatialCropSamplesd
from semantic_bac_segment.data_loader import TrainSplit, collate_fn
from semantic_bac_segment.utils import tensor_debugger


class ComposeInspect(object):
    def __init__(self, keys=("image", "label"), step=''):
        self.keys = keys
        self.step = step

    def __call__(self, data):
        for key in self.keys:
            if isinstance(data[key], torch.Tensor):
                tensor_debugger(data[key], key+self.step)
            elif isinstance(data[key], np.ndarray):
                x = torch.from_numpy(data[key].copy())
                tensor_debugger(x, key+self.step)
            else:
                print(f'Loading key: {data[key]}')        
                        
        return data


class TIFFLoader(object):
    def __init__(self, keys=("image", "label"), add_channel_dim=True):
        self.keys = keys
        self.add_channel_dim = add_channel_dim

    def __call__(self, data):
        for key in self.keys:
            if isinstance(data[key], str):
                data[key] = tifffile.imread(data[key])
                if self.add_channel_dim:
                # Add an extra channel dimension
                    data[key] = np.expand_dims(data[key], axis=0)

            elif isinstance(data[key], np.ndarray):
                # The data is already an array, so no need to load it
                pass
            else:
                print(f'Loading key: {data[key]}')        
                raise ValueError(f"Unsupported data type for key '{key}': {type(data[key])}")
                        
        return data


class ClearBackgroundTransform(object):
    def __init__(self, keys, sigma_r=25, method='divide', convert_32=True):
        self.keys = keys
        self.sigma_r = sigma_r
        self.method = method
        self.convert_32 = convert_32

    def __call__(self, data):
        for key in self.keys:
            img = data[key].copy()

            if self.convert_32:
                img = img.astype(np.float32)

            if img.ndim == 2:
                img = self.backgroud_remove_2d(img, self.sigma_r, self.method)
            
            elif img.ndim == 3:

                for i in range(img.shape[0]):
                    img[i]=self.backgroud_remove_2d(img[i], self.sigma_r, self.method)
            
            else:
                raise NotImplementedError("Background removal for 3D images is not implemented yet.")
            
            data[key] = img
        return data

    def round_to_odd(self, number):
        return int(number) if number % 2 == 1 else int(number) + 1

    def backgroud_remove_2d(self, img, sigma_r, method):
        # Gaussian blur
        sigma_r = self.round_to_odd(self.sigma_r)
        gaussian_blur = cv2.GaussianBlur(img, (sigma_r, sigma_r), 0)

        # Background remove
        if self.method == 'subtract':
            background_removed = cv2.subtract(img, gaussian_blur)
        elif self.method == 'divide':
            background_removed = cv2.divide(img, gaussian_blur)
        else:
            raise ValueError("Invalid method. Choose either 'subtract' or 'divide'")
        
        return background_removed



class NormalizePercentileTransform(object):
    def __init__(self, keys, pmin=1, pmax=99.8, clip=True):
        self.keys = keys
        self.pmin = pmin
        self.pmax = pmax
        self.clip = clip

    def __call__(self, data):
        for key in self.keys:
            if isinstance(data[key], torch.Tensor):
                data[key] = self.tensor_method(data[key])
            elif isinstance(data[key], np.ndarray):
                data[key] = self.numpy_method(data[key])
            else:
                raise ValueError("Data must be torch.Tensor or numpy.ndarray")

        return data
    
    def tensor_method(self, x):
        x=x.clone()
        x = x.to(torch.float32)
        mi = torch.quantile(x, self.pmin / 100.0)
        ma = torch.quantile(x, self.pmax / 100.0)
        eps = torch.finfo(torch.float32).eps  # Get the smallest positive value for the data type
        x = (x - mi) / (ma - mi + eps)

        if self.clip:
            x = torch.clamp(x, 0, 1)
        return x
    
    def numpy_method(self, x):
        x = x.copy()
        x = x.astype(np.float32, copy=False)
        mi = np.percentile(x, self.pmin)
        ma = np.percentile(x, self.pmax)
        eps = np.finfo(np.float32).eps  # Get the smallest positive value for the data type

        x = (x - mi) / (ma - mi + eps)

        if self.clip:
            x = np.clip(x, 0, 1)

        return x

class Ensure4D(object):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            img = data[key]

            if isinstance(img, torch.Tensor):
                if img.ndim == 2:
                    # Add channel and batch dimensions if the tensor is 2D
                    img = img.unsqueeze(0).unsqueeze(0)
                elif img.ndim == 3:
                    # Add a batch dimension if the tensor is 3D
                    img = img.unsqueeze(0)
                elif img.ndim == 4:
                    # No modification needed if the tensor is already 4D
                    pass
                else:
                    raise ValueError(f"Unsupported tensor dimensions: {img.ndim}")
            elif isinstance(img, np.ndarray):
                if img.ndim == 2:
                    # Add channel and batch dimensions if the array is 2D
                    img = np.expand_dims(img, axis=(0, 1))
                elif img.ndim == 3:
                    # Add a batch dimension if the array is 3D
                    img = np.expand_dims(img, axis=0)
                elif img.ndim == 4:
                    # No modification needed if the array is already 4D
                    pass
                else:
                    raise ValueError(f"Unsupported array dimensions: {img.ndim}")
            else:
                raise TypeError(f"Unsupported data type: {type(img)}")

            data[key] = img

        return data


class BacSegmentDatasetCreator:
    def __init__(self, 
                 source_folder: str, 
                 mask_folder: str, 
                 val_ratio: float = 0.3

                 ):
        self.source_folder = source_folder
        self.mask_folder = mask_folder
        self.val_ratio = val_ratio

    def create_datasets(self, train_transform, val_transform) -> Tuple[PatchDataset, PatchDataset]:
        splitter = TrainSplit(self.source_folder, self.mask_folder, val_ratio=self.val_ratio)
        splitter.get_samplepairs()
        train_pairs, val_pairs = splitter.split_samples()

        train_data = [{"image": image, "label": label} for image, label in train_pairs]
        val_data = [{"image": image, "label": label} for image, label in val_pairs]

        self.train_dataset = Dataset(data=train_data, transform=train_transform)
        self.val_dataset = Dataset(data=val_data, transform=val_transform)

        return self.train_dataset, self.val_dataset

    def create_patches(self, 
                 roi_size: Tuple[int, int] = (256, 256), 
                 num_samples: int = 20,
                 train_transforms=None,
                 val_transforms=None):
        
        patch_func = RandSpatialCropSamplesd(
            keys=["image", "label"],
            roi_size=roi_size,
            num_samples=num_samples,
            random_size=False,
        )

        train_patch_dataset = PatchDataset(
            data=self.train_dataset,
            patch_func=patch_func,
            samples_per_image=num_samples, 
            transform=train_transforms,

        )

        val_patch_dataset = PatchDataset(
            data=self.val_dataset,
            patch_func=patch_func,
            samples_per_image=num_samples,
            transform=val_transforms,

        )
        
        train_ds = DataLoader(train_patch_dataset, 
                              batch_size=num_samples)

        val_ds = DataLoader(val_patch_dataset,
                            batch_size=num_samples)


        return train_ds, val_ds