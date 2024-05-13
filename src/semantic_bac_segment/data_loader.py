import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

import torch
import numpy as np
import glob
from typing import Tuple
from torch.utils.data.dataset import Dataset
from monai.data import Dataset, PatchDataset, DataLoader
from monai.transforms import RandSpatialCropSamplesd

class TrainSplit:
    def __init__(self, image_path, mask_path, filetype='.tiff', val_ratio=0.1):
        self.image_path = image_path
        self.mask_path = mask_path
        self.val_ratio = val_ratio
        self.filetype = filetype
    def get_samplepairs(self):
        assert os.path.exists(self.image_path), "Image directory does not exist"
        assert os.path.exists(self.mask_path), "Mask directory does not exist"

        image_files = sorted(glob.glob(os.path.join(self.image_path, '*'+ self.filetype)))
        mask_files = sorted(glob.glob(os.path.join(self.mask_path, '*'+ self.filetype)))

        assert len(image_files) == len(mask_files), "Number of images and masks do not match"
        self.image_mask_pairs = list(zip(image_files, mask_files))

        return self.image_mask_pairs

    def split_samples(self, verbose=True):
        train_dicts, valid_dicts = self.image_mask_pairs, []
        if self.val_ratio > 0:

            # Obtain & shuffle data indices
            num_data_dicts = len(self.image_mask_pairs)
            indices = np.arange(num_data_dicts)
            np.random.shuffle(indices)

            # Divide train/valid indices by the proportion
            valid_size = int(num_data_dicts * self.val_ratio)
            train_indices = indices[valid_size:]
            valid_indices = indices[:valid_size]

            # Assign data dicts by split indices
            train_dicts = [self.image_mask_pairs[idx] for idx in train_indices]
            valid_dicts = [self.image_mask_pairs[idx] for idx in valid_indices]
            
            if verbose:
                print(f"\n(DataLoaded) Training data size: {len(train_dicts)}, Validation data size: {len(valid_dicts)}\n")

        return train_dicts, valid_dicts

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

def collate_fn(batch):
    # Unzip the batch
    images, masks = zip(*batch)
    
    # Stack the images and masks along a new dimension
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)

    # Get the number of crops and the batch size
    num_crops = images.shape[1]
    batch_size = images.shape[0]
    
    # Calculate total number of samples
    total_samples = num_crops * batch_size
    
    # Reshape the images and masks to combine crops with the total number of samples
    images = images.view(total_samples, 1, images.shape[-2], images.shape[-1])
    masks = masks.view(total_samples, 1, masks.shape[-2], masks.shape[-1])

    return images, masks