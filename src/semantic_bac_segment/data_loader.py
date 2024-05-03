import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 


import numpy as np
import tifffile
import glob
from semantic_bac_segment.utils import normalize_percentile, normalize_min_max
from semantic_bac_segment.image_augment import ImageAugmenter
from semantic_bac_segment.pre_processing import ImageAdapter
import torch
from torch.utils.data.dataset import Dataset
import os
import tifffile
import random

import os
import numpy as np
import glob
class TrainSplit:
    def __init__(self, image_path, mask_path, val_ratio=0.1):
        self.image_path = image_path
        self.mask_path = mask_path
        self.val_ratio = val_ratio
        
    def get_samplepairs(self):
        assert os.path.exists(self.image_path), "Image directory does not exist"
        assert os.path.exists(self.mask_path), "Mask directory does not exist"

        image_files = sorted(glob.glob(os.path.join(self.image_path, '*')))
        mask_files = sorted(glob.glob(os.path.join(self.mask_path, '*')))

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

class BacSegmentDataset(Dataset):
    def __init__(self, 
                 image_pairs, 
                 in_channels=1, 
                 out_channels=1, 
                 mode='train', 
                 patch_size=512, 
                 overlap_ratio=0.0, 
                 subsetting=0, 
                 filter_threshold=None,
                 precision='half'):
                
        self.image_pairs=image_pairs
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels= out_channels 
        self.mode = mode
        self.filter_threshold = filter_threshold
        self.overlap_ratio = overlap_ratio
        self.precision = precision

        if isinstance(subsetting, int):
            self.subsetting = subsetting
        elif isinstance(subsetting, float) and 0 < subsetting < 1:
            self.subsetting = int(len(self.image_pairs) * subsetting)
        else:
            raise ValueError("Invalid value for subsetting. It should be an integer or a fraction between 0 and 1.")

    def __getitem__(self, index):
        img_path, mask_path = self.image_pairs[index]
        img = tifffile.imread(img_path)
        mask = tifffile.imread(mask_path)

        if self.mode == 'train':
            augmenter = ImageAugmenter(seed=None)
            aug_img, aug_mask = augmenter.transform(img, mask)

            prepare_img = ImageAdapter(aug_img, self.patch_size, self.overlap_ratio)
            prepare_mask = ImageAdapter(aug_mask, self.patch_size, self.overlap_ratio)
            img_patches = prepare_img.create_patches()
            mask_patches = prepare_mask.create_patches()

            if self.filter_threshold:
                high_prop_mask = self.filter_samples(mask_patches, self.filter_threshold)
                img_patches = img_patches[high_prop_mask]
                mask_patches = mask_patches[high_prop_mask]

            if self.subsetting:
                if self.subsetting > img_patches.shape[0]:
                    subset_size = img_patches.shape[0]
                else:
                    subset_size = self.subsetting
                random_subset = np.random.choice(img_patches.shape[0], size=subset_size, replace=False)
                img_patches = img_patches[random_subset]
                mask_patches = mask_patches[random_subset]

        else:
            prepare_img = ImageAdapter(img, self.patch_size, self.overlap_ratio)
            prepare_mask = ImageAdapter(mask, self.patch_size, self.overlap_ratio)
            img_patches = prepare_img.create_patches()
            mask_patches = prepare_mask.create_patches()


        normalized_images = np.empty_like(img_patches, dtype=np.float32)
        normalized_masks = np.empty_like(mask_patches, dtype=np.float32)

        for i in range(img_patches.shape[0]):
            normalized_images[i] = normalize_percentile(img_patches[i])
            normalized_masks[i] = normalize_min_max(mask_patches[i])

        # convert to tensors
        torch_img = torch.from_numpy(normalized_images).float()
        torch_mask = torch.from_numpy(normalized_masks).float()
                
        return torch_img, torch_mask

    def filter_samples(self, mask_patches, threshold):
        # Filter out samples with no bacteria
        
        high_prop_mask=np.where(np.mean(mask_patches, axis=(1,2)) > threshold)[0]
        return high_prop_mask
    
    def __len__(self):
        return len(self.image_pairs)


#def collate_fn(batch):
#    # Unzip the batch
#    images, masks = zip(*batch)
#    
#    # Stack the images and masks along a new dimension
#    images = torch.stack(images, dim=0)
#    masks = torch.stack(masks, dim=0)
#
#    # Get the number of crops and the batch size
#    num_crops = images.shape[1]
#    batch_size = images.shape[0]
#    
#    # Calculate total number of samples
#    total_samples = num_crops * batch_size
#    
#    # Reshape the images and masks to combine crops with the total number of samples
#    images = images.view(total_samples, 1, images.shape[-2], images.shape[-1])
#    masks = masks.view(total_samples, 1, masks.shape[-2], masks.shape[-1])
#
#    return images, masks

from semantic_bac_segment.utils import tensor_debugger
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
    
    # Determine the number of dimensions (2D or 3D)
    num_dims_images = images.ndim - 2
    num_dims_masks = masks.ndim - 2
    
    # Reshape the images and masks to combine crops with the total number of samples
    if num_dims_images == 2 and num_dims_masks == 2:
        images = images.view(total_samples, images.shape[-3], images.shape[-2], images.shape[-1])
        masks = masks.view(total_samples, 1, masks.shape[-2], masks.shape[-1])
    elif num_dims_images == 2 and num_dims_masks == 3:
        images = images.view(total_samples, 1, images.shape[-2], images.shape[-1])
        masks = masks.view(total_samples, masks.shape[-3], masks.shape[-2], masks.shape[-1])
    elif num_dims_images == 3 and num_dims_masks == 2:
        images = images.view(total_samples, images.shape[-3], images.shape[-2], images.shape[-1])
        masks = masks.view(total_samples, 1, masks.shape[-2], masks.shape[-1])
    elif num_dims_images == 3 and num_dims_masks == 3:
        images = images.view(total_samples, images.shape[-3], images.shape[-2], images.shape[-1])
        masks = masks.view(total_samples, masks.shape[-3], masks.shape[-2], masks.shape[-1])
    else:
        raise ValueError("Unsupported combination of image and mask dimensions.")

    return images, masks
