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


class BacSegmentDataset(Dataset):
    def __init__(self, 
                 image_path, 
                 mask_path, 
                 in_channels=1, 
                 out_channels=1, 
                 mode='train', 
                 patch_size=512, 
                 overlap_ratio=0.0, 
                 subsetting=0, 
                 filter_threshold=None,
                 precision='half'):
        
        assert os.path.exists(image_path), "Image directory does not exist"
        assert os.path.exists(mask_path), "Mask directory does not exist"

        self.image_files = sorted(glob.glob(os.path.join(image_path, '*')))
        self.mask_files = sorted(glob.glob(os.path.join(mask_path, '*')))

        assert len(self.image_files) == len(self.mask_files), "Number of images and masks do not match"
        
        
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
            self.subsetting = int(len(self.image_files) * subsetting)
        else:
            raise ValueError("Invalid value for subsetting. It should be an integer or a fraction between 0 and 1.")

    def __getitem__(self, index):
        img = tifffile.imread(self.image_files[index])
        mask = tifffile.imread(self.mask_files[index])

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
                random_subset = np.random.choice(img_patches.shape[0], size=self.subsetting, replace=False)
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

        if self.precision == 'half':
            torch_img = torch.from_numpy(normalized_images).float().half()
            torch_mask = torch.from_numpy(normalized_masks).float().half()
        else:
            torch_img = torch.from_numpy(normalized_images).float()
            torch_mask = torch.from_numpy(normalized_masks).float()
                
        return torch_img, torch_mask

    def filter_samples(self, mask_patches, threshold):
        # Filter out samples with no bacteria
        
        high_prop_mask=np.where(np.mean(mask_patches, axis=(1,2)) > threshold)[0]
        return high_prop_mask
    
    def __len__(self):
        return len(self.image_files)


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
