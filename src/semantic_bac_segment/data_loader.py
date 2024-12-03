import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import glob
from typing import Optional, Union, Tuple, Dict, Any, List
from torch.utils.data.dataset import Dataset
from monai.data import Dataset, PatchDataset, DataLoader, SmartCacheDataset
from monai.transforms import RandSpatialCropSamplesd


class TrainSplit:
    """
    Class for splitting image-mask pairs into training and validation sets.

    Args:
        image_path (str): Path to the directory containing the images.
        mask_path (str): Path to the directory containing the masks.
        filetype (str, optional): File extension of the images and masks. Defaults to '.tiff'.
        val_ratio (float, optional): Ratio of samples to use for validation. Defaults to 0.1.
    """

    def __init__(
        self,
        image_path: str,
        mask_path: str,
        filetype: str = ".tiff",
        val_ratio: float = 0.1,
    ):
        self.image_path = image_path
        self.mask_path = mask_path
        self.val_ratio = val_ratio
        self.filetype = filetype

    def get_samplepairs(self) -> List[Tuple[str, str]]:
        """
        Retrieves the image-mask pairs from the specified directories.

        Returns:
            List[Tuple[str, str]]: List of tuples containing image-mask pairs.
        """
        assert os.path.exists(self.image_path), "Image directory does not exist"
        assert os.path.exists(self.mask_path), "Mask directory does not exist"

        image_files = sorted(
            glob.glob(os.path.join(self.image_path, "*" + self.filetype))
        )
        mask_files = sorted(
            glob.glob(os.path.join(self.mask_path, "*" + self.filetype))
        )

        assert len(image_files) == len(
            mask_files
        ), "Number of images and masks do not match"
        self.image_mask_pairs = list(zip(image_files, mask_files))

        return self.image_mask_pairs

    def split_samples(
        self, verbose: bool = True
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Splits the image-mask pairs into training and validation sets.

        Args:
            verbose (bool, optional): Whether to print the sizes of the training and validation sets. Defaults to True.

        Returns:
            Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]: Tuple containing the training and validation image-mask pairs.
        """

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
                print(
                    f"\n(DataLoaded) Training data size: {len(train_dicts)}, Validation data size: {len(valid_dicts)}\n"
                )

        return train_dicts, valid_dicts


class BacSegmentDatasetCreator:
    """
    Class for creating datasets and patches for bacterial segmentation with optional subsampling
    and smart caching capabilities.

    Args:
        source_folder (str): Path to the directory containing the source images
        mask_folder (str): Path to the directory containing the mask images
        val_ratio (float, optional): Ratio of samples to use for validation. Must be between 0 and 1. Defaults to 0.3
        batch_size (Optional[int], optional): Batch size for data loading. Defaults to None
        nsamples (Optional[Union[int, float]], optional): Number of samples or fraction to use. 
            If float, must be between 0-1. If int, must be positive. Defaults to None
        use_smart_cache (bool, optional): Whether to use MONAI's SmartCacheDataset for faster iterations. 
            Defaults to False
    """

    def __init__(
        self, 
        source_folder: str,
        mask_folder: str,
        val_ratio: float = 0.3,
        batch_size: Optional[int] = None,
        nsamples: Optional[Union[int, float]] = None,
        use_smart_cache: bool = False
    ):
        assert 0 <= val_ratio <= 1, f"Validation ratio must be between 0 and 1, got {val_ratio}"
        if nsamples is not None and nsamples != 0:
            if isinstance(nsamples, float):
                assert 0 < nsamples <= 1, f"Float nsamples must be between 0 and 1, got {nsamples}"
            else:
                assert isinstance(nsamples, int) and nsamples > 0, f"Integer nsamples must be positive, got {nsamples}"

        self.source_folder = source_folder
        self.mask_folder = mask_folder
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.nsamples = None if nsamples == 0 else nsamples
        self.use_smart_cache = use_smart_cache

    def _subsample_pairs(self, data_pairs: List[Tuple[str, str]], n_samples: Optional[Union[int, float]] = None) -> List[Tuple[str, str]]:
        """Subsample data pairs based on nsamples parameter."""
        if n_samples is None:
            return data_pairs
            
        n_total = len(data_pairs)
        sample_size = int(n_total * n_samples) if isinstance(n_samples, float) else min(n_samples, n_total)
        indices = np.random.choice(n_total, sample_size, replace=False)
        return [data_pairs[i] for i in indices]

    def create_datasets(
        self, 
        train_transform: Any, 
        val_transform: Any,
        **kwargs
        ) -> Tuple[Union[Dataset, SmartCacheDataset], Union[Dataset, SmartCacheDataset]]:
        """
        Creates training and validation datasets.

        Args:
            train_transform: Transformation to apply to the training dataset.
            val_transform: Transformation to apply to the validation dataset.
            kwargs: Keyword arguments passed to the SmartCacheDataset monai class. Only used when use_smart_cache=True.
        Returns:
            Tuple[Dataset, Dataset]: Tuple containing the training and validation datasets.
                
        Raises:
            ValueError: If kwargs are provided when use_smart_cache is False
        """
        if kwargs and not self.use_smart_cache:
            raise ValueError("Additional arguments can only be used with SmartCacheDataset (use_smart_cache=True)")


        splitter = TrainSplit(self.source_folder, self.mask_folder, val_ratio=self.val_ratio)
        splitter.get_samplepairs()
        train_pairs, val_pairs = splitter.split_samples()

        # Apply subsampling
        train_pairs = self._subsample_pairs(train_pairs, self.nsamples)
        val_pairs = self._subsample_pairs(val_pairs, self.nsamples)

        train_data = [{"image": image, "label": label} for image, label in train_pairs]
        val_data = [{"image": image, "label": label} for image, label in val_pairs]

        dataset_class = SmartCacheDataset if self.use_smart_cache else Dataset
        
        self.train_dataset = dataset_class(data=train_data, transform=train_transform, **kwargs)
        self.val_dataset = dataset_class(data=val_data, transform=val_transform, **kwargs)

        return self.train_dataset, self.val_dataset

    def create_dataloaders(self, **kwargs) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and validation DataLoaders.
        This method initializes DataLoader objects for both the training and validation datasets
        using the specified batch size. The training DataLoader shuffles the data, while the 
        validation DataLoader does not.
        Returns:
            Tuple[DataLoader, DataLoader]: A tuple containing the training DataLoader and the 
            validation DataLoader.
        """
        
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader
    
    def create_patches(
        self,
        roi_size: Tuple[int, int] = (256, 256),
        num_samples: int = 20,
        train_transforms=None,
        val_transforms=None,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Creates training and validation patch datasets.

        Args:
            roi_size (Tuple[int, int], optional): Size of the region of interest (ROI) for patch extraction. Defaults to (256, 256).
            num_samples (int, optional): Number of patches to extract per image. Defaults to 20.
            train_transforms: Transformation to apply to the training patches.
            val_transforms: Transformation to apply to the validation patches.

        Returns:
            Tuple[DataLoader, DataLoader]: Tuple containing the training and validation patch data loaders.
        """

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

        train_ds = DataLoader(train_patch_dataset, batch_size=num_samples)

        val_ds = DataLoader(val_patch_dataset, batch_size=num_samples)

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
