import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import glob
from typing import List, Tuple
from torch.utils.data.dataset import Dataset
from monai.data import Dataset, PatchDataset, DataLoader
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
    Class for creating datasets and patches for bacterial segmentation.

    Args:
        source_folder (str): Path to the directory containing the source images.
        mask_folder (str): Path to the directory containing the mask images.
        val_ratio (float, optional): Ratio of samples to use for validation. Defaults to 0.3.
    """

    def __init__(self, source_folder: str, mask_folder: str, val_ratio: float = 0.3, batch_size : int = None):
        self.source_folder = source_folder
        self.mask_folder = mask_folder
        assert (
            0 <= val_ratio <= 1
        ), f"Validation ratio must be between 0 and 1, but got {val_ratio}"
        self.val_ratio = val_ratio
        self.batch_size = batch_size


    def create_datasets(
        self, train_transform, val_transform
    ) -> Tuple[Dataset, Dataset]:
        """
        Creates training and validation datasets.

        Args:
            train_transform: Transformation to apply to the training dataset.
            val_transform: Transformation to apply to the validation dataset.

        Returns:
            Tuple[Dataset, Dataset]: Tuple containing the training and validation datasets.
        """

        splitter = TrainSplit(
            self.source_folder, self.mask_folder, val_ratio=self.val_ratio
        )
        splitter.get_samplepairs()
        train_pairs, val_pairs = splitter.split_samples()

        train_data = [{"image": image, "label": label} for image, label in train_pairs]
        val_data = [{"image": image, "label": label} for image, label in val_pairs]

        self.train_dataset = Dataset(data=train_data, transform=train_transform)
        self.val_dataset = Dataset(data=val_data, transform=val_transform)

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
