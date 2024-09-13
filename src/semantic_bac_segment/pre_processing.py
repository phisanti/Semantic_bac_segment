from pathlib import Path
import sys
import numpy as np
import cv2
from itertools import product
from typing import Tuple, List
from skimage.util import view_as_windows


class ImageAdapter:
    def __init__(self, img: np.ndarray, patch_size: int, overlap_ratio: float) -> None:
        """
        Initialize the ImageAdapter class. Adapt images to the U-net input and
        allow to stich them back together to the original shape.

        Args:
            img (np.ndarray): Input image array.
            patch_size (int): Width and height of square patches.
            overlap_ratio (float): Fraction of pixels to overlap.
        """
        self.img = img
        self.source_shape = img.shape
        self.patch_size = patch_size
        self.overlap_size = int(patch_size * overlap_ratio)

    def create_patches(self) -> np.ndarray:
        """
        Split the image into patches using view_as_windows.

        Returns:
            np.ndarray: Array of image patches.
        """

        step_size = (
            self.patch_size - self.overlap_size,
            self.patch_size - self.overlap_size,
        )
        img = self.fit_image(self.img, step_size)
        image_patches = view_as_windows(
            img, (self.patch_size, self.patch_size), step_size
        )

        # Flatten into patches
        self.n_patches_w = image_patches.shape[1]
        self.n_patches_h = image_patches.shape[0]
        image_patches = image_patches.reshape(-1, self.patch_size, self.patch_size)

        return image_patches

    def stich_patches(self, image_patches: np.ndarray) -> np.ndarray:
        """
        Stitch the image patches back into a single image.

        Args:
            image_patches (np.ndarray): Array of image patches.

        Returns:
            np.ndarray: Reconstructed image array.
        """

        overlap_size = self.overlap_size

        reconstructed = np.zeros(
            (
                self.n_patches_h * (self.patch_size - overlap_size) + self.overlap_size,
                self.n_patches_w * (self.patch_size - overlap_size) + self.overlap_size,
            )
        )
        reconstructed_slices = [
            np.zeros_like(reconstructed) for _ in range(len(image_patches))
        ]

        idx_table = product(range(self.n_patches_h), range(self.n_patches_w))

        for n, (i, j) in enumerate(idx_table):
            patch = image_patches[n, :, :]
            h_idx = i * (self.patch_size - overlap_size)
            w_idx = j * (self.patch_size - overlap_size)
            reconstructed_slices[n][
                h_idx : h_idx + self.patch_size, w_idx : w_idx + self.patch_size
            ] = patch

        reconstructed_slices = np.asarray(reconstructed_slices)
        reconstructed = np.max(reconstructed_slices, axis=0)

        # Crop the reconstructed image to the original size
        reconstructed = reconstructed[: self.source_shape[0], : self.source_shape[1]]

        return reconstructed

    def fit_image(self, img: np.ndarray, step_size: Tuple[int, int]) -> np.ndarray:
        """
        Pad the image to ensure it is divisible by the effective patch size.

        Args:
            img (np.ndarray): Input image array.
            step_size (Tuple[int, int]): Step size for the sliding window.

        Returns:
            np.ndarray: Padded image array.
        """
        # Calculate the padding dims
        remainder_h = img.shape[0] % step_size[0]
        remainder_w = img.shape[1] % step_size[1]

        pad_h = 2 * step_size[0] - remainder_h if remainder_h else 0
        pad_w = 2 * step_size[1] - remainder_w if remainder_w else 0

        # Pad the image
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="symmetric")

        return img

    def clear_background(self, sigma_r=25, method="divide", convert_32=True):
        # Input checks
        img = self.img.copy()

        if img.ndim != 2:
            raise ValueError("Input image must be 2D")
        if convert_32:
            img = img.astype(np.float32)

        def round_to_odd(number):
            return int(number) if number % 2 == 1 else int(number) + 1

        # Gaussian blur
        sigma_r = round_to_odd(sigma_r)
        gaussian_blur = cv2.GaussianBlur(img, (sigma_r, sigma_r), 0)

        # Background remove
        if method == "subtract":
            background_removed = cv2.subtract(img, gaussian_blur)
        elif method == "divide":
            background_removed = cv2.divide(img, gaussian_blur)
        else:
            raise ValueError("Invalid method. Choose either 'subtract' or 'divide'")

        self.img = background_removed
