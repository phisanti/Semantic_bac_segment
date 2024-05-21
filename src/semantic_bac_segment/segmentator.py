import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import gc
from typing import Any
import numpy as np
from semantic_bac_segment.pre_processing import ImageAdapter
from semantic_bac_segment.utils import normalize_percentile, get_device, empty_gpu_cache
from monai.inferers import SlidingWindowInferer


class Segmentator:
    """
    A class representing a segmentation model.

    Attributes:
        model (torch.nn.Module): The segmentation model.
        device (torch.device): The device on which the model is loaded.
    """

    def __init__(
        self,
        model_path: str,
        model_graph: torch.nn.Module,
        patch_size: int,
        overlap_ratio: float,
        half_precision: bool = False,
    ) -> None:
        """
        Initializes a Segmentator object.

        Args:
            model_path (str): The path to the saved model weights.
            model_graph (torch.nn.Module): The model architecture.
            patch_size (int): The size of the patches for sliding window inference.
            overlap_ratio (float): The overlap ratio between patches for sliding window inference.
            half_precision (bool, optional): Whether to use half-precision (FP16) for inference. Defaults to False.
        """

        assert (
            isinstance(patch_size, int) and patch_size > 0
        ), "patch_size must be a positive integer"
        assert (
            isinstance(overlap_ratio, float) and 0 <= overlap_ratio < 1
        ), "overlap_ratio must be a float between 0 and 1"

        self.device = get_device()
        self.model = self.get_model(model_path, self.device, model_graph=model_graph)
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio
        self.model.eval()
        self.half_precision = half_precision
        if self.half_precision:
            self.model.half()
        else:
            self.model.float()

    def predict(self, image: np.ndarray, is_3D: bool=False, sigmoid: bool=True, **kwargs: Any) -> np.ndarray:
        """
        Predicts the segmentation mask for the given image. It can handle 2D images or a stack of 2D images.
        This method connects the model with the monai SlidingWindowInferer to perform inference on patches of the input image.
        It also takes care the normalisation and the conversion to tensor.

        Args:
            image (numpy.ndarray): The input image or image stack.
            **kwargs: Additional keyword arguments to pass to the SlidingWindowInferer.

        Returns:
            numpy.ndarray: The segmentation mask or stack of segmentation masks.
        """

        # Prepare image
        original_shape = image.ndim
        image = self.ensure_4d(image, is_3D)
        image = self.normalize_percentile_batch(image)
        img_tensor = torch.from_numpy(image).to(self.device)
        if self.half_precision:
            img_tensor = img_tensor.half()  # Convert input to half-precision
        else:
            img_tensor = img_tensor.float()

        # Create SlidingWindowInferer
        inferer = SlidingWindowInferer(
            roi_size=self.patch_size, overlap=self.overlap_ratio, **kwargs
        )

        with torch.no_grad():
            output_mask = inferer(img_tensor, self.model)

            output_mask = output_mask.cpu().numpy()

        # Free up tensors
        del img_tensor, image
        gc.collect()
        empty_gpu_cache(self.device)

        if sigmoid:
            output_mask = self.sigmoid(output_mask)

        return output_mask  

    def get_model(
        self, path: str, device: torch.device, model_graph: torch.nn.Module = None
    ) -> torch.nn.Module:
        """
        Loads a model from the specified path and returns it.

        Args:
            path (str): The path to the model file.
            device (str): The device to load the model onto.
            model_graph (Optional[torch.nn.Module]): An optional pre-initialized model graph.

        Returns:
            torch.nn.Module: The loaded model.

        Raises:
            FileNotFoundError: If the model file is not found at the specified path.
            RuntimeError: If an error occurs while loading the model.
            Exception: If an unexpected error occurs.
        """
        try:
            if model_graph is None:
                model = torch.load(path, map_location=device)
            else:
                model = model_graph
                state_dict = torch.load(path, map_location=device)

                # Check if the loaded state dictionary is compatible with the model architecture
                if not set(state_dict.keys()).issubset(set(model.state_dict().keys())):
                    raise ValueError(
                        "Loaded state dictionary does not match the model architecture."
                    )

                model.load_state_dict(state_dict)

            model.to(device)
            torch.compile(model, mode="max-autotune")

            return model

        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at path: {path}")

        except RuntimeError as e:
            raise RuntimeError(f"Error occurred while loading the model: {str(e)}")

        except Exception as e:
            raise Exception(f"Unexpected error occurred: {str(e)}")

    def ensure_4d(self, img: np.ndarray, is_3D: bool) -> np.ndarray:
        """
        Ensures that the input image has 4 dimensions (batch, channel, height, width).
        This is the standard format expected by PyTorch models.

        Args:
            img (numpy.ndarray): The input image.
            is_3D (bool): Wether the image is a 3D volume or a 2D image.

        Returns:
            numpy.ndarray: The image with 4 dimensions.
        """
        if img.ndim == 2:
            # Add channel and batch dimensions if the array is 2D
            img = np.expand_dims(img, axis=(0, 1))
        elif img.ndim == 3 and is_3D:
            # Add a channel dimension if the array is multi-stack
            img = np.expand_dims(img, axis=0)
        elif img.ndim == 3 and not is_3D:
            # Add a batch dimension if the array is multi-channel
            img = np.expand_dims(img, axis=1)

        elif img.ndim == 4:
            # No modification needed if the array is already 4D
            pass
        else:
            raise ValueError(
                f"Unsupported array dimensions: {img.ndim}, current shape is {img.shape}"
            )

        return img

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def normalize_percentile_batch(
        images: np.ndarray,
        pmin: float = 1,
        pmax: float = 99.8,
        clip: bool = True,
        dtype: np.dtype = np.float32,
    ) -> np.ndarray:
        """
        Percentile-based image normalization for a batch of images.

        Args:
            images (numpy.ndarray): Input array of shape (batch_size, height, width).
            pmin (float): Lower percentile value (default: 1).
            pmax (float): Upper percentile value (default: 99.8).
            clip (bool): Whether to clip the output values to the range [0, 1] (default: False).
            dtype (numpy.dtype): Output data type (default: np.float32).

        Returns:
            numpy.ndarray: Normalized array of shape (batch_size, height, width).
        """
        images = images.astype(dtype, copy=False)
        mi = np.percentile(images, pmin, axis=(2, 3), keepdims=True)
        ma = np.percentile(images, pmax, axis=(2, 3), keepdims=True)
        eps = np.finfo(dtype).eps  # Get the smallest positive value for the data type

        images = (images - mi) / (ma - mi + eps)

        if clip:
            images = np.clip(images, 0, 1)

        # Force output type
        images = images.astype(dtype, copy=False)
        
        return images