from typing import Optional
from logging import Logger
import tifffile
import numpy as np
import torch


def get_bit_depth(img: np.ndarray) -> int:
    """
    Get the bit depth of an image based on its data type.

    Args:
        img (numpy.ndarray): Input image.

    Returns:
        int: Bit depth of the image.
    """
    dtype_to_bit_depth = {
        "uint8": 8,
        "uint16": 16,
        "uint32": 32,
        "uint64": 64,
        "int8": 8,
        "int16": 16,
        "int32": 32,
        "int64": 64,
        "float32": 32,
        "float64": 64,
    }

    bit_depth = dtype_to_bit_depth[str(img.dtype)]

    return bit_depth


def invert_image(img: np.ndarray, bit_depth: int) -> np.ndarray:
    """
    Invert the pixel values of an image based on its bit depth.

    Args:
        img (numpy.ndarray): Input image.
        bit_depth (int): Bit depth of the image.

    Returns:
        numpy.ndarray: Inverted image.
    """
    bit_depth = bit_depth
    inverted_image = 2**bit_depth - 1 - img

    return inverted_image


def normalize_percentile(
    x: np.ndarray,
    pmin: float = 1,
    pmax: float = 99.8,
    clip: bool = False,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Percentile-based image normalization.

    Args:
        x (numpy.ndarray): Input array.
        pmin (float): Lower percentile value (default: 1).
        pmax (float): Upper percentile value (default: 99.8).
        clip (bool): Whether to clip the output values to the range [0, 1] (default: False).
        dtype (numpy.dtype): Output data type (default: np.float32).

    Returns:
        numpy.ndarray: Normalized array.
    """

    x = x.astype(dtype, copy=False)
    mi = np.percentile(x, pmin)
    ma = np.percentile(x, pmax)
    eps = np.finfo(dtype).eps  # Get the smallest positive value for the data type

    x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)

    return x


def empty_gpu_cache(device: torch.device) -> None:
    """
    Clear the GPU cache.

    Args:
        device (torch.device): The device to clear the cache for.
    """
    # Clear the GPU cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def get_device() -> torch.device:
    """
    Detects the available GPU device.

    Returns:
        torch.device: The device to be used for inference.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def tensor_debugger(
        tensor: torch.Tensor, 
        name: str = "tensor", 
        logger: Optional[Logger] = None) -> None:
    """
    Print or log debugging information about a tensor.

    Args:
        tensor (torch.Tensor): The tensor to debug.
        name (str): The name of the tensor (default: "tensor").
        logger (logging.Logger, optional): The logger to use for logging (default: None).
    """
    messages = [
        f"{name} shape: {tensor.shape}",
        f"nans {name} max: {torch.isnan(tensor).any()}",
        f"inf {name} max: {torch.isinf(tensor).any()}",
        f"{name} max: {tensor.max()}",
        f"{name} min: {tensor.min()}",
    ]

    for message in messages:
        if logger is not None and logger.is_level("DEBUG"):
            logger.log(message, level="DEBUG")
        else:
            print(message)


def write_tensor_debug(tensor, prefix, path, id):
    # Write input image
    print(id)
    tensor_img = tensor.detach().cpu().numpy()
    tensor_img = (tensor_img * 255).astype(np.uint8)
    tifffile.imwrite(f'{path}{prefix}_{id}.tiff', tensor_img)
