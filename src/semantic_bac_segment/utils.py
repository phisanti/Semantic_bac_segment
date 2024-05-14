import numpy as np
import itertools
import torch


def get_bit_depth(img):
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


def invert_image(img, bit_depth):
    bit_depth = bit_depth
    inverted_image = 2**bit_depth - 1 - img

    return inverted_image


def normalize_percentile(x, pmin=1, pmax=99.8, clip=False, dtype=np.float32):
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


def empty_gpu_cache(device):
    # Clear the GPU cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def get_device():
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


def tensor_debugger(tensor, name="tensor", logger=None):
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
