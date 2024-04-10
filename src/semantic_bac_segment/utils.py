import numpy as np
import itertools
import yaml
import torch

def read_cofig(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def adjust_dimensions(img, dim_order):
    target_order = 'TSCXY'
    missing_dims = set(target_order) - set(dim_order)

    # Add missing dimensions
    for dim in missing_dims:
        index = target_order.index(dim)
        img = np.expand_dims(img, axis=index)
        dim_order = dim_order[:index] + dim + dim_order[index:]

    # Reorder dimensions
    order = [dim_order.index(dim) for dim in target_order]
    img = np.transpose(img, order)

    return img


def stack_indexer(nframes=[0], nslices=[0], nchannels=[0]):

    dimensions =[]
    for dimension in [nframes, nslices, nchannels]:
        if isinstance(dimension, int):
            if dimension < 0:
                raise ValueError("Dimensions must be positive integers or lists.")
            dimensions.append([dimension])
        elif isinstance(dimension, (list, range)):
            if not all(isinstance(i, int) and i >= 0 for i in dimension):
                raise ValueError("All elements in the list dimensions must be positive integers.")
            dimensions.append(dimension)
        else:
            raise TypeError("All dimensions must be either positive integers or lists of positive integers.")

    combinations = list(itertools.product(*dimensions))
    index_table = np.array(combinations)
    return index_table


def unit_converter(value, conversion_factor=1, to_unit='pixel'):
    if to_unit == 'um':
        return value * conversion_factor
    elif to_unit == 'pixel':
        return value / conversion_factor
    else:
        raise ValueError("Invalid unit. Choose either 'um' or 'pixel'.")


def get_bit_depth(img):
    dtype_to_bit_depth = {
        'uint8': 8,
        'uint16': 16,
        'uint32': 32,
        'uint64': 64,
        'int8': 8,
        'int16': 16,
        'int32': 32,
        'int64': 64,
        'float32': 32,
        'float64': 64
    }

    bit_depth = dtype_to_bit_depth[str(img.dtype)]

    return bit_depth


def convert_image(img, dtype):
    # Shift the image intensity range to [0, 255]
    img_shifted = img - np.min(img)
    img_scaled = img_shifted / np.max(img_shifted)
    
    if dtype == np.uint8:
        img_converted = (img_scaled * 255).astype(np.uint8)
    elif dtype == np.uint16:
        img_converted = (img_scaled * 65535).astype(np.uint16)
    elif dtype == np.float16 or dtype == np.float32:
        img_converted = img_scaled.astype(dtype)
    else:
        raise ValueError(f"Unsupported data type: {dtype}")
    
    return img_converted


def invert_image(img, bit_depth):
    
    bit_depth=bit_depth
    inverted_image=2**bit_depth - 1 - img
    
    return inverted_image


def round_to_odd(number):
    return int(number) if number % 2 == 1 else int(number) + 1


def crop_image(img, start_h, h, start_w, w):
    crop = img[start_h:start_h+h, start_w:start_w+w]
    return crop


def make_binary(image):
    """
    Args:
        image : numpy array of image in datatype uint8
    Return :
        image : numpy array of image in datatype uint8 only with 255 and 0
    """
    if image.dtype != np.uint8:
        raise ValueError("Input image must be of type uint8")
    
    image[image > 127.5] = 255
    image[image < 127.5] = 0
    image = image.astype("uint8")
    return image


def scale_image(image):
    """
    Args:
        image : numpy array of image
    Return :
        scaled_image : the standarised image to (x-mean)/sd
    """
    mean = image.mean()
    std = image.std()
    scaled_image = (image - mean) / std
    return scaled_image


def range_01(image):
    """
    Args:
        image : numpy array of image
    Return :
        image : numpy array of image with values scaled between 0 and 1
    """
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image

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

def normalize_min_max(x, dtype=np.float32):
    """
    Min-max normalization for masks.

    Args:
        x (numpy.ndarray): Input array.
        dtype (numpy.dtype): Output data type (default: np.float32).

    Returns:
        numpy.ndarray: Normalized array.
    """
    x = x.astype(dtype, copy=False)
    eps = np.finfo(dtype).eps  # Get the smallest positive value for the data type
    x = (x - np.min(x)) / (np.max(x) - np.min(x) + eps)
    return x


def add_padding(img, pad_size, mode='symmetric'):
    """Pad the image to in_size
    Args :
        img : numpy array of images
        in_size(int) : the input_size of model
        out_size(int) : the output_size of model
        mode(str) : mode of padding
    Return :
        padded_img: numpy array of padded image
    """
    padded_img = np.pad(img, pad_size, mode=mode)
    return padded_img

def empty_gpu_cache(device):
    # Clear the GPU cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
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