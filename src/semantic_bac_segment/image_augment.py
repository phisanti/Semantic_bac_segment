import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from typing import Tuple
import numpy as np
import random
import cv2
from scipy.ndimage import map_coordinates, gaussian_filter
from semantic_bac_segment.utils import get_bit_depth, invert_image

class ImageAugmenter:
    """Apply a random transformation to the input image and mask for image augmentation during training.
    Args:
        img (numpy.ndarray): Input image as a NumPy array.
        mask (numpy.ndarray): Corresponding mask as a NumPy array.
        verbose (bool, optional): Whether to print the applied transformation. Defaults to False.
    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Augmented image and mask.
    """
    def __init__(self, seed = None):
        self.transfroms = [
#            self.flip,
#            self.elastic_transform,
            self.gaussian_noise,
            self.change_brightness,
#            self.invert,
            self.no_transformation 
        ]
        self.seed = seed

    def transform(self, img: np.ndarray, mask: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if self.seed:
            random.seed(self.seed)
        transform = random.choice(self.transfroms)
        self.last_transform = transform.__name__
        if verbose:
            print(f'Applying transform: {self.last_transform}')
        augmented_images = transform(img, mask)

        return augmented_images
    

    def flip(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            image : numpy array of image
        Return :
            image : numpy array of flipped image
        """
        flip_axis = random.randint(0, 2)

        if flip_axis == 0:
            img = np.flip(img, flip_axis)
            mask = np.flip(mask, flip_axis)
        elif flip_axis == 1:
            img = np.flip(img, flip_axis)
            mask = np.flip(mask, flip_axis)
        elif flip_axis == 2:
            img = np.flip(img, 0)
            img = np.flip(img, 1)

            mask = np.flip(mask, 0)
            mask = np.flip(mask, 1)
        else:
            pass            

        return img, mask


    def gaussian_noise(self, 
                       img: np.ndarray, 
                       mask: np.ndarray, 
                       mean=None, 
                       std=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adds Gaussian noise to an input image.

        Args:
            image (numpy.ndarray): Input image as a NumPy array.
            mean (float, optional): Mean value for the Gaussian noise distribution. If not provided, it is calculated from the input image.
            std (float, optional): Standard deviation value for the Gaussian noise distribution. If not provided, it is calculated from the input image.

        Returns:
            numpy.ndarray: Image with added Gaussian noise.
        """
        # Inputs
        if not isinstance(img, np.ndarray):
            raise TypeError("Input image must be a NumPy array.")
        if mean is None:
            mean = img.mean()
        if std is None:
            std = img.std()

        # Add Gaussian noise
        noise = np.random.normal(mean, std, img.shape)
        noisy_img = img + noise

        # Clip values to image range for
        noisy_img = noisy_img.astype("uint16")
        noisy_img = clip_image(noisy_img)
        return noisy_img, mask


    def invert(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Invert the input image.

        Args:
            img (numpy.ndarray): Input image as a NumPy array.
            mask (numpy.ndarray): Corresponding mask as a NumPy array.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: Inverted image and original mask.
        """
        bit_depth = get_bit_depth(img)
        inverted_img = invert_image(img, bit_depth)
        return inverted_img, mask


    def change_brightness(self, 
                          img: np.ndarray, 
                          mask: np.ndarray, 
                          value=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            image : numpy array of image
            value : brightness
        Return :
            image : numpy array of image with brightness added
        """
        if value is None:
            value = random.randint(-20, 20)

        img = img.astype("int16")
        img = img + value
        img = img.astype("uint16")

        img = clip_image(img)
        return img, mask


    def elastic_transform(self, 
                          img: np.ndarray, 
                          mask: np.ndarray, 
                          alpha=34, 
                          sigma=5, 
                          random_state=None) -> Tuple[np.ndarray, np.ndarray]:
        """Apply elastic deformation to images as augmentation.
        
        Args:
            img: Input image as NumPy array.
            mask: Corresponding mask as NumPy array.
            alpha: Scale factor for deformation.
            sigma: Gaussian filter parameter.
            random_state: RandomState instance.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Augmented image and mask.
        """
        if random_state is None:
            random_state = np.random.RandomState(None)
            
        shape = img.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

        img=map_coordinates(img, indices, order=1).reshape(shape)
        mask=map_coordinates(mask, indices, order=1).reshape(shape)
        

        return img, mask
    

    def no_transformation(self, 
                          img: np.ndarray, 
                          mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return img, mask


    def edges(self, img, mask, min_n=100, max_n=200):
        edges = cv2.Canny(cv2.convertScaleAbs(mask), min_n, max_n)

        return img, edges


def clip_image(img: np.ndarray) -> np.ndarray:
    """Clip values in an image to be within valid range.
    Args:
        img : numpy array of image
    Return :
        img : numpy array of clipped image
    """

    bit_depth= get_bit_depth(img)
    if img.dtype.kind in ['i', 'u']:
        if bit_depth == 8:
            img = np.clip(img, 0, 255)
        elif bit_depth == 16:
            img = np.clip(img, 0, 65535)
        elif bit_depth == 32:
            img = np.clip(img, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
        else:
            img = np.clip(img, np.iinfo(img.dtype).min, np.iinfo(img.dtype).max)
    elif img.dtype.kind == 'f':
        img = np.clip(img, np.finfo(img.dtype).min, np.finfo(img.dtype).max)
    
    return img

# TODO: implement zoom and gaussian blur

if __name__ == '__main__':
    img = np.random.randint(0, 256, (512, 512), dtype=np.uint8) 
    img_copy= np.copy(img)
    mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)

    augmenter = ImageAugmenter(seed=None)

    for i in range(100):
        aug_img, aug_mask = augmenter.transform(img, mask)

        print(f"Applied transform: {augmenter.last_transform}")
        # Assert augmented image and mask have expected properties
        assert aug_img.shape == img.shape
        assert aug_mask.shape == mask.shape

        if augmenter.last_transform == "change_brightness" or augmenter.last_transform == "gaussian_noise":
            print(f"Image diff: {img.mean() - aug_img.mean()}")

        if augmenter.last_transform == "flip":
            print(f"Image flipped: {not np.all(aug_img[0, :] == img_copy[0, :])}")
            assert not np.all(aug_img[0, :] == img_copy[0, :])


def elastic_transform_ndim(
                      img: np.ndarray, 
                      ndim: int,
                      alpha=34, 
                      sigma=5, 
                      random_state=None) -> Tuple[np.ndarray, np.ndarray]:
    """Apply elastic deformation to images as augmentation.
    
    Args:
        img: Input image as NumPy array.
        ndim: Number of dimensions of the input image (2 or 3).
        alpha: Scale factor for deformation.
        sigma: Gaussian filter parameter.
        random_state: RandomState instance.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Augmented image and mask.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
        
    shape = img.shape
    
    if ndim == 2:
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
        
        img = map_coordinates(img, indices, order=1).reshape(shape)
    
    elif ndim == 3:
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        
        x, y, z = np.meshgrid(np.arange(shape[2]), np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(z+dz, (-1, 1)), np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
        
        img = map_coordinates(img, indices, order=1).reshape(shape)
    
    else:
        raise ValueError("Invalid number of dimensions. `ndim` must be either 2 or 3.")
    
    return img



def elastic_transform(self, 
                        img: np.ndarray, 
                        alpha=34, 
                        sigma=5, 
                        random_state=None) -> Tuple[np.ndarray, np.ndarray]:
    """Apply elastic deformation to images as augmentation.
    
    Args:
        img: Input image as NumPy array.
        mask: Corresponding mask as NumPy array.
        alpha: Scale factor for deformation.
        sigma: Gaussian filter parameter.
        random_state: RandomState instance.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Augmented image and mask.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
        
    shape = img.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    img=map_coordinates(img, indices, order=1).reshape(shape)
        