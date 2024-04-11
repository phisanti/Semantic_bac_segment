import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
import torch
import numpy as np
import gc
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

    def __init__(self, model_path, model_graph, patch_size, overlap_ratio, half_precision=False):
        """
        Initializes a Segmentator object.

        Args:
            model_path (str): The path to the model.
            model_graph (str): The model graph.
            patch_size (int): The size of the patches.
            overlap_ratio (float): The overlap ratio between patches.
        """
        self.device = get_device()
        self.model = self.get_model(model_path, self.device, model_graph=model_graph)
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio
        self.model.eval()
        self.half_precision=half_precision
        if self.half_precision:
            self.model.half()  


    def predict(self, image):
        """
        Predicts the segmentation mask for the given image. It assumes it receives plain 2D images

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            torch.Tensor: The segmentation mask.
        """
        
        # Crop image into patches of the size required by the net
        image_adapter = ImageAdapter(image, self.patch_size, self.overlap_ratio)
        img_patches = image_adapter.create_patches()
        normalized_patches = np.empty_like(img_patches, dtype=np.float32)

        for i in range(img_patches.shape[0]):
            normalized_patches[i] = normalize_percentile(img_patches[i])

         # Add batch and channel dimensions
        img_tensor = torch.from_numpy(normalized_patches).unsqueeze(1).to(self.device)
        if self.half_precision:
            img_tensor = img_tensor.half()  # Convert input to half-precision

        with torch.no_grad():
            patch_pred= self.model(img_tensor)
        
        patch_pred = patch_pred.squeeze(1).cpu().numpy()
        output_mask = image_adapter.stich_patches(patch_pred)
        
        # Free up tensors
        del image_adapter, img_patches, normalized_patches, img_tensor, patch_pred  
        gc.collect() 
        empty_gpu_cache(self.device)

        return output_mask
   

    def get_model(self, path, device, model_graph=None):
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
                    raise ValueError("Loaded state dictionary does not match the model architecture.")
                
                model.load_state_dict(state_dict)
            
            model.to(device)
            torch.compile(model)
            
            return model
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at path: {path}")
        
        except RuntimeError as e:
            raise RuntimeError(f"Error occurred while loading the model: {str(e)}")
        
        except Exception as e:
            raise Exception(f"Unexpected error occurred: {str(e)}")
        



########## Alternative segmentator using Monai for sliding inference
class Segmentator2:
    """
    A class representing a segmentation model.

    Attributes:
        model (torch.nn.Module): The segmentation model.
        device (torch.device): The device on which the model is loaded.
    """

    def __init__(self, model_path, model_graph, patch_size, overlap_ratio, half_precision=False):
        """
        Initializes a Segmentator object.

        Args:
            model_path (str): The path to the model.
            model_graph (str): The model graph.
            patch_size (int): The size of the patches.
            overlap_ratio (float): The overlap ratio between patches.
        """
        self.device = get_device()
        self.model = self.get_model(model_path, self.device, model_graph=model_graph)
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio
        self.model.eval()
        self.half_precision=half_precision
        if self.half_precision:
            self.model.half()  


    def predict(self, image):
        """
        Predicts the segmentation mask for the given image. It assumes it receives plain 2D images

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            torch.Tensor: The segmentation mask.
        """
        
        # Normalize image
        image = normalize_percentile(image)

        # Add batch and channel dimensions
        img_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)
        if self.half_precision:
            img_tensor = img_tensor.half()  # Convert input to half-precision

        # Create SlidingWindowInferer
        inferer = SlidingWindowInferer(roi_size=self.patch_size, sw_batch_size=1, overlap=self.overlap_ratio)

        with torch.no_grad():
            output_mask = inferer(img_tensor, self.model)

        output_mask = output_mask.squeeze(0).squeeze(0).cpu().numpy()

        # Free up tensors
        del img_tensor, image  
        gc.collect() 
        empty_gpu_cache(self.device)

        return output_mask
   

    def get_model(self, path, device, model_graph=None):
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
                    raise ValueError("Loaded state dictionary does not match the model architecture.")
                
                model.load_state_dict(state_dict)
            
            model.to(device)
            torch.compile(model)
            
            return model
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at path: {path}")
        
        except RuntimeError as e:
            raise RuntimeError(f"Error occurred while loading the model: {str(e)}")
        
        except Exception as e:
            raise Exception(f"Unexpected error occurred: {str(e)}")