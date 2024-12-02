import torch
from typing import Dict, Optional, Union, Type, Mapping
from semantic_bac_segment.models.pytorch_cnnunet import Unet as atomai_unet
from semantic_bac_segment.models.multiscaleunet import MultiResUnet
from semantic_bac_segment.models.flexmultiscaleunet import FlexMultiScaleUNet
from semantic_bac_segment.models.pytorch_altmodel import UNET as base_unet
from semantic_bac_segment.models.pytorch_attention import AttentionUNet
from monai.networks.nets import UNet as MonaiUnet
from monai.networks.nets import AttentionUnet as MonaiAttentionUNet
from monai.networks.nets import (
    UNETR,
    AHNet,
    DenseNet169,
    EfficientNet,
    DynUNet,
    SwinUNETR,
    BasicUNetPlusPlus,
)


class ModelRegistry:
    """A registry for storing and retrieving model architectures."""
    
    def __init__(self, models: Optional[Mapping[str, Type[torch.nn.Module]]] = None) -> None:
        """
        Initialize the model registry.
        
        Args:
            models: Initial mapping of model names to model classes
        """
        self._models: Dict[str, Type[torch.nn.Module]] = models or {
            "MonaiUnet": MonaiUnet,
            "MultiResUnet": MultiResUnet,
            "FlexMultiScaleUNet": FlexMultiScaleUNet,
            "MonaiAttentionUNet": MonaiAttentionUNet,
            "UNETR": UNETR,
            "AHNet": AHNet,
            "DenseNet169": DenseNet169,
            "EfficientNet": EfficientNet, 
            "DynUNet": DynUNet,
            "SwinUNETR": SwinUNETR,
            "BasicUNetPlusPlus": BasicUNetPlusPlus,
            "atomai_unet": atomai_unet,
            "base_unet": base_unet,
            "AttentionUNet": AttentionUNet
        }

    def register(self, name: str, model_class: Type[torch.nn.Module]) -> None:
        """
        Register a new model architecture.
        
        Args:
            name: Name identifier for the model
            model_class: The model class to register
        """
        self._models[name] = model_class

    def get_model(self, name: str) -> Type[torch.nn.Module]:
        """
        Retrieve a model class by name.
        
        Args:
            name: Name of the model to retrieve
            
        Returns:
            The requested model class
            
        Raises:
            ValueError: If model name not found in registry
        """
        if name not in self._models:
            raise ValueError(f"Unknown model: {name}")
        return self._models[name]

def model_loader(
    network_architecture: Dict[str, Union[str, Dict]], 
    device: torch.device,
    model_registry: ModelRegistry = ModelRegistry(),
    weights: Optional[str] = None
    ) -> torch.nn.Module:
    """
    Load and initialize a model based on provided architecture specification.
    
    Args:
        network_architecture: Dictionary containing model name and arguments
        device: Device to load model onto
        model_registry: Registry containing available model architectures
        weights: Optional path to pretrained weights
        
    Returns:
        Initialized model on specified device
        
    Raises:
        ValueError: If model type not found in registry
    """
    model_name = network_architecture["model_name"]
    model_args = network_architecture["model_args"]
    
    model_type = model_name.split("-")[0] if "-" in model_name else model_name
    model_class = model_registry.get_model(model_type)
    model = model_class(**model_args).to(device)

    if weights:
        model.load_state_dict(torch.load(weights, map_location=device))

    return model