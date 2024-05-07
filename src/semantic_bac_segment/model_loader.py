from semantic_bac_segment.models.pytorch_cnnunet import Unet as atomai_unet
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
    BasicUNetPlusPlus
)
def model_loader(network_architecture, device):

    model_name = network_architecture["model_name"]
    model_args = network_architecture["model_args"]

    if "-" in model_name:
        model_type = model_name.split("-")[0]
    else:
        model_type = model_name
    
    # Create the model instance based on the model name
    if model_type == "MonaiUnet":
        model = MonaiUnet(**model_args).to(device)
    elif model_type == "MonaiAttentionUNet":
        model = MonaiAttentionUNet(**model_args).to(device)
    elif model_type == "UNETR":
        model = UNETR(**model_args).to(device)
    elif model_type == "AHNet":
        model = AHNet(**model_args).to(device)
    elif model_type == "DenseNet169":
        model = DenseNet169(**model_args).to(device)
    elif model_type == "EfficientNet":
        model = EfficientNet(**model_args).to(device)
    elif model_type == "DynUNet":
        model = DynUNet(**model_args).to(device)
    elif model_type == "SwinUNETR":
        model = SwinUNETR(**model_args).to(device)
    elif model_type == "BasicUNetPlusPlus":
        model = BasicUNetPlusPlus(**model_args).to(device)
    elif model_type == "atomai_unet":
        model = atomai_unet(**model_args).to(device)
    elif model_type == "base_unet":
        model = base_unet(**model_args).to(device)
    elif model_type == "AttentionUNet":
        model = AttentionUNet(**model_args).to(device)
    else:
        raise ValueError(f"Unknown model family: {model_type}")
    
    return model