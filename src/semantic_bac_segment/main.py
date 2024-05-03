import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from semantic_bac_segment.train import UNetTrainer2
from semantic_bac_segment.models.pytorch_attention import AttentionUNet as AttentionUNet
from semantic_bac_segment.models.pytorch_altmodel import UNET as UNet2, DualPathwayUNet, UNET_edges
from semantic_bac_segment.models.pytorch_bilinealup import UNET_bilineal as UNET_bilineal
from semantic_bac_segment.models.pytorch_hed import HEDUNet
from semantic_bac_segment.models.pytorch_unetplusinv import double_UNET
from semantic_bac_segment.models.pytorch_cnnunet import Unet, SegResNet, dilnet, Unet_dynamic

from monai.networks.nets import unet
from semantic_bac_segment.loss_functions import DiceLoss, MultiClassDiceLoss, WeightedBinaryCrossEntropy, weightedDiceLoss
from semantic_bac_segment.data_loader import collate_fn
from semantic_bac_segment.utils import read_cofig
#from monai.losses import DiceLoss


if __name__ == '__main__':
    
    # Get config settings
    train_settings=read_cofig('./train_config.yaml')

    # Init trainer with data and model
    trainer = UNetTrainer2(**train_settings['trainer_params'], metadata=train_settings, log_level='DEBUG')
    #trainer.add_model(Unet(nb_classes=3), **train_settings['model_params'])
    trainer.add_model(UNet2(out_channels=3, features=[16, 32, 64]), **train_settings['model_params'])
    trainer.read_data(**train_settings['data_params'], collate_fn=collate_fn)

    # Train the model
    #criterion=DiceLoss(is_sigmoid=False)
    criterion=MultiClassDiceLoss(is_sigmoid=True)
    #criterion=weightedDiceLoss(is_sigmoid=True,  zero_weight=0.5, edge_weight=2.0)
    #criterion=WeightedBinaryCrossEntropy(is_sigmoid=True, class_weights = [0.15, 1.85])
    trainer.train(**train_settings['optimizer_params'], criterion=criterion)