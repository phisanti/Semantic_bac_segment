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
#from semantic_bac_segment.loss_functions import DiceLoss, MultiClassDiceLoss, WeightedBinaryCrossEntropy, weightedDiceLoss, MultiChannelWeightedBinaryCrossEntropy
from semantic_bac_segment.data_loader import collate_fn
from semantic_bac_segment.utils import read_cofig
#from monai.losses import DiceLoss
from semantic_bac_segment.loss_functions import DiceLoss


configurations = [
    {
        'id': 'unet_16_2_2_2_3',
        'model': Unet(nb_filters=16, layers=[2, 2, 2, 3]),
        'nb_filters': 16,
        'layers': [2, 2, 2, 3],
    },
        {
        'id': 'unet_16_2_2_2_3_dil',
        'model': Unet(nb_filters=16, layers=[2, 2, 2, 3], with_dilation=True),
        'nb_filters': 16,
        'layers': [2, 2, 2, 3],
    },
    {
        'id': 'unet2_16_32_64_128',
        'model': UNet2(features=[16, 32, 64, 128]),
        'features': [16, 32, 64, 128],
    },
    {
        'id': 'unet2_32_64_128_256',
        'model': UNet2(features=[32, 64, 128, 256]),
        'features': [32, 64, 128, 256],
    },
    {
        'id': 'unet_16_1_2_2_3',
        'model': Unet(nb_filters=16, layers=[1, 2, 2, 3]),
        'nb_filters': 16,
        'layers': [1, 2, 2, 3],
    },
    {
        'id': 'unet_32_1_2_2_3',
        'model': Unet(nb_filters=32, layers=[1, 2, 2, 3]),
        'nb_filters': 32,
        'layers': [1, 2, 2, 3],
    },
        {
        'id': 'unet_32_1_2_2_3_dil',
        'model': Unet(nb_filters=32, layers=[1, 2, 2, 3], with_dilation=True),
        'nb_filters': 32,
        'layers': [1, 2, 2, 3],
    },
    {
        'id': 'unet2_64_128_256',
        'model': UNet2(features=[64, 128, 256]),
        'features': [64, 128, 256],
    },
    {
        'id': 'unet_64_2_2_2_3',
        'model': Unet(nb_filters=64, layers=[2, 2, 2, 3]),
        'nb_filters': 64,
        'layers': [2, 2, 2, 3],
    },
    {
        'id': 'unet_32_2_2_2_3',
        'model': Unet(nb_filters=32, layers=[2, 2, 2, 3]),
        'nb_filters': 32,
        'layers': [2, 2, 2, 3],
    },
]


if __name__ == '__main__':
    
    
    # Get config settings
    train_settings=read_cofig('./train_config.yaml')
    # Init trainer with data and model
    train_settings['optimizer_params']['model_name']='unet2_64-256-channel0'
    trainer = UNetTrainer2(**train_settings['trainer_params'], metadata=train_settings, log_level='INFO')
    trainer.add_model(UNet2(features=[16, 32, 64, 128]), **train_settings['model_params'])
    trainer.read_data(**train_settings['data_params'], collate_fn=collate_fn)

    # Train the model
    criterion = DiceLoss(is_sigmoid=True)
    best_loss_i=trainer.train(**train_settings['optimizer_params'], criterion=criterion)

"""
    for config in configurations:
        print(f'Training {config["id"]}')
        # Init trainer with data and model
        trainer = UNetTrainer2(**train_settings['trainer_params'], metadata=train_settings, log_level='INFO')
        trainer.add_model(config['model'], **train_settings['model_params'])
        trainer.read_data(**train_settings['data_params'], collate_fn=collate_fn)

        # Train the model
        criterion = DiceLoss(is_sigmoid=True)
        best_loss_i=trainer.train(**train_settings['optimizer_params'], criterion=criterion)
        print(f'Best loss for {config["id"]}: {best_loss_i}')

"""