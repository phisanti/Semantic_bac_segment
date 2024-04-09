import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from semantic_bac_segment.train import UNetTrainer
# from semantic_bac_segment.models.pytorch_basemodel import UNet as UNet
from semantic_bac_segment.models.pytorch_altmodel import UNET as UNet2
from semantic_bac_segment.loss_functions import DiceLoss, WeightedBinaryCrossEntropy
from semantic_bac_segment.data_loader import collate_fn
from semantic_bac_segment.utils import read_cofig


# Get config settings
train_settings=read_cofig('./train_config.yaml')


# Init trainer with data and model
trainer = UNetTrainer(**train_settings['trainer_params'])
trainer.add_model(UNet2, **train_settings['model_params'])
trainer.read_data(**train_settings['data_params'], collate_fn=collate_fn)

# Train the model
#criterion=DiceLoss(is_sigmoid=False)
criterion=WeightedBinaryCrossEntropy(is_sigmoid=False)
trainer.train(**train_settings['optimizer_params'], criterion=criterion)
