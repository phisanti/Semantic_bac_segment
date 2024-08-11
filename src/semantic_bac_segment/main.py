import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import traceback
import json
import torch
from torch.nn import CrossEntropyLoss
from monai.transforms import (
    Compose,
    RandRotate90d,
    Rand2DElasticd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    ScaleIntensityd,
    RandZoomd,
    ToTensord,
)
from monai.losses import DiceLoss as monai_dice
from monai.metrics import DiceMetric, compute_iou
from semantic_bac_segment.trainlogger import TrainLogger
from semantic_bac_segment.confreader import ConfReader
from semantic_bac_segment.utils import get_device, tensor_debugger
from semantic_bac_segment.data_loader import BacSegmentDatasetCreator
from semantic_bac_segment.trainer import MonaiTrainer
from semantic_bac_segment.loss_functions import (
    DiceLoss,
    WeightedBinaryCrossEntropy,
    MultiClassDiceLoss,
    MultiClassWeightedBinaryCrossEntropy,
    MaxDiceLoss,
    FocalLoss,
)
from semantic_bac_segment.transforms import (
    TIFFLoader,
    ClearBackgroundTransform,
    NormalizePercentileTransform,
    HistEq,
    Ensure4D,
    ComposeInspect,
)
from semantic_bac_segment.model_loader import model_loader


# Read configs
def main():
    # 1. Read configs and set basic

    if len(sys.argv) != 2:
        raise ValueError("Incorrect number of arguments. Usage: python main.py <path_to_train_configs.yml>")

    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}. Please provide a valid path to the train_configs.yml file.")

    if not config_path.endswith(('.yml', '.yaml')):
        raise ValueError("Invalid file format. Please provide a YAML file (.yml or .yaml extension).")

    c_reader = ConfReader(config_path)
    configs = c_reader.opt
    device = get_device()
    debugging = True
    log_level = "DEBUG" if debugging else "INFO"
    trainlogger = TrainLogger("MonaiTrainer", level=log_level)

    # 2. Compose data transformations
    # 2.1 Full image transform (read and remove background)
    img_transforms = Compose(
        [
            TIFFLoader(keys=["image"]),
            TIFFLoader(
                keys=["label"], add_channel_dim=configs.trainer_params["binary_segmentation"]
            ),  # If running on 2D images, change to True
            ClearBackgroundTransform(
                keys=["image"], sigma_r=100, method="divide", convert_32=True
            ),
            HistEq(keys=["image"]),

        ]
    )

    # 2.2 Patch transfroms
    patch_train_trans = Compose(
        [
            RandZoomd(keys=["image", "label"], prob=0.5, min_zoom=0.8, max_zoom=1.2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.1),
            RandGaussianSmoothd(
                keys=["image"], prob=0.5, sigma_x=(0.1, 1.1), sigma_y=(0.1, 1.1)
            ),
            NormalizePercentileTransform(keys=["image"], pmax=99),
            ScaleIntensityd(keys=["label"]),
            ToTensord(keys=["image", "label"]),
        ]
    )
    patch_val_trans = Compose(
        [
            #NormalizePercentileTransform(keys=["image"], pmax=95),
            ScaleIntensityd(keys=["label"]),
            ToTensord(keys=["image", "label"]),
        ]
    )

    # 3. Get datasets
    dataset_creator = BacSegmentDatasetCreator(
        configs.dataset_params["source_dir"],
        configs.dataset_params["masks_dir"],
        configs.dataset_params["val_ratio"],
    )
    train_dataset, val_dataset = dataset_creator.create_datasets(
        img_transforms, img_transforms
    )
    train_patch_dataset, val_patch_dataset = dataset_creator.create_patches(
        num_samples=configs.dataset_params["n_patches"],
        roi_size=(configs.dataset_params["input_size"], 
                  configs.dataset_params["input_size"]),
        train_transforms=patch_train_trans,
        val_transforms=patch_val_trans,
    )

    # 4. Get loss and metrics
    loss_function = DiceLoss(is_sigmoid=True)
    metrics = {
        #"MaxDice": MaxDiceLoss(is_sigmoid=True, include_background=False),
        #"Dice": MultiClassDiceLoss(is_sigmoid=True),
        "Monai_diceloss": monai_dice(to_onehot_y=False, sigmoid=False),
        #"CrossEntropy": MultiClassWeightedBinaryCrossEntropy(
        #    is_sigmoid=True
        #),
        "Cross_entropy_pytorch": CrossEntropyLoss(),
        'Focal_loss' : FocalLoss(is_sigmoid=True)
    }

    # 5. Get list of architectures and run Training loop
    num_epochs = configs.trainer_params["num_epochs"]
    with open(configs.trainer_params["model_settings"]) as file:
        network_arch = json.load(file)
        
    trainlogger.log(c_reader.pretty_print(configs), level="INFO")
    # 6. Iterate over models for training
    for model_i in network_arch:
        try:
            m = model_loader(model_i, device)
            torch.compile(m)
            optimizer = torch.optim.Adam(
                m.parameters(), lr=configs.optimizer_params["learning_rate"]
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=configs.optimizer_params["step_size"],
                gamma=configs.optimizer_params["gamma"],
            )

            trainer = MonaiTrainer(
                m,
                train_patch_dataset,
                val_patch_dataset,
                optimizer,
                scheduler,
                device,
                sigmoid_transform=True,
                logger=trainlogger,
                debugging=configs.trainer_params["debugging"],
            )
            trainer.set_early_stop(patience=configs.trainer_params["early_stop_patiente"])
            trainer.logger.log(
                f"Training on {device} for {num_epochs} epochs", level="INFO"
            )
            trainer.train(
                loss_function,
                metrics,
                num_epochs,
                configs.trainer_params["model_save"],
                model_i["model_name"],
                model_i["model_args"],
            )

        except Exception as e:
            error_message = f"An error occurred while training model {model_i['model_name']}: {str(e)}\n"
            error_message += f"Traceback: {traceback.format_exc()}\n"
            trainer.trainlogger.log(
                error_message, level="ERROR"
            )  # Use the logger instance directly


if __name__ == "__main__":
    main()
