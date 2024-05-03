import sys
import os
import random
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from datetime import datetime
import json
import time
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from semantic_bac_segment.loss_functions import DiceLoss, WeightedBinaryCrossEntropy
from semantic_bac_segment.data_loader import BacSegmentDataset, collate_fn, TrainSplit
from semantic_bac_segment.utils import empty_gpu_cache, get_device, tensor_debugger
from semantic_bac_segment.trainlogger import Logger

class UNetTrainer2:
    """
    A class that represents a UNetTrainer for training a U-Net model for semantic bacterial segmentation.

    Args:
        train_dir (str): The directory path where the training data is located.
        models_dir (str): The directory path where the trained models will be saved.
        input_size (int): The size of the input images for the U-Net model.
        precision (str): The precision of the model (e.g., 'float32', 'float16').
        metadata (dict): Additional metadata for the model.

    Attributes:
        train_dir (str): The directory path where the training data is located.
        models_dir (str): The directory path where the trained models will be saved.
        input_size (int): The size of the input images for the U-Net model.
        precision (str): The precision of the model (e.g., 'float32', 'float16').
        previous_weights (None or str): The path to the previous weights file, if available.
        device (str): The device (e.g., 'cpu', 'cuda') on which the model will be trained.
        metadata (dict): Additional metadata for the model.
        model (nn.Module): The U-Net model.

    """
    

    def __init__(self, train_dir, models_dir, input_size, precision, metadata, log_level='INFO'):
        self.train_dir = train_dir
        self.models_dir = models_dir
        self.input_size = input_size
        self.precision = precision
        self.previous_weights = None
        self.device = get_device()
        self.metadata= metadata
        self.logger = Logger('UnetTrainer', level=log_level)

    def add_model(self, nn_model, pooling_steps=4, features=[64, 128, 256, 512], previous_weights=None, dropout=.2):
        """
        Adds a neural network model to the training object.

        Parameters:
            nn_model (nn.Module): The neural network model to be added.
            pooling_steps (int): The number of pooling steps in the model (default: 4).
            features (list): The number of features in each layer of the model (default: [64, 128, 256, 512]).
            previous_weights (str): Path to the previous weights file to load (default: None).
        """
#        model = nn_model(pooling_steps=pooling_steps, features=features, dropout_rate=dropout)
#        if previous_weights:
#            model.load_weights(previous_weights)
        model=nn_model
        model = model.to(self.device)
        torch.compile(model)
        
        self.model = model


    def read_data(self, batch_size, num_workers, subsetting, filter_threshold, val_ratio, collate_fn):
        """
        Attachs the data loader to the trainer. This will read and prepare the training and validation images.

        Args:
            batch_size (int): The batch size for the data loader.
            num_workers (int): The number of worker threads to use for data loading.
            subsetting (str): The subsetting mode for the dataset.
            filter_threshold (float): The filter threshold for the dataset.
            val_ratio (float): The ratio of validation data to split from the training data.
            collate_fn (callable): The function used to collate the samples into batches.
        """
        self.source_folder='source_norm2/'
        self.mask_folder='multiclass_masks/'
        splitter = TrainSplit(os.path.join(self.train_dir, self.source_folder), 
                              os.path.join(self.train_dir, self.mask_folder), val_ratio=val_ratio)
        splitter.get_samplepairs()
        train_pairs, val_pairs = splitter.split_samples()

        # Read data
        dataset = BacSegmentDataset(train_pairs, 
                                    mode='train', 
                                    patch_size=self.input_size, 
                                    subsetting=subsetting, 
                                    filter_threshold=filter_threshold, 
                                    precision=self.precision,
                                    logger=self.logger)
        val_dataset = BacSegmentDataset(val_pairs, 
                                        mode='validation', 
                                        patch_size=self.input_size, 
                                        subsetting=subsetting, 
                                        precision=self.precision,
                                        logger=self.logger)

        num_workers = num_workers
        self.batch_size = batch_size
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
        self.validation_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn, pin_memory=True)


    def train(self, num_epochs=5, model_name="", verbose=True, learning_rate=0.001, gamma=0.1, step_size=1, criterion=DiceLoss()):
        """
        Trains the model for a specified number of epochs.

        Args:
            num_epochs (int): The number of epochs to train the model (default: 5).
            model_name (str): The name of the model (default: "").
            verbose (bool): Whether to print training progress (default: True).
            learning_rate (float): The learning rate for the optimizer (default: 0.001).
            gamma (float): The gamma value for the StepLR scheduler (default: 0.1).
            step_size (int): The step size for the StepLR scheduler (default: 1).
            criterion: The loss function used for training (default: DiceLoss()).
        """
        self.writer = SummaryWriter(comment=f'-{model_name}')

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        best_loss = float('inf')
        tic = time.time()

        if verbose:
            self.logger.log(f'Start training model {model_name}')
            self.logger.log(f'Training on source: {self.source_folder} and mask: {self.mask_folder}')

        for epoch in range(num_epochs):
            self.model.train()
            train_loss, _ = self.run_epoch(self.data_loader, self.model, criterion, optimizer, is_train=True, epoch=epoch)
            scheduler.step()

            self.model.eval()
            with torch.no_grad():
                val_loss, inference_time = self.run_epoch(self.validation_loader, self.model, criterion, is_train=False, epoch=epoch)

            self.log_to_tensorboard('Train', train_loss, _, epoch)
            self.log_to_tensorboard('Validation', val_loss, inference_time, epoch)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.models_dir, f'{model_name}_best_model.pth'))
                now = datetime.now()
                formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")
                self.metadata['date'] = formatted_date
                with open(os.path.join(self.models_dir, f'{model_name}_metadata.json'), 'w') as f:
                    json.dump(self.metadata, f)

            if verbose:
                self.logger.log(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        torch.save(self.model.state_dict(), os.path.join(self.models_dir, f'{model_name}_last_model.pth'))
        self.writer.close()

        if verbose:
            self.logger.log(f'Total training time: {time.time()-tic:.1f} seconds')

    def move_to_device(self, data):
        device=self.device
        if isinstance(data, (list,tuple)):
            return [self.move_to_device(x, device) for x in data]
        return data.to(device)

    def run_epoch(self, loader, model, criterion, optimizer=None, is_train=True, epoch=0):
        total_loss = 0.0
        inference_times = []
        tic = time.time()

        if not is_train:
            log_image_index = random.randint(0, len(loader) - 1)

        for batch_idx, (data, target) in enumerate(loader):
            data, target = self.move_to_device(data), self.move_to_device(target)
            if self.logger.is_level('DEBUG'):
                tensor_debugger(data, 'data', self.logger)
                tensor_debugger(target, 'target', self.logger)

            output = model(data)
            if self.logger.is_level('DEBUG'):
                tensor_debugger(output, 'output', self.logger)

            loss = criterion(output, target)
            total_loss += loss.item()

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            inference_times.append(time.time() - tic)
            tic = time.time()
            

            if not is_train and batch_idx == log_image_index:
                self.writer.add_images('images', data, epoch)
                self.writer.add_images('masks', target, epoch)
                self.writer.add_images('predictions', output, epoch)

        return total_loss / len(loader), np.mean(inference_times)

    def log_to_tensorboard(self, name, loss, time, epoch):
        self.writer.add_scalar(f'{name} Loss', loss, epoch)
        self.writer.add_scalar(f'{name} Time', time, epoch)
