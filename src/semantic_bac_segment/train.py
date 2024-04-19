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
from semantic_bac_segment.utils import empty_gpu_cache, get_device, tensor_debbuger


class UNetTrainer:
    def __init__(self, train_dir, models_dir, input_size, precision, metadata):
        self.train_dir = train_dir
        self.models_dir = models_dir
        self.input_size = input_size
        self.precision = precision
        self.previous_weights = None
        self.device = get_device()
        self.metadata= metadata

    def add_model(self, nn_model, pooling_steps=4, previous_weights=None):
        
        model = nn_model(pooling_steps=pooling_steps)
        if previous_weights:
            model.load_weights(previous_weights)
        
        model = model.to(self.device)
        torch.compile(model)
        
        self.model=model


    def read_data(self, batch_size, num_workers, subsetting, filter_threshold, val_ratio, collate_fn):
        
        splitter=TrainSplit(os.path.join(self.train_dir, 'source_norm_cropped/'), 
                   os.path.join(self.train_dir, 'mask_cropped_expanded/'), val_ratio=val_ratio)
        splitter.get_samplepairs()
        train_pairs, val_pairs=splitter.split_samples()
        # Read data
        dataset = BacSegmentDataset(train_pairs, 
                                         mode='train', 
                                         patch_size=self.input_size, 
                                         subsetting=subsetting, 
                                         filter_threshold=filter_threshold, 
                                         precision=self.precision)
        val_dataset = BacSegmentDataset(val_pairs, 
                                             mode='validation', 
                                             patch_size=self.input_size, 
                                             subsetting=subsetting, 
                                             precision=self.precision)
        num_workers=num_workers
        self.batch_size=batch_size
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
        self.validation_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn, pin_memory=True)
        
        self.training_size=len(self.data_loader)
        self.validation_size=len(self.validation_loader)

    def train(self, num_epochs=5, model_name="", verbose=True, learning_rate=0.001, gamma=0.1, step_size=1, criterion=DiceLoss()):

        device=self.device
        writer = SummaryWriter()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        best_loss = float('inf')
        tic = time.time()

        if verbose:
            print(f'Start training')

        for epoch in range(num_epochs):
            epoch_tic = time.time()
            self.model.train()
            train_loss=0.0
            for images, masks in self.data_loader:
                                
                images = images.to(device)
                masks = masks.to(device)

                outputs = self.model(images)
                
                loss = criterion(outputs, masks)
                loss_cpu = loss.to('cpu')
                train_loss += loss_cpu.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            self.model.eval()
            del images, masks
            empty_gpu_cache(self.device)

            with torch.no_grad():
                n=random.randint(0, self.validation_size)
                index=0
                val_loss = 0.0
                inference_times=np.zeros(self.validation_size)

                for val_images, val_masks in self.validation_loader:
                    
                    inference_tic = time.time()
                    val_images = val_images.to(device)
                    val_masks = val_masks.to(device)
                    

                    output_imgs = self.model(val_images)
                    
                    inference_tac = time.time()
                    val_loss += criterion(output_imgs, val_masks)

                    if index == n:
                        # Add images and masks to TensorBoard
                        writer.add_images('images', val_images, epoch)
                        writer.add_images('masks', val_masks, epoch)
                        writer.add_images('predictions', output_imgs, epoch)

                    if verbose:
                        print(f"Inference time: {inference_tac-inference_tic:.2f} seconds")
                    inference_times[index]=inference_tac-inference_tic
                    index += 1

                    del val_images, val_masks, output_imgs
                    empty_gpu_cache(self.device)


                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(self.model.state_dict(), os.path.join(self.models_dir, f'{model_name}_best_model.pth'))
                    now = datetime.now()
                    formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")
                    self.metadata['date'] = formatted_date
                    with open(os.path.join(self.models_dir, f'{model_name}_metadata.json'), 'w') as f:
                        json.dump(self.metadata, f)
            epoch_tac = time.time()
            train_loss= train_loss/len(self.data_loader)
            val_loss= val_loss/len(self.data_loader)
            writer.add_scalar('Train Loss', train_loss, epoch)
            writer.add_scalar('Validation Loss', val_loss, epoch)
            writer.add_scalar('Epoch Time', epoch_tac-epoch_tic, epoch)
            writer.add_scalar('Inference Time', np.mean(inference_times), epoch)

            if verbose:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
                print(f'Training epoch time: {epoch_tac-epoch_tic} seconds')

        torch.save(self.model.state_dict(), os.path.join(self.models_dir, f'{model_name}_last_model.pth'))
        writer.close()

        if verbose:
            tac=time.time()
            print(f'Total training time: {tac-tic:.1f} seconds')

