import os
import torch
import time
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from semantic_bac_segment.loss_functions import DiceLoss
from monai.metrics import DiceMetric
from semantic_bac_segment.utils import tensor_debugger, empty_gpu_cache
from semantic_bac_segment.trainlogger import Logger
from tqdm.auto import tqdm


class MonaiTrainer:
    def __init__(self, model, train_dataset, val_dataset, optimizer, scheduler, device, sigmoid_transform,  debugging=False):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.debugging = debugging
        log_level = 'DEBUG' if self.debugging else 'INFO'
        self.logger = Logger('MonaiTrainer', level=log_level)
        self.check_early_stop=False
        self.stop_training=False
        self.sigmoid_transform= sigmoid_transform

    def set_early_stop(self, patience=5):
        self.check_early_stop=True
        self.patience = patience
        self.epochs_without_improvement=0

    def train(self, criterion, metrics, num_epochs,saving_folder, model_name, model_args):
        self.saving_folder = saving_folder
        self.metrics = metrics if metrics else []
        self.loss_function = criterion
        best_val_loss = 10000
        self.logger.log(f'Training model: {model_name}')
        self.writer = SummaryWriter(comment=f'-{model_name}')

        for epoch in range(num_epochs):
            self.logger.log(f'Iteration {epoch}')
            train_loss, train_metrics, _ = self.run_epoch(self.train_dataset, is_train=True)
            self.scheduler.step()

            val_loss, val_metrics, _ = self.run_epoch(self.val_dataset, is_train=False)

            self.logger.log(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            self.writer.add_scalar('Train Loss', train_loss, epoch)
            self.writer.add_scalar('Val Loss', val_loss, epoch)
            
            for metric_name in self.metrics.keys():
                self.writer.add_scalar(f'Train {metric_name}', train_metrics[metric_name], epoch)
                self.writer.add_scalar(f'Val {metric_name}', val_metrics[metric_name], epoch)

            if self.check_early_stop:
                self.stop_training=self.early_stop(val_loss, best_val_loss)
            
            if self.stop_training:
                self.logger.log(f"Early stopping. Validation loss did not improve for {self.patience} epochs. Model's best loss is:  {best_val_loss:.4f}")
                break


            if val_loss < best_val_loss:
                best_val_loss = val_loss
                metrics_output = {
                    metric_name: metric_value.item() for metric_name, metric_value in val_metrics.items()
                }
                self.save_model(model_name, model_args, metrics_output, best=True)
                self.logger.log(f"Saved best model with validation loss: {best_val_loss:.4f}")

        metrics_output = {
            metric_name: metric_value.item() for metric_name, metric_value in val_metrics.items()
        }
        self.save_model(model_name, model_args, metrics_output, best=False)
        self.writer.close()

    def run_epoch(self, dataset, is_train=True):

        epoch_loss = 0
        epoch_dice = 0
        epoch_metrics = {metric_name: 0 for metric_name in self.metrics.keys()}
        inference_times = []
        tic = time.time()
        self.model.train() if is_train else self.model.eval()

        for batch_data in dataset:
            inputs, labels = batch_data["image"].to(self.device), batch_data["label"].to(self.device)

            if is_train:
                self.optimizer.zero_grad()

            if self.debugging:
                tensor_debugger(inputs, 'inputs', self.logger)
                tensor_debugger(labels, 'labels', self.logger)

            with torch.set_grad_enabled(is_train):
                outputs = self.model(inputs)
                if self.sigmoid_transform:
                    outputs = torch.sigmoid(outputs)
                if self.debugging:
                    tensor_debugger(outputs, 'outputs', self.logger)

                loss = self.loss_function(outputs, labels)


            if is_train:
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item()
            for metric_name, metric_fn in self.metrics.items():
                epoch_metrics[metric_name] += metric_fn(outputs, labels)/len(dataset)

            inference_times.append(time.time() - tic)
            tic = time.time()


        epoch_loss /= len(dataset)
        epoch_dice /= len(dataset)
        inference_time = np.mean(inference_times)
        empty_gpu_cache(self.device)

        return epoch_loss, epoch_metrics, inference_time

    def early_stop(self, val_loss, best_val_loss):

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                return True
            
        return False
    
    def save_model(self, model_name, model_args, metrics_output, best=False):
        # Save the model weights
        model_filename = f"{model_name}_{'best' if best else 'final'}_model.pth"
        model_path = os.path.join(self.saving_folder, model_filename)
        torch.save(self.model.state_dict(), model_path)

        # Save the model configuration and metrics output as JSON
        json_filename = f"{model_name}_{'best' if best else 'final'}_config.json"
        json_path = os.path.join(self.saving_folder, json_filename)
        config_data = {
            "model_name": model_name,
            "model_args": model_args,
            "metrics_output": metrics_output
        }
        with open(json_path, "w") as json_file:
            json.dump(config_data, json_file, indent=4)