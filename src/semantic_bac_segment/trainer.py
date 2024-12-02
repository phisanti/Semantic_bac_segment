import os
import torch
import time
import json
import traceback
import numpy as np
from typing import Dict, List, Tuple, Any, Callable
from torch.utils.tensorboard import SummaryWriter
from semantic_bac_segment.utils import tensor_debugger, empty_gpu_cache
from semantic_bac_segment.trainlogger import TrainLogger
from semantic_bac_segment.schedulerfactory import SchedulerFactory
from semantic_bac_segment.model_loader import model_loader, ModelRegistry
from torch.utils.data import Subset
import random
from tqdm.auto import tqdm


class MonaiTrainer:
    """
    A trainer class for training and evaluating models using MONAI framework.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_dataset (torch.utils.data.Dataset): The training dataset.
        val_dataset (torch.utils.data.Dataset): The validation dataset.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        device (torch.device): The device to run the training on.
        sigmoid_transform (bool): Whether to apply sigmoid transformation to the model outputs.
        logger (TrainLogger, optional): The logger for logging training progress. Defaults to TrainLogger('MonaiTrainer', level='INFO').
        debugging (bool, optional): Whether to enable debugging mode. Defaults to False.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        sigmoid_transform: bool,
        logger: TrainLogger = TrainLogger("MonaiTrainer", level="INFO"),
        debugging: bool = False,
        nsamples: int = None,
        accumulation_steps: int = 1
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.debugging = debugging
        self.logger = logger
        self.metrics = {}
        self.check_early_stop = False
        self.stop_training = False
        self.sigmoid_transform = sigmoid_transform
        self.val_ratio = len(self.val_dataset) / (len(self.train_dataset) + len(self.val_dataset))
        self.accumulation_steps = accumulation_steps
        if nsamples is None or nsamples == 'None' or nsamples == 0:
            self.nsamples = len(self.train_dataset)
        else:
            self.nsamples = nsamples

    def set_early_stop(self, patience=5):
        """
        Set the early stopping configuration.

        Args:
            patience (int, optional): The number of epochs to wait for improvement before stopping. Defaults to 5.
        """

        self.check_early_stop = True
        self.patience = patience
        self.epochs_without_improvement = 0

    def train(
        self,
        criterion: torch.nn.Module,
        metrics: Dict[str, Callable],
        num_epochs: int,
        saving_folder: str,
        model_name: str,
        model_args: Dict[str, Any],
    ) -> None:
        """
        Train the model.

        Args:
            criterion (torch.nn.Module): The loss function for training.
            metrics (Dict[str, Callable]): A dictionary of metric functions.
            num_epochs (int): The number of epochs to train for.
            saving_folder (str): The folder to save the trained models and configurations.
            model_name (str): The name of the model.
            model_args (Dict[str, Any]): The arguments used to initialize the model.
        """
        self.saving_folder = saving_folder
        self.metrics = metrics if metrics else []
        self.loss_function = criterion
        best_val_loss = 10000
        self.logger.log(f"Training model: {model_name}")
        self.writer = SummaryWriter(comment=f"-{model_name}")

        for epoch in range(num_epochs):
            self.logger.log(f"Iteration {epoch}")
            train_loss, train_metrics, _ = self.run_epoch(
                self.train_dataset, is_train=True, nsamples=self.nsamples
            )
            self.scheduler.step()

            val_loss, val_metrics, _ = self.run_epoch(self.val_dataset, is_train=False, nsamples=int(self.nsamples * self.val_ratio))

            self.logger.log(
                f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            self.writer.add_scalar("Train Loss", train_loss, epoch)
            self.writer.add_scalar("Val Loss", val_loss, epoch)

            for metric_name in self.metrics.keys():
                self.writer.add_scalar(
                    f"Train {metric_name}", train_metrics[metric_name], epoch
                )
                self.writer.add_scalar(
                    f"Val {metric_name}", val_metrics[metric_name], epoch
                )

            if self.check_early_stop:
                self.stop_training = self.early_stop(val_loss, best_val_loss)

            if self.stop_training:
                self.logger.log(
                    f"Early stopping. Validation loss did not improve for {self.patience} epochs. Model's best loss is:  {best_val_loss:.4f}"
                )
                break

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                metrics_output = {
                    metric_name: metric_value.item()
                    for metric_name, metric_value in val_metrics.items()
                }
                self.save_model(model_name, model_args, metrics_output, best=True)
                self.logger.log(
                    f"Saved best model with validation loss: {best_val_loss:.4f}"
                )

        metrics_output = {
            metric_name: metric_value.item()
            for metric_name, metric_value in val_metrics.items()
        }
        self.save_model(model_name, model_args, metrics_output, best=False)
        self.writer.close()

    def multi_train(
        self,
        network_architectures: List[Dict],
        criterion: torch.nn.Module,
        metrics: Dict[str, Callable],
        num_epochs: int,
        saving_folder: str,
        optimizer_params: Dict,
        scheduler_factory: SchedulerFactory,
        model_registry: ModelRegistry = ModelRegistry()
        ) -> None:
        """
        Train multiple model architectures sequentially.

        Args:
            network_architectures (List[Dict]): List of architecture configurations that will be passed to the ModelRegistry to build the model, each containing:
                - model_name: Name of the model
                - model_args: Model initialization parameters
                - weights: Optional path to pretrained weights
            criterion (torch.nn.Module): Loss function for training
            metrics (Dict[str, Callable]): Dictionary of metric functions for evaluation
            num_epochs (int): Number of epochs to train each architecture
            saving_folder (str): Directory to save model checkpoints and configurations
            optimizer_params (Dict): Parameters for the AdamW optimizer including:
                - learning_rate: Initial learning rate
                - weight_decay: Weight decay factor
                - scheduler: Optional scheduler configuration
            scheduler_factory (SchedulerFactory): Factory class for creating learning rate schedulers
            model_registry (ModelRegistry): Registry containing available model architectures

        Each architecture is trained independently with its own optimizer and scheduler.
        Training results and errors are logged using the trainer's logger.
        """
        for arch in network_architectures:
            # Load model, optim and scheduler
            model = model_loader(
                arch, 
                self.device, 
                model_registry=model_registry,
                weights=arch.get('weights')
            )
            torch.compile(model)
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=optimizer_params["learning_rate"], 
                weight_decay=optimizer_params.get('weight_decay', 1e-5)
            )
            scheduler = scheduler_factory.create_scheduler(
                optimizer,
                optimizer_params.get("scheduler", {}),
                num_epochs
            )
            
            # Pass to self
            self.model = model
            self.optimizer = optimizer 
            self.scheduler = scheduler
            
            # Train
            try:
                self.train(
                    criterion,
                    metrics, 
                    num_epochs,
                    saving_folder,
                    arch["model_name"],
                    arch["model_args"]
                )
            except Exception as e:
                self.logger.log(
                    f"Error training {arch['model_name']}: {str(e)}\n{traceback.format_exc()}", 
                    level="ERROR"
                )


    def run_epoch(
        self, dataset: torch.utils.data.Dataset, 
        is_train: bool = True,
        nsamples: int = 1
    ) -> Tuple[float, Dict[str, float], float]:
        """
        Run a single epoch of training or validation.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to use for the epoch.
            is_train (bool, optional): Whether it is a training epoch. Defaults to True.

        Returns:
            Tuple[float, Dict[str, float], float]: A tuple containing the epoch loss, epoch metrics, and inference time.
        """

        epoch_loss = 0
        epoch_metrics = {metric_name: 0 for metric_name in self.metrics.keys()}
        inference_times = []
        tic = time.time()
        self.model.train() if is_train else self.model.eval()
        if nsamples == len(dataset) or is_train == False:
            sampled_dataset = dataset
        else:
            indices = random.sample(range(len(dataset)), nsamples)
            sampled_dataset = Subset(dataset, indices)

        if is_train:
            self.optimizer.zero_grad()
        for i, batch_data in enumerate(sampled_dataset):
            inputs, labels = (
                batch_data["image"].to(self.device),
                batch_data["label"].to(self.device),
            )

            if self.debugging:
                tensor_debugger(inputs, "inputs", self.logger)
                tensor_debugger(labels, "labels", self.logger)

            # For the EverFocus models, we train on the noise structure because that normalises the images
            with torch.set_grad_enabled(is_train):
                outputs = self.model(inputs)
                if self.sigmoid_transform:
                   outputs = torch.sigmoid(outputs)
                if self.debugging:
                    tensor_debugger(outputs, "outputs", self.logger)

                #cleaned = inputs - outputs 
                loss = self.loss_function(outputs, labels)
                
            radom_int = np.random.randint(0, 10)
            if not is_train and radom_int == 0 and self.debugging:
                # Write masks and source to tiff to check prediction
                import tifffile
                tifffile.imwrite('./data/inspection_images/inputs.tiff', inputs.cpu().numpy()[:,0])
                tifffile.imwrite('./data/inspection_images/mask.tiff', labels.cpu().numpy()[:,0])
                tifffile.imwrite('./data/inspection_images/outputs.tiff', outputs.cpu().numpy()[:,0])
                tifffile.imwrite('./data/inspection_images/loss.tiff', labels.cpu().numpy()[:,0] - outputs.cpu().numpy()[:,0])
            if is_train:
                loss = loss / self.accumulation_steps
                loss.backward()
                if (i + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                reported_loss = loss.item() * self.accumulation_steps  # Adjust back for reporting
            else:
                reported_loss = loss.item()

            epoch_loss += reported_loss
            for metric_name, metric_fn in self.metrics.items():
                epoch_metrics[metric_name] += metric_fn(outputs, labels) / len(sampled_dataset)

            inference_times.append(time.time() - tic)
            tic = time.time()

        epoch_loss /= len(sampled_dataset)
        inference_time = np.mean(inference_times)
        empty_gpu_cache(self.device)

        return epoch_loss, epoch_metrics, inference_time

    def early_stop(self, val_loss: float, best_val_loss: float) -> bool:
        """
        Check if early stopping criteria is met.

        Args:
            val_loss (float): The validation loss for the current epoch.
            best_val_loss (float): The best validation loss so far.

        Returns:
            bool: Whether to stop training early.
        """

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                return True

        return False

    def save_model(
        self,
        model_name: str,
        model_args: Dict[str, Any],
        metrics_output: Dict[str, float],
        best: bool = False,
    ) -> None:
        """
        Save the trained model and its configuration.

        Args:
            model_name (str): The name of the model.
            model_args (Dict[str, Any]): The arguments used to initialize the model.
            metrics_output (Dict[str, float]): The output metrics of the model.
            best (bool, optional): Whether it is the best model so far. Defaults to False.
        """

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
            "metrics_output": metrics_output,
        }
        with open(json_path, "w") as json_file:
            json.dump(config_data, json_file, indent=4)
