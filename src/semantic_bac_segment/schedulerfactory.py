from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

class SchedulerFactory:
    @staticmethod
    def create_scheduler(optimizer, scheduler_config, num_epochs):
        """
        Create and return the appropriate scheduler based on the configuration.

        Args:
            optimizer: The optimizer to use with the scheduler.
            scheduler_config: A dictionary containing scheduler configuration.
            num_epochs: Total number of epochs for training.

        Returns:
            A scheduler object.
        """
        scheduler_type = scheduler_config.get("type", "StepLR")

        if scheduler_type == "StepLR":
            return StepLR(
                optimizer,
                step_size=int(scheduler_config.get("step_size", 15)),
                gamma=float(scheduler_config.get("gamma", 0.1))
            )
        elif scheduler_type == "CosineAnnealingLR":
            return CosineAnnealingLR(
                optimizer,
                T_max=int(scheduler_config.get("T_max", num_epochs)),
                eta_min=float(scheduler_config.get("eta_min", 1e-7))
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")