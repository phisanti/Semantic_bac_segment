import torch
import torch.nn as nn
import torch.nn.functional as F
from semantic_bac_segment.utils import tensor_debugger
from monai.transforms import DistanceTransformEDT as monai_distance_transform_edt
from monai.losses import DiceFocalLoss
from monai.utils import LossReduction
from typing import Literal

class DiceLoss(nn.Module):
    """
    Computes the Dice loss between the predicted and target masks.

    Args:
        weight (float or list): Weight(s) to assign to each class.
        size_average (bool): Whether to average the loss across all pixels.
        is_sigmoid (bool): Whether to apply sigmoid activation to the predicted mask.
    """

    def __init__(self, weight=None, size_average=True, is_sigmoid=True):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.is_sigmoid = is_sigmoid

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.float()
        targets = targets.float()

        if self.is_sigmoid:
            pass
        else:
            inputs = torch.sigmoid(inputs)

        if targets.shape != inputs.shape:
            raise AssertionError(
                f"ground truth has different shape ({targets.shape}) from input ({inputs.shape})"
            )

        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        if self.weight is not None:
            dice = dice * self.weight

        if self.size_average:
            dice = dice.mean()
        else:
            dice = dice.sum()

        return 1 - dice


class MaxDiceLoss(nn.Module):
    """
    Computes the Dice loss using the maximum value of each channel. This is useful
    when missclassification between classes is not as important as missclassification
    against background.

    Args:
        weight (float or list): Weight(s) to assign to each class.
        size_average (bool): Whether to average the loss across all pixels.
        is_sigmoid (bool): Whether to apply sigmoid activation to the predicted mask.
        include_background (bool): Whether to include the background class in the loss computation.
    """

    def __init__(
        self, weight=None, size_average=True, is_sigmoid=True, include_background=True
    ):
        super(MaxDiceLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.is_sigmoid = is_sigmoid
        self.include_background = include_background

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.float()
        targets = targets.float()

        if self.is_sigmoid:
            pass
        else:
            inputs = torch.sigmoid(inputs)

        if targets.shape != inputs.shape:
            raise AssertionError(
                f"ground truth has different shape ({targets.shape}) from input ({inputs.shape})"
            )

        if not self.include_background:
            # remove background from masks
            inputs = inputs[:, 1:]
            targets = targets[:, 1:]

        # Flatten all channels into one with the maximum value of each channel
        inputs = torch.max(inputs, dim=1, keepdim=True)[0]
        targets = torch.max(targets, dim=1, keepdim=True)[0]

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        if self.weight is not None:
            dice = dice * self.weight

        if self.size_average:
            dice = dice.mean()
        else:
            dice = dice.sum()
        return 1 - dice


class weightedDiceLoss(nn.Module):
    """
    Computes the Dice loss with custom weights for edges and background.

    Args:
        weight (float or list): Weight(s) to assign to each class.
        size_average (bool): Whether to average the loss across all pixels.
        is_sigmoid (bool): Whether to apply sigmoid activation to the predicted mask.
        include_background (bool): Whether to include the background class in the loss computation.
    """

    def __init__(
        self,
        weight=None,
        size_average=True,
        is_sigmoid=True,
        zero_weight=0.5,
        edge_weight=2.0,
    ):
        super(weightedDiceLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.is_sigmoid = is_sigmoid
        self.zero_weight = zero_weight
        self.edge_weight = edge_weight

    def forward(self, inputs, targets, smooth=0.1):
        inputs = inputs.float()
        targets = targets.float()

        if self.is_sigmoid:
            pass
        else:
            inputs = torch.sigmoid(inputs)

        if targets.shape != inputs.shape:
            raise AssertionError(
                f"ground truth has different shape ({targets.shape}) from input ({inputs.shape})"
            )

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Apply weights where target is 0 (background)
        zero_mask = targets == 0
        inputs = inputs * torch.where(zero_mask, self.zero_weight, 1)

        # Apply weights to edges
        edge_mask = self.get_edge_mask(targets)
        inputs = inputs * torch.where(edge_mask, self.edge_weight, 1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        if self.weight is not None:
            dice = dice * self.weight

        if self.size_average:
            dice = dice.mean()
        else:
            dice = dice.sum()

        return 1 - dice

    def get_edge_mask(self, targets):
        # Create a binary edge mask by applying edge detection on the target tensor
        edge_mask = torch.zeros_like(targets, dtype=torch.bool)
        edge_mask = edge_mask.bool()  # Convert to boolean tensor
        edge_mask[1:] |= (targets[1:] != targets[:-1]).bool()
        edge_mask[:-1] |= (targets[:-1] != targets[1:]).bool()
        return edge_mask


class MultiClassDiceLoss(nn.Module):
    """
    Computes the Dice loss for multi-class segmentation.

    Args:
        weight (list): Weight(s) to assign to each class.
        size_average (bool): Whether to average the loss across all pixels.
        is_sigmoid (bool): Whether to apply sigmoid activation to the predicted mask.
    """

    def __init__(self, weight=None, size_average=True, is_sigmoid=True):
        super(MultiClassDiceLoss, self).__init__()
        if weight is None:
            self.weight = [1.0] * 1  # Default to equal weight for single-channel input
        else:
            self.weight = weight
        self.size_average = size_average
        self.is_sigmoid = is_sigmoid

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.float()
        targets = targets.float()

        if self.is_sigmoid:
            pass
        else:
            inputs = torch.sigmoid(inputs)

        if targets.shape != inputs.shape:
            raise AssertionError(
                f"ground truth has different shape ({targets.shape}) from input ({inputs.shape})"
            )

        # Compute the Dice loss for each class
        dice_scores = []
        num_channels = inputs.shape[1]

        for channel in range(num_channels):
            input_channel = inputs[:, channel, :, :]
            target_channel = targets[:, channel, :, :]

            # Flatten the input and target tensors for the current channel
            input_channel = input_channel.reshape(-1)
            target_channel = target_channel.reshape(-1)

            intersection = (input_channel * target_channel).sum()
            dice = (2.0 * intersection + smooth) / (
                input_channel.sum() + target_channel.sum() + smooth
            )
            dice_scores.append(dice)

        # Calculate the weighted sum of the Dice scores
        dice_scores = [
            score * weight for score, weight in zip(dice_scores, self.weight)
        ]
        dice_loss = 1 - sum(dice_scores) / num_channels

        if self.size_average:
            dice_loss = dice_loss.mean()
        else:
            dice_loss = dice_loss.sum()

        return dice_loss


class FocalLoss(nn.Module):
    """
    Computes the Focal loss between the predicted and target masks.

    Args:
        alpha (float or list): Weighting factor for the classes.
        gamma (float): Focusing parameter to scale the loss.
        size_average (bool): Whether to average the loss across all pixels.
    """

    def __init__(self, alpha=None, gamma=2.0, size_average=True, is_sigmoid=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.is_sigmoid = is_sigmoid

    def forward(self, inputs, targets):
        # Ensure inputs are in the form of probabilities
        inputs = inputs.float()
        targets = targets.float()

        if self.is_sigmoid:
            pass
        else:
            inputs = torch.sigmoid(inputs)

        BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Apply weights to the classes
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha = torch.tensor(self.alpha).expand_as(targets).to(targets.device)
            else:
                alpha = torch.tensor(self.alpha).float().to(targets.device)
                alpha = alpha.view(1, -1, 1, 1).expand_as(targets)
            
            at = alpha * targets + (1 - alpha) * (1 - targets)
            BCE_loss = at * BCE_loss

        # Calculate the Focal Loss
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.size_average:
            return F_loss.mean()
        else:
            return F_loss.sum()



class WeightedBinaryCrossEntropy(nn.Module):
    """
    Computes the weighted binary cross-entropy loss.

    Args:
        weight (list): Weight(s) to assign to each class.
        is_sigmoid (bool): Whether to apply sigmoid activation to the predicted mask.
    """

    def __init__(self, weight=[0.8, 1.2], is_sigmoid=True):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.register_buffer("class_weights", torch.tensor(weight))
        self.is_sigmoid = is_sigmoid

    def forward(self, inputs, targets):
        if self.is_sigmoid:
            pass
        else:
            inputs = torch.sigmoid(inputs)

        if targets.shape != inputs.shape:
            raise AssertionError(
                f"ground truth has different shape ({targets.shape}) from input ({inputs.shape})"
            )

        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate the binary cross-entropy loss
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction="none")

        # Apply class weights to the loss
        weight_vector = targets * self.weight[1] + (1 - targets) * self.weight[0]
        weighted_bce_loss = weight_vector * bce_loss

        # Take the mean of the weighted loss
        weighted_bce_loss = torch.mean(weighted_bce_loss)

        return weighted_bce_loss


class MultiClassWeightedBinaryCrossEntropy(nn.Module):
    """
    Computes the weighted binary cross-entropy loss for multi-class segmentation.

    Args:
        weight (list): Weight(s) to assign to each class.
        is_sigmoid (bool): Whether to apply sigmoid activation to the predicted mask.
    """

    def __init__(self, weight=None, is_sigmoid=True):
        super(MultiClassWeightedBinaryCrossEntropy, self).__init__()
        self.is_sigmoid = is_sigmoid
        self.weight = weight

    def forward(self, inputs, targets):
        num_channels = inputs.shape[1]
        if self.weight is None:
            self.weight = [
                1.0
            ] * num_channels  # Default to equal weight for single-channel input
        else:
            self.weight = self.weight

        if targets.shape != inputs.shape:
            raise AssertionError(
                f"ground truth has different shape ({targets.shape}) from input ({inputs.shape})"
            )

        if self.is_sigmoid:
            pass
        else:
            inputs = torch.sigmoid(inputs)

        # Compute the weighted binary cross-entropy loss for each channel
        bce_losses = []

        for channel in range(num_channels):
            inputs_channel = inputs[:, channel, :, :]
            targets_channel = targets[:, channel, :, :]

            # Flatten tensors for the channel
            inputs_channel = inputs_channel.reshape(-1)
            targets_channel = targets_channel.reshape(-1)

            # Calculate the binary cross-entropy loss
            bce_loss = nn.functional.binary_cross_entropy(
                inputs_channel, targets_channel, reduction="none"
            )

            # Apply class weights to the loss for the current channel
            weight_vector = (
                targets_channel * self.weight[channel]
                + (1 - targets_channel) * self.weight[channel]
            )
            weighted_bce_loss = weight_vector * bce_loss

            # Take the mean of the weighted loss for the current channel
            weighted_bce_loss = torch.mean(weighted_bce_loss)

            bce_losses.append(weighted_bce_loss)

        # Average binary cross-entropy across all channels
        multi_channel_bce_loss = sum(bce_losses) / num_channels

        return multi_channel_bce_loss


class AdjustedDiceFocalLoss(nn.Module):
    """
    Extension of MONAI's DiceFocalLoss that adds an extra penalty for false positives
    in all-background images (where ground truth contains no positive pixels).
    Assumes input tensor contains raw logits.

    Args:
        fp_penalty_weight (float): Weight factor for the false positive penalty term.
            Higher values enforce stronger penalties for predicting objects
            when none exist. Default: 10.0.
        reduction (str): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``.
        empty_mask_threshold (float): Threshold below which the target sum is considered empty.
            Default: 1e-6.
        **kwargs: Arguments to pass to MONAI's DiceFocalLoss constructor (e.g., sigmoid,
                  softmax, include_background, gamma, lambda_dice, lambda_focal).
                  Ensure these are set consistent with expecting logits as input.
    """
    def __init__(self,
                fp_penalty_weight: float = 10.0,
                reduction: Literal['none', 'mean', 'sum'] = 'mean',
                empty_mask_threshold: float = 1e-6,
                is_sigmoid: bool = False,
                **kwargs) -> None:
        super().__init__()
        # Store these values for later use
        self.include_background = kwargs.pop('include_background', False)
        self.is_softmax = kwargs.pop('softmax', False)
        self.is_sigmoid = is_sigmoid
        self.fp_penalty_weight = fp_penalty_weight
        self.reduction_mode = LossReduction(reduction)
        self.empty_mask_threshold = empty_mask_threshold
        
        # Force per-item losses from base DiceFocalLoss
        kwargs_copy = dict(kwargs)
        kwargs_copy['reduction'] = 'none'
        kwargs_copy['include_background'] = self.include_background
        kwargs_copy['softmax'] = self.is_softmax
        kwargs_copy['sigmoid'] = not self.is_sigmoid 
        
        # Initialize dice_focal with our parameters
        self.dice_focal = DiceFocalLoss(**kwargs_copy)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args:
        input: Model predictions (raw logits or sigmoid-activated). Shape (B, C, H, W, [D]).
        target: Ground truth masks. Shape (B, C, H, W, [D]).
        Returns:
        Combined loss value."""
        
        # Calculate base loss per item
        per_item_loss = self.dice_focal(input, target)

        # Identify empty masks (all background)
        spatial_dims = tuple(range(2, target.dim())) # H, W, [D]
        target_sum_per_channel = target.sum(dim=spatial_dims) # Shape: (B, C)
        is_empty = (target_sum_per_channel.sum(dim=1) <= self.empty_mask_threshold) # Shape: (B,)

        if not torch.any(is_empty):
            if self.reduction_mode == LossReduction.MEAN:
                return per_item_loss.mean()
            elif self.reduction_mode == LossReduction.SUM:
                return per_item_loss.sum()
            else: # LossReduction.NONE
                return per_item_loss
        
        # Calc. penalty for each empty mask
        penalties = torch.zeros_like(per_item_loss)
        empty_indices = torch.where(is_empty)[0]
       
        if len(empty_indices) > 0:
            empty_predictions = input[empty_indices]
            if self.is_softmax and not self.include_background:
                fg_predictions = empty_predictions[:, 1:]
            else:
                fg_predictions = empty_predictions
                
            if fg_predictions.shape[1] > 0:
                zero_target = torch.zeros_like(fg_predictions)
                
                # Apply BCE with logits or regular BCE depending on input type
                if self.is_sigmoid:
                    # If input is already sigmoid-activated, use regular BCE
                    bce_penalty_per_element = F.binary_cross_entropy(
                        fg_predictions, zero_target, reduction='none')
                else:
                    # If input is raw logits, use BCE with logits
                    bce_penalty_per_element = F.binary_cross_entropy_with_logits(
                        fg_predictions, zero_target, reduction='none')
                penalties[empty_indices] = bce_penalty_per_element * self.fp_penalty_weight
        
        # Combine and reduce        
        combined_loss_per_item = per_item_loss + penalties
        
        if self.reduction_mode == LossReduction.MEAN:
            return combined_loss_per_item.mean()
        elif self.reduction_mode == LossReduction.SUM:
            return combined_loss_per_item.sum()
        else: # LossReduction.NONE
            return combined_loss_per_item
