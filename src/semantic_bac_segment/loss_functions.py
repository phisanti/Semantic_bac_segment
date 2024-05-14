import torch
import torch.nn as nn
from semantic_bac_segment.utils import tensor_debugger


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
