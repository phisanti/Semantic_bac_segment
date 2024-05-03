import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, is_sigmoid=True):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.is_sigmoid = is_sigmoid

    def forward(self, inputs, targets, smooth=1):
        
        inputs = inputs.float()
        targets = targets.float()
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        if self.is_sigmoid:
            pass
        else:
            inputs = torch.sigmoid(inputs)
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        

        if self.weight is not None:
            dice = dice * self.weight

        if self.size_average:
            dice = dice.mean()
        else:
            dice = dice.sum()
        
        return 1 - dice


class weightedDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, is_sigmoid=True, zero_weight=0.5, edge_weight=2.0):
        super(weightedDiceLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.is_sigmoid = is_sigmoid
        self.zero_weight = zero_weight
        self.edge_weight = edge_weight

    def forward(self, inputs, targets, smooth=.1):
        inputs = inputs.float()
        targets = targets.float()

        if self.is_sigmoid:
            pass
        else:
            inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Weight lower the predictions where the target is 0
        zero_mask = (targets == 0)
        inputs = inputs * torch.where(zero_mask, self.zero_weight, 1)

        # Reward edges and borders of positive features
        edge_mask = self.get_edge_mask(targets)
        inputs = inputs * torch.where(edge_mask, self.edge_weight, 1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

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


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from semantic_bac_segment.utils import tensor_debugger 
class MultiClassDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, is_sigmoid=True, ignore_index=-100):
        super(MultiClassDiceLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.is_sigmoid = is_sigmoid
        self.ignore_index = ignore_index
    def forward(self, inputs, targets, smooth=1):
        if self.is_sigmoid:
            inputs = torch.sigmoid(inputs)
        
        # Flatten the input and target tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Ignore the samples with the ignore_index
        valid_indices = targets != self.ignore_index
        inputs = inputs[valid_indices]
        targets = targets[valid_indices]

        # Compute the Dice loss for each class
        dice_losses = []
        num_classes = inputs.unique().size(0)
        for class_id in range(num_classes):
            class_inputs = (inputs == class_id).float()
            class_targets = (targets == class_id).float()
            intersection = (class_inputs * class_targets).sum()
            dice = (2. * intersection + smooth) / (class_inputs.sum() + class_targets.sum() + smooth)
            dice_losses.append(1 - dice)

        # Take the average or sum of the Dice losses
        if self.size_average:
            dice_loss = torch.mean(torch.stack(dice_losses))
        else:
            dice_loss = torch.sum(torch.stack(dice_losses))

        return dice_loss


class WeightedBinaryCrossEntropy(nn.Module):
    def __init__(self, class_weights = [0.8, 1.2], is_sigmoid=True):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.register_buffer('class_weights', torch.tensor(class_weights))
        self.is_sigmoid = is_sigmoid

    def forward(self, output, target):
        # Apply sigmoid activation to the output
        if self.is_sigmoid:
            pass
        else:
            output = torch.sigmoid(output)
        
        # Flatten the output and target tensors
        output = output.view(-1)
        target = target.view(-1)
        
        # Calculate the binary cross-entropy loss
        bce_loss = nn.functional.binary_cross_entropy(output, target, reduction='none')
        
        # Apply class weights to the loss
        weight_vector = target * self.class_weights[1] + (1 - target) * self.class_weights[0]
        weighted_bce_loss = weight_vector * bce_loss
        
        # Take the mean of the weighted loss
        weighted_bce_loss = torch.mean(weighted_bce_loss)
        
        return weighted_bce_loss