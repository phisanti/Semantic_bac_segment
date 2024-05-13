import torch
import torch.nn as nn
from semantic_bac_segment.utils import tensor_debugger

class MaxDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, is_sigmoid=True):
        super(MaxDiceLoss, self).__init__()
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
        # Flatten all channels into one with the maximum value of each channel
        inputs = torch.max(inputs, dim=1, keepdim=True)[0]
        targets = torch.max(targets, dim=1, keepdim=True)[0]

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


class MultiClassDiceLoss(nn.Module):
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
            dice = (2. * intersection + smooth) / (input_channel.sum() + target_channel.sum() + smooth)
            dice_scores.append(dice)


        # Calculate the weighted sum of the Dice scores
        dice_scores = [score * weight for score, weight in zip(dice_scores, self.weight)]
        dice_loss = 1 - sum(dice_scores) / num_channels

        if self.size_average:
            dice_loss = dice_loss.mean()
        else:
            dice_loss = dice_loss.sum()

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


class MultiClassWeightedBinaryCrossEntropy(nn.Module):
    def __init__(self, class_weights=None, is_sigmoid=True):
        super(MultiClassWeightedBinaryCrossEntropy, self).__init__()
        self.is_sigmoid = is_sigmoid
        self.class_weights = class_weights

    def forward(self, output, target):

        num_channels = output.shape[1]
        if self.class_weights is None:
            self.class_weights = [1.0] * num_channels  # Default to equal weight for single-channel input
        else:
            self.class_weights = self.class_weights


        # Apply sigmoid activation to the output
        if self.is_sigmoid:
            pass
        else:
            output = torch.sigmoid(output)

        # Compute the weighted binary cross-entropy loss for each channel
        bce_losses = []

        for channel in range(num_channels):
            output_channel = output[:, channel, :, :]
            target_channel = target[:, channel, :, :]

            # Flatten the output and target tensors for the current channel
            output_channel = output_channel.reshape(-1)
            target_channel = target_channel.reshape(-1)

            # Calculate the binary cross-entropy loss for the current channel
            bce_loss = nn.functional.binary_cross_entropy(output_channel, target_channel, reduction='none')

            # Apply class weights to the loss for the current channel
            weight_vector = target_channel * self.class_weights[channel] + (1 - target_channel) * self.class_weights[channel]
            weighted_bce_loss = weight_vector * bce_loss

            # Take the mean of the weighted loss for the current channel
            weighted_bce_loss = torch.mean(weighted_bce_loss)

            bce_losses.append(weighted_bce_loss)

        # Calculate the average of the weighted binary cross-entropy losses across all channels
        multi_channel_bce_loss = sum(bce_losses) / num_channels

        return multi_channel_bce_loss
