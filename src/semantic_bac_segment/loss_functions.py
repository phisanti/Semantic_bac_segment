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