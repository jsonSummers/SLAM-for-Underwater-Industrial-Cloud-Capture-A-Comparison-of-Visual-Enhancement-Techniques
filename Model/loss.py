import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        loss = torch.mean(F.relu(distance_positive - distance_negative + self.margin))
        return loss


class QuadrupletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(QuadrupletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative1, negative2):
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative1 = F.pairwise_distance(anchor, negative1)
        distance_negative2 = F.pairwise_distance(anchor, negative2)
        loss = torch.mean(F.relu(distance_positive - distance_negative1 + self.margin))
        loss += torch.mean(F.relu(distance_positive - distance_negative2 + self.margin))
        return loss


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, output, target):
        return F.mse_loss(output, target)
