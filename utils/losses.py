import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.pairwise_distance(anchor, positive, p=2)
        distance_negative = torch.pairwise_distance(anchor, negative, p=2)
        loss = torch.clamp(self.margin + distance_positive - distance_negative, min=0.0).mean()
        return loss

class QuadrupletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(QuadrupletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative1, negative2):
        distance_positive = torch.pairwise_distance(anchor, positive, p=2)
        distance_negative1 = torch.pairwise_distance(anchor, negative1, p=2)
        distance_negative2 = torch.pairwise_distance(anchor, negative2, p=2)

        loss = torch.clamp(self.margin + distance_positive - distance_negative1 - distance_negative2, min=0.0).mean()
        return loss
