import torch.nn.functional as F
import torch


def triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = F.pairwise_distance(anchor, positive)
    distance_negative = F.pairwise_distance(anchor, negative)
    loss = F.relu(distance_positive - distance_negative + margin)
    return loss.mean()


def adversarial_loss(logits, is_real):
    targets = torch.ones_like(logits) if is_real else torch.zeros_like(logits)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss
