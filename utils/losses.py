import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = F.pairwise_distance(anchor, positive)
    distance_negative = F.pairwise_distance(anchor, negative)
    loss = F.relu(distance_positive - distance_negative + margin)
    return loss.mean()

