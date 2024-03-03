# losses.py

import torch
import torch.nn.functional as F
import torchvision.models as models

def adversarial_loss(predictions, is_real):
    targets = torch.ones_like(predictions) if is_real else torch.zeros_like(predictions)
    loss = F.binary_cross_entropy_with_logits(predictions, targets)
    return loss


def l1_loss(output, target):
    loss = F.l1_loss(output, target)
    return loss


def content_loss(vgg_model, enhanced_image, clean_image):
    # Extract features from the block5_conv2 layer
    features_enhanced = vgg_model(enhanced_image)
    features_clean = vgg_model(clean_image)

    # Compute the content loss
    loss_content = F.mse_loss(features_enhanced, features_clean)

    return loss_content

def make_content_loss(vgg_weights_path, device):
    vgg_model = models.vgg19(pretrained=False)
    vgg_model.load_state_dict(torch.load(vgg_weights_path))
    vgg_model.eval().to(device)

def triplet_loss(anchor, positive, negative, margin=1.0):
    # Compute the Euclidean distances
    distance_positive = F.pairwise_distance(anchor, positive)
    distance_negative = F.pairwise_distance(anchor, negative)

    # Compute the triplet loss
    loss_triplet = F.relu(distance_positive - distance_negative + margin)

    return loss_triplet.mean()