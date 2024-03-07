# losses.py

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


def adversarial_loss(predictions, is_real):
    # minimise
    targets = torch.ones_like(predictions) if is_real else torch.zeros_like(predictions)
    loss = F.binary_cross_entropy_with_logits(predictions, targets)
    return loss


def l1_loss(output, target):
    # minimise
    loss = F.l1_loss(output, target)
    return loss


def content_loss(vgg_model, enhanced_image, clean_image):
    # minimise
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
    # maximise
    distance_positive = F.pairwise_distance(anchor, positive)
    distance_negative = F.pairwise_distance(anchor, negative)

    # Compute the triplet loss
    loss_triplet = F.relu(margin + distance_negative - distance_positive)

    return loss_triplet.mean()


def poly_loss(anchor, positive, number_of_negatives, margin=1.0):
    # Compute the distance between anchor and positive
    distance_positive = F.pairwise_distance(anchor, positive).unsqueeze(1)

    # Generate negatives by applying random transformations to the positive
    random_transforms = transforms.Compose([
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    ])

    negatives = [random_transforms(positive) for _ in range(number_of_negatives)]

    # Compute the distances between anchor and negatives
    distances_negative = [F.pairwise_distance(anchor, negative).unsqueeze(1) for negative in negatives]

    # Compute the poly loss
    loss = torch.tensor(0.0, device=anchor.device)
    margin_tensor = torch.tensor(margin, device=anchor.device).expand_as(distance_positive)

    # Compute the poly loss
    losses = [torch.max(distance_positive - dist_neg + margin_tensor, torch.tensor(0.0, device=anchor.device)) for dist_neg in distances_negative]
    loss += torch.mean(torch.stack(losses))

    return loss