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


def content_loss_old(vgg_model, enhanced_image, clean_image):
    # minimise
    # Extract features from the block5_conv2 layer
    features_enhanced = vgg_model(enhanced_image)
    features_clean = vgg_model(clean_image)

    # Compute the content loss
    loss_content = F.mse_loss(features_enhanced, features_clean)

    return loss_content


def content_loss(enhanced_image, clean_image):
    # Compute the content loss using mean squared error (MSE)
    loss_content = F.mse_loss(enhanced_image, clean_image)
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


def poly_loss_old(anchor, positive, number_of_negatives, margin=1.0):
    # Compute the distance between anchor and positive
    distance_positive = F.pairwise_distance(anchor, positive).unsqueeze(1)

    # Generate negatives by applying random transformations to the positive
    random_transforms = transforms.Compose([
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    ])

    negatives = [random_transforms(positive) for _ in range(number_of_negatives)]

    # Compute the distances between anchor and negatives
    distances_negative = [F.pairwise_distance(anchor, negative).unsqueeze(1) for negative in negatives]

    # Compute the poly loss
    loss = torch.tensor(0.0, device=anchor.device)
    margin_tensor = torch.tensor(margin, device=anchor.device).expand_as(distance_positive)

    # Compute the poly loss
    losses = [torch.max(distance_positive - dist_neg + margin_tensor, torch.tensor(0.0, device=anchor.device)) for
              dist_neg in distances_negative]
    loss += torch.mean(torch.stack(losses))

    return loss


def poly_loss_old2(anchor, positive, encoder, num_negatives, margin=1.0):
    # Encode anchor and positive images
    anchor_embedding, _ = encoder(anchor)
    positive_embedding, _ = encoder(positive)

    # Compute the distance between anchor and positive embeddings
    distance_positive = F.pairwise_distance(anchor_embedding, positive_embedding).unsqueeze(1)

    # Generate negatives by applying random transformations to the positive
    random_transforms = transforms.Compose([
        #transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        #transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    ])

    negatives = [random_transforms(positive) for _ in range(num_negatives)]

    # Compute embeddings for negative images
    negative_embeddings = [encoder(negative)[0] for negative in negatives]

    # Compute the distances between anchor and negative embeddings
    distances_negative = [F.pairwise_distance(anchor_embedding, neg_emb).unsqueeze(1) for neg_emb in
                          negative_embeddings]

    # Compute the poly loss
    loss = torch.tensor(0.0, device=anchor.device)
    margin_tensor = torch.tensor(margin, device=anchor.device).expand_as(distance_positive)

    # Compute the poly loss for each negative
    for dist_neg in distances_negative:
        losses = torch.max(distance_positive - dist_neg + margin_tensor, torch.tensor(0.0, device=anchor.device))
        loss += torch.mean(losses)

    # Average the loss over all negatives
    loss /= num_negatives

    return loss

############################################################################################################

def generate_negatives(positive, num_negatives):
    # Define the random transformations
    random_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    ])

    # Apply transformations to the positive image to generate negatives
    negatives = []
    for _ in range(num_negatives):
        # Apply transformations to generate negatives
        transformed_negative = random_transforms(positive)
        negatives.append(transformed_negative)
    return negatives


def build_batch_embedding_matrix(positive_embeddings, negative_embeddings):
    # Convert list of negative embeddings to a single tensor
    negative_embeddings_tensor = torch.stack(negative_embeddings, dim=1)

    # Combine positive and negative embeddings into a single tensor
    all_embeddings = torch.cat((positive_embeddings.unsqueeze(1), negative_embeddings_tensor), dim=1)
    print('all_embeddings shape:', all_embeddings.shape)
    return all_embeddings


def build_distance_matrix(positive_embeddings, negative_embeddings, num_extreme_negatives):
    all_embeddings = build_batch_embedding_matrix(positive_embeddings, negative_embeddings)

    batch_size, num_samples, embedding_size = all_embeddings.size()

    # Reshape embeddings to make computations easier
    reshaped_embeddings = all_embeddings.view(batch_size * num_samples, embedding_size)

    # Compute pairwise distances using broadcasting
    pairwise_distances = torch.cdist(reshaped_embeddings, reshaped_embeddings)

    # Reshape the pairwise distances back to a 4D tensor
    pairwise_distances = pairwise_distances.view(batch_size, num_samples, batch_size, num_samples)

    # Print for debugging
    print('pairwise_distances shape:', pairwise_distances.shape)

    # Now we need to compute the distances for each pair of embeddings for each slice
    # In the batch to find the extreme negatives
    extreme_negatives = []
    for i in range(batch_size):
        distances_slice = pairwise_distances[i]
        extreme_negatives_slice = find_extreme_negatives(distances_slice, num_extreme_negatives)
        extreme_negatives.append(extreme_negatives_slice)

    extreme_negatives = torch.stack(extreme_negatives, dim=0)

    return extreme_negatives

    return extreme_negatives


def find_extreme_negatives(distance_matrix, num_extreme_negatives):
    return distance_matrix  # Placeholder for actual implementation


def poly_loss(targets, anchors, encoder, num_extreme_negatives, num_negatives):

    # Generate negative batch
    negatives = generate_negatives(anchors, num_negatives)

    # Encode target, anchor, and negatives
    target_embedding, _ = encoder(targets)
    anchor_embedding, _ = encoder(anchors)
    negative_embeddings = [encoder(negative)[0] for negative in negatives]

    # Build distance matrix
    distance_matrix = build_distance_matrix(anchor_embedding, negative_embeddings, num_extreme_negatives)
    print('distance_matrix shape:', distance_matrix.shape)

    # Find extreme negatives
    extreme_negative_distances = find_extreme_negatives(distance_matrix, num_extreme_negatives)

    # Compute distances
    positive_distance = F.pairwise_distance(anchor_embedding, target_embedding)
    negative_distances = extreme_negative_distances

    # Compute loss
    loss = positive_distance - torch.min(negative_distances, dim=1)[0]
    loss = torch.clamp(loss, min=0.0)  # Ensure non-negative loss
    return loss.mean()