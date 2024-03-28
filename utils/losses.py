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


def content_loss_vgg(vgg_model, enhanced_image, clean_image):
    # minimise
    # Extract features from the block5_conv2 layer
    features_enhanced = vgg_model(enhanced_image)
    features_clean = vgg_model(clean_image)

    # Compute the content loss
    loss_content = F.mse_loss(features_enhanced, features_clean)

    return loss_content


def content_loss_non_deep(enhanced_image, clean_image):
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
        # transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
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


def generate_negatives(positive, num_negatives, negative_transforms):

    # Apply transformations to the positive image to generate negatives
    negatives = []
    for _ in range(num_negatives):
        # Apply transformations to generate negatives
        transformed_negative = negative_transforms(positive)
        negatives.append(transformed_negative)
    return negatives


def build_batch_embedding_matrix(positive_embeddings, negative_embeddings):
    # Convert list of negative embeddings to a single tensor
    negative_embeddings_tensor = torch.stack(negative_embeddings, dim=1)

    # Combine positive and negative embeddings into a single tensor
    all_embeddings = torch.cat((positive_embeddings.unsqueeze(1), negative_embeddings_tensor), dim=1)

    return all_embeddings


def build_distance_matrix(all_embeddings):
    batch_size = all_embeddings.shape[0]
    num_positive_and_negatives = all_embeddings.shape[1]
    embedding_dimensions = all_embeddings.shape[1:]
    # Reshape the tensor to merge the first two dimensions
    merged_tensor = all_embeddings.view(-1, embedding_dimensions[0], embedding_dimensions[1], embedding_dimensions[2])

    reshaped_tensor = merged_tensor.view(batch_size, num_positive_and_negatives, -1)

    # Calculate pairwise distances
    distances = torch.cdist(reshaped_tensor, reshaped_tensor, p=2)

    distance_matrix = distances.view(batch_size, num_positive_and_negatives, num_positive_and_negatives)
    return distance_matrix


def find_extreme_negatives_recursive(all_embeddings, num_extreme_negatives, extreme_negatives_indices=None,
                                     current_distances=None):
    if extreme_negatives_indices is None:
        extreme_negatives_indices = []

    if current_distances is None:
        current_distances = build_distance_matrix(all_embeddings)

    if num_extreme_negatives == 1:
        positive_embedding = all_embeddings[:, 0]
        negative_embeddings = all_embeddings[:, 1:]

        positive_to_negatives = current_distances[:, 0, 1:]
        furthest_negative_idx = torch.argmax(positive_to_negatives).item() + 1
        return [furthest_negative_idx]

    if len(extreme_negatives_indices) == num_extreme_negatives:
        return extreme_negatives_indices

    positive_embedding = all_embeddings[:, 0]
    negative_embeddings = all_embeddings[:, 1:]

    positive_to_negatives = current_distances[:, 0, 1:]
    furthest_negative_idx = torch.argmax(positive_to_negatives).item() + 1
    furthest_negative_distance = positive_to_negatives[0, furthest_negative_idx - 1].item()

    extreme_negatives_indices.append(furthest_negative_idx)

    current_distances[:, furthest_negative_idx, :] = 0
    current_distances[:, :, furthest_negative_idx] = 0

    furthest_distance = -1
    next_negative_idx = -1

    for i in range(1, current_distances.shape[-1]):
        if i not in extreme_negatives_indices:
            distance_to_positive = current_distances[0, 0, i].item()
            min_distance_to_extremes = min(current_distances[0, extreme_negatives_indices, i].tolist())
            total_distance = distance_to_positive + min_distance_to_extremes

            if total_distance > furthest_distance:
                furthest_distance = total_distance
                next_negative_idx = i
    extreme_negatives_indices.append(next_negative_idx)

    return find_extreme_negatives_recursive(all_embeddings, num_extreme_negatives, extreme_negatives_indices,
                                            current_distances)


def find_extreme_negatives_batch(all_embeddings, num_extreme_negatives):
    extreme_negative_embeddings = []
    for embeddings_slice in all_embeddings:
        extreme_negative_indices = find_extreme_negatives_recursive(embeddings_slice.unsqueeze(0),
                                                                    num_extreme_negatives)
        extreme_negative_embeddings_slice = embeddings_slice[extreme_negative_indices]
        extreme_negative_embeddings.append(extreme_negative_embeddings_slice)

    extreme_negative_embeddings = torch.stack(extreme_negative_embeddings, dim=1)
    return extreme_negative_embeddings


def poly_loss(targets, anchors, encoder, negative_transforms, num_extreme_negatives, negative_batch_size, margin=1.0):
    # Generate negative batch
    negatives = generate_negatives(anchors, negative_batch_size, negative_transforms)

    # Encode target, anchor, and negatives
    target_embedding, _ = encoder(targets)
    anchor_embedding, _ = encoder(anchors)
    negative_embeddings = [encoder(negative)[0] for negative in negatives]

    # Build batch embedding matrix
    all_embeddings = build_batch_embedding_matrix(target_embedding, negative_embeddings)

    # Extract extreme negative embeddings
    extreme_negatives_embeddings = find_extreme_negatives_batch(all_embeddings, num_extreme_negatives)

    # Calculate contrastive loss
    loss = 0.0
    for extreme_negative_embedding in extreme_negatives_embeddings:
        # Calculate distance between anchor and positive
        distance_positive = torch.pairwise_distance(anchor_embedding, target_embedding)

        # Calculate distance between anchor and extreme negative
        distance_negative = torch.pairwise_distance(anchor_embedding, extreme_negative_embedding)

        # Contrastive loss
        loss += F.relu(distance_positive - distance_negative + margin)

    # Average the loss
    loss = loss.mean()

    return loss
