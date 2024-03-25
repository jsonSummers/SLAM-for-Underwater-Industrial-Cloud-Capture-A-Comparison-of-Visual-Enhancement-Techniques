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


def find_extreme_negatives_recursive(all_embeddings, num_extreme_negatives, extreme_negatives=None):
    if extreme_negatives is None:
        extreme_negatives = []

    # Base case: If we have found the desired number of extreme negatives, return
    if len(extreme_negatives) == num_extreme_negatives:
        print("Base case reached. Extreme negatives:", extreme_negatives)
        return extreme_negatives

    # Step 1: Build the distance matrix
    distance_matrix = build_distance_matrix(all_embeddings)
    print('Distance matrix:', distance_matrix.shape)

    # Get distances between positive and negatives
    pos_to_neg_distances = distance_matrix[0, 1:]
    print('Positive to negative distances:', pos_to_neg_distances)

    # Ensure the current negative has infinite distance from itself and any selected negatives
    for neg_index in extreme_negatives:
        pos_to_neg_distances[neg_index] = float('inf')

    # Find the furthest negative embedding
    furthest_negative_index = torch.argmax(pos_to_neg_distances)
    print('Furthest negative index:', furthest_negative_index)
    extreme_negatives.append(furthest_negative_index)

    # Recur with the updated list of extreme negatives
    print('Extreme negatives so far:', extreme_negatives)
    return find_extreme_negatives_recursive(all_embeddings, num_extreme_negatives, extreme_negatives)


def find_extreme_negatives_batch(all_embeddings, num_extreme_negatives):
    batch_size = all_embeddings.size(0)
    extreme_negative_embeddings_batch = []
    for i in range(batch_size):
        slice = all_embeddings[i]
        print('Processing slice', i)
        extreme_negative_embeddings = find_extreme_negatives_recursive(slice.unsqueeze(0), num_extreme_negatives)
        extreme_negative_embeddings = torch.stack(extreme_negative_embeddings, dim=0)  # Convert the list to a tensor
        extreme_negative_embeddings_batch.append(extreme_negative_embeddings.squeeze(0))  # Remove the batch dimension
    return torch.stack(extreme_negative_embeddings_batch, dim=0)


def poly_loss(targets, anchors, encoder, num_extreme_negatives, negative_batch_size, margin=1.0):
    # Generate negative batch
    negatives = generate_negatives(anchors, negative_batch_size)

    # Encode target, anchor, and negatives
    target_embedding, _ = encoder(targets)
    anchor_embedding, _ = encoder(anchors)
    negative_embeddings = [encoder(negative)[0] for negative in negatives]

    # Build batch embedding matrix
    all_embeddings = build_batch_embedding_matrix(target_embedding, negative_embeddings)

    # Extract extreme negative embeddings
    extreme_negatives_embeddings = find_extreme_negatives_batch(all_embeddings, num_extreme_negatives)
    print('Extreme negatives:', len(extreme_negatives_embeddings))

    # Calculate contrastive loss
    loss = 0.0
    for extreme_negative_embedding in extreme_negatives_embeddings:
        print('Extreme negative:', extreme_negative_embedding.shape)
        print('Anchor:', anchor_embedding.shape)
        # Calculate distance between anchor and positive
        distance_positive = torch.pairwise_distance(anchor_embedding, target_embedding)

        # Calculate distance between anchor and extreme negative
        distance_negative = torch.pairwise_distance(anchor_embedding, extreme_negative_embedding)

        # Contrastive loss
        loss += F.relu(distance_positive - distance_negative + margin)

    # Average the loss
    loss = loss.mean()

    return loss