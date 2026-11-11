import torch
from scipy.optimize import linear_sum_assignment

def calculate_cluster_centers(features, labels, weight=None, num_classes=9,lambda_dist=0.1):













    if weight is None:
        weight = torch.ones_like(labels)

    B, C, H, W = features.shape

    features = features.view(B, C, -1)
    features = features.permute(0, 2, 1)
    features = features.contiguous().view(-1, C)
    labels = labels.view(-1)
    weight = weight.view(-1)
    mask1 = (weight == 1)
    features=features[mask1]
    labels = labels[mask1]
    centroids = torch.zeros(num_classes, C).to(features.device)
    variances = torch.zeros(num_classes, C).to(features.device)

    for class_idx in range(num_classes):
        mask = (labels == class_idx)
        class_features = features[mask]


        if class_features.size(0) > 0:
            class_mean = class_features.mean(dim=0)
            centroids[class_idx] = class_mean

            variance = ((class_features - class_mean) ** 2).mean(dim=0)
            variances[class_idx] = variance

    return centroids, variances


def hungarian_assign(centers, empirical_centers):

    distances = torch.cdist(empirical_centers, centers, p=2).detach()
    rows, columns = linear_sum_assignment(distances.cpu().numpy())
    result = empirical_centers.clone()
    row_idx = torch.as_tensor(rows, device=result.device, dtype=torch.long)
    col_idx = torch.as_tensor(columns, device=centers.device, dtype=torch.long)
    result[row_idx] = centers[col_idx]
    return result.detach()


def empirical_centers(features, labels, num_classes, mask=None):

    means, _ = calculate_cluster_centers(
        features, labels, weight=mask, num_classes=int(num_classes))
    return means.detach()
