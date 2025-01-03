
import numpy as np
import math 
import torch

def torch_uniformity(features_modality1, features_modality2):
    """
    Calculate the uniformity metric for two modalities based on their features.

    Args:
        features_modality1 (torch.Tensor): Feature matrix of modality 1 with shape [bs, d].
        features_modality2 (torch.Tensor): Feature matrix of modality 2 with shape [bs, d].

    Returns:
        float: Uniformity metric (-W2).
    """
    # Concatenate the features of the two modalities
    x = torch.cat([features_modality1, features_modality2], dim=0)  # Shape: [2 * bs, d]

    N = x.size(0)
    # Compute the sample mean \mu_hat and covariance \Sigma
    mu_hat = torch.mean(x, dim=0)  # Shape: [d]
    Sigma = (x - mu_hat).T @ (x - mu_hat) / N

    # Calculate the trace and square root of the covariance matrix
    trace_Sigma = torch.trace(Sigma)  # Scalar
    # Compute the matrix square root using eigenvalue decomposition
    eigvals, eigvecs = torch.linalg.eigh(Sigma)  # Symmetric matrix decomposition
    eigvals = eigvals + 1e-8  # Add epsilon before taking the square root
    sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=0))  # Ensure non-negative eigenvalues
    sqrt_Sigma = eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T  # Reconstruct the square root matrix

    trace_sqrt_Sigma = torch.trace(sqrt_Sigma)  # Scalar

    # Dimensionality of the features
    m = x.shape[1]

    # Compute the quadratic Wasserstein distance W2
    W2 = torch.sqrt(
        torch.norm(mu_hat)**2 + 1 + trace_Sigma - (2 / torch.sqrt(torch.tensor(m, dtype=Sigma.dtype))) * trace_sqrt_Sigma
    )

    # Return the uniformity metric (-W2)
    return -W2.item()


def numpy_uniformity(features_modality1, features_modality2):
    x = torch.cat([features_modality1, features_modality2], dim=0)
    N = x.size(0)
    dim = x.size(1)

    x_center = torch.mean(x, dim=0, keepdim=True)
    covariance = torch.mm((x - x_center).t(), x - x_center) / N

    mean =  x.mean(0)
    np_mean = mean.data.cpu().numpy()
    np_covariance = covariance.data.cpu().numpy()
   
    ##calculation of part1
    part1 = np.sum(np.multiply(np_mean, np_mean))

    ##calculation of part2
    eps = 1e-8 
    S, Q = np.linalg.eig(np_covariance)
    S = S + eps

    mS = np.sqrt(np.diag(S.clip(min=0)))

    covariance_2 = np.dot(np.dot(Q, mS), Q.T)

    part2 = np.trace(np_covariance - 2.0/np.sqrt(dim) * covariance_2)
    wasserstein_distance = math.sqrt(part1 + 1 + part2)
    return -wasserstein_distance 



# Example usage

bs, d = 64, 512
features_modality1 = torch.randn(bs, d)
features_modality2 = torch.randn(bs, d)

uniformity = numpy_uniformity(features_modality1, features_modality2)

uniformity2 = torch_uniformity(features_modality1, features_modality2)

print(uniformity)
print(uniformity2)