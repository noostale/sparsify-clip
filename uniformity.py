
import numpy as np
import math 
import torch

def torch_uniformity1(features_modality1):
    """
    Calculate the uniformity metric for two modalities based on their features.

    Args:
        features_modality1 (torch.Tensor): Feature matrix of modality 1 with shape [bs, d].
        features_modality2 (torch.Tensor): Feature matrix of modality 2 with shape [bs, d].

    Returns:
        float: Uniformity metric (-W2).
    """
    # Concatenate the features of the two modalities
    #x = torch.cat([features_modality1, features_modality2], dim=0)  # Shape: [2 * bs, d]
    x = features_modality1

    N = x.size(0)
    # Compute the sample mean \mu_hat and covariance \Sigma
    #mu_hat = torch.mean(x, dim=0)  # Shape: [d]
    #Sigma = (x - mu_hat).T @ (x - mu_hat) / N
    mu_hat = torch.mean(x, dim=0, keepdim=True)
    Sigma = torch.mm((x - mu_hat).t(), x - mu_hat) / N
    #Sigma = Sigma + 1e-4

    # Calculate the trace and square root of the covariance matrix
    trace_Sigma = torch.trace(Sigma)  # Scalar
    #print(trace_Sigma)
    trace_Sigma = torch.clamp(trace_Sigma, min=0)  # Ensure non-negative trace
    # Compute the matrix square root using eigenvalue decomposition
    #eigvals, eigvecs = torch.linalg.eigh(Sigma)  # Symmetric matrix decomposition
    eigvecs, eigvals , _ = torch.linalg.svd(Sigma)  # Symmetric matrix decomposition
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
    return W2

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
    mu_hat = torch.mean(x, dim=0, keepdim=True)
    Sigma = torch.mm((x - mu_hat).t(), x - mu_hat) / N

    Sigma = Sigma + 1e-6

    #print(Sigma.shape)
    #mu_hat = torch.mean(x, dim=0)  # Shape: [d]
    #Sigma = (x - mu_hat).T @ (x - mu_hat) / N

    # Calculate the trace and square root of the covariance matrix
    trace_Sigma = torch.trace(Sigma)  # Scalar
    #trace_Sigma += 1e-8
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
    return -W2


def numpy_uniformity(features_modality1, features_modality2):
    x = torch.cat([features_modality1, features_modality2], dim=0)
    N = x.size(0)
    dim = x.size(1)

    x_center = torch.mean(x, dim=0, keepdim=True)
    covariance = torch.mm((x - x_center).t(), x - x_center) / N
    print(covariance.shape)

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



import torch
import math

import torch
import math

def torch_uniformity_equivalent(features_modality1):
    # Concatenate the input features along the first dimension (rows)
    x = features_modality1#torch.cat([features_modality1, features_modality2], dim=0)
    
    # Get the dimensions of x
    N = x.size(0)
    dim = x.size(1)

    # Compute the mean of x along the first dimension (axis 0)
    x_center = torch.mean(x, dim=0, keepdim=True)
    
    # Compute the covariance matrix
    covariance = torch.mm((x - x_center).t(), x - x_center) / N

    # Mean of the features along the first dimension
    mean = x.mean(0)
    
    # Calculation of part1
    part1 = torch.sum(mean * mean)

    # Calculation of part2:
    eps = 1e-8

    # Eigen decomposition in PyTorch (use real part of eigenvalues and eigenvectors)
    eigenvalues, eigenvectors = torch.linalg.eig(covariance)

    # Ensure eigenvalues and eigenvectors are real and move them to the same device
    eigenvalues = eigenvalues.real + eps  # Add epsilon for stability
    eigenvectors = eigenvectors.real  # Use real part of eigenvectors

    # Get the diagonal matrix of square roots of eigenvalues
    mS = torch.sqrt(torch.diag(torch.clamp(eigenvalues, min=0)))

    # Reconstruct the covariance_2 matrix using real parts
    covariance_2 = torch.mm(torch.mm(eigenvectors, mS), eigenvectors.t())

    # Part2 calculation
    part2 = torch.trace(covariance - 2.0 / math.sqrt(dim) * covariance_2)
    
    # Wasserstein distance
    wasserstein_distance = torch.sqrt(part1 + 1 + part2)
    
    return wasserstein_distance

def uniformity10( z1):
    z = z1 #torch.cat((z1, z2), 0)
    N = z.size(0)
    D = z.size(1)
    z_center = torch.mean(z, dim=0, keepdim=True)
    mean = z.mean(0)
    covariance = torch.mm((z-z_center).t(), z-z_center)/N

    #############calculation of part1
    part1 = torch.sum(torch.multiply(mean, mean))

    ######################################################
    S, Q = torch.linalg.eig(covariance)#torch.eig(covariance, eigenvectors=True)

    #print(S)
    S = torch.abs(S)#[:,0])
    Q = torch.abs(Q)
    mS = torch.sqrt(torch.diag(S))
    covariance2 = torch.mm(torch.mm(Q, mS), Q.T)

    #############calculation of part2
    part2 = torch.trace(covariance - 2.0/math.sqrt(D)*covariance2)
    wasserstein_loss = torch.sqrt(part1+1+part2)
    return wasserstein_loss
# Example usage

bs, d = 256, 512
features_modality1 = torch.randn(bs, d).cuda()
features_modality2 = torch.randn(bs, d).cuda()

uniformity = numpy_uniformity(features_modality1, features_modality2)

uniformity2 = torch_uniformity(features_modality1, features_modality2)

uniformity3 = torch_uniformity_equivalent(features_modality1)

uniformity4 = torch_uniformity1(features_modality1)

uniformity_real = uniformity10(features_modality1)

print(uniformity)
print(uniformity2)
print(uniformity3)
print(uniformity4)
print(uniformity_real)