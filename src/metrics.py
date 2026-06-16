import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.linalg import inv

def compute_dtw_distance(signature1: np.ndarray, signature2: np.ndarray) -> float:
    """
    Compute the Dynamic Time Warping (DTW) distance between two signatures.

    Parameters:
        signature1: np.ndarray of shape (T1, 48)
        signature2: np.ndarray of shape (T2, 48)
    
    Returns:
        float: The DTW distance between the two signatures.
    """
    # Euclidean distance to compare two 48-dimensional frames
    distance, path = fastdtw(signature1, signature2, dist=euclidean)

    # Normalize by the length of the warping path to get an average distance
    # This helps to compare signatures of different lengths
    normalized_distance = distance / len(path)

    return normalized_distance

def compute_kl_divergence(signature_user: np.ndarray, signature_ref: np.ndarray) -> float:
    """
    Compute the Kullback-Leibler (KL) divergence between two signatures.

    Parameters:
        signature_user: np.ndarray of shape (T1, 48)
        signature_ref: np.ndarray of shape (T2, 48)
    
    Returns:
        float: The KL divergence between the two signatures.
    """
    # Dimensionality of the feature space
    d = signature_user.shape[1]  # Dimensionality (48)

    # 1. Compute mean for both signatures
    mean_user = np.mean(signature_user, axis=0)
    mean_ref = np.mean(signature_ref, axis=0)

    # 2. Compute covariance for both signatures
    cov_user = np.cov(signature_user, rowvar=False)
    cov_ref = np.cov(signature_ref, rowvar=False)

    # 3. Mathematic regularization
    epsilon = 1e-5
    cov_user += np.eye(d) * epsilon
    cov_ref += np.eye(d) * epsilon

    # 4. Compute KL divergence
    inv_cov_ref = inv(cov_ref)

    # Use slogdet to avoid numerical issues with determinant
    _, logdet_user = np.linalg.slogdet(cov_user)
    _, logdet_ref = np.linalg.slogdet(cov_ref)

    term1 = np.trace(inv_cov_ref @ cov_user)
    term2 = (mean_ref - mean_user).T @ inv_cov_ref @ (mean_ref - mean_user)
    term3 = logdet_ref - logdet_user

    kl_divergence = 0.5 * (term1 + term2 - d + term3)

    return max(0.0, float(kl_divergence))