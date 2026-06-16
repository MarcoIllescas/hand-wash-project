import numpy as np
from src.metrics import compute_dtw_distance, compute_kl_divergence

sig_gold_standard = np.load("data/signatures/hw_00002.npy")
sig_user = np.load("data/signatures/hw_00004.npy")

# =================================================================== #
#               Distance Gold Standard vs Gold Standard               #
# =================================================================== #
dtw_distance_itself = compute_dtw_distance(sig_gold_standard, sig_gold_standard)
print(f"DTW Distance between the same signature: {dtw_distance_itself:.4f}")

kl_divergence_itself = compute_kl_divergence(sig_gold_standard, sig_gold_standard)
print(f"KL Divergence between the same signature: {kl_divergence_itself:.4f}")


# =================================================================== #
#               Distance Gold Standard vs User Signature              #
# =================================================================== #
dtw_distance = compute_dtw_distance(sig_gold_standard, sig_user)
print(f"DTW Distance between gold standard and user signature: {dtw_distance:.4f}")

kl_divergence = compute_kl_divergence(sig_gold_standard, sig_user)
print(f"KL Divergence between gold standard and user signature: {kl_divergence:.4f}")