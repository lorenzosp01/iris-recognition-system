import json
import os

from src.lib.cnn_utils import identification_test_all_vs_all, identification_test_probe_vs_gallery, \
    verification_probe_vs_gallery, verification_all_vs_all
from src.utils.plotting import plot_far_frr_roc


def save_results(results, filename="results.json"):
    """Save results to a JSON file."""
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

def load_results(filename="results.json"):
    """Load results from a JSON file."""
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    import numpy as np

    # Load matrix from CSV
    matrix = np.loadtxt('distance_matrix_3724.csv', delimiter=',')

    # Check the shape of the loaded matrix
    print(matrix.shape)

    # Load labels from file
    with open('labels.txt', 'r') as f:
        labels = f.readlines()

    # Remove newline characters
    labels = [label.strip() for label in labels]

    # Identify rows to keep where the diagonal element is zero
    rows_to_keep = np.where(np.diag(matrix) == 0)[0]

    # If no rows meet the condition, return an empty square matrix
    if len(rows_to_keep) == 0:
        filtered_matrix = np.zeros((0, 0), dtype=matrix.dtype)
    else:
        # Filter the matrix
        filtered_matrix = matrix[rows_to_keep, :]
        # Ensure it is square by keeping only the relevant columns
        filtered_matrix = filtered_matrix[:, rows_to_keep]

    labels = np.array([labels[i] for i in rows_to_keep])

    thresholds, DIR, GRR, FAR, FRR = identification_test_all_vs_all(filtered_matrix, labels, log=True)

    plot_far_frr_roc(thresholds, FAR, FRR, GRR, DIR=np.array(DIR))

    GARs, FARs, FRRs, GRRs = verification_all_vs_all(filtered_matrix, labels, log=True)

    plot_far_frr_roc(thresholds, FARs, FRRs, GRRs, DIR=None)

    # Perform identification test
    # Probe vs gallery
    # Example: Create a mock distance matrix (N x N)

    # Step 1: Split indices into Gallery (60%) and Probe (40%)
    N = len(filtered_matrix)
    indices = np.arange(N)
    np.random.shuffle(indices)  # Shuffle indices to randomize splitting

    split_point = int(0.6 * N)  # Calculate the 60% split point
    gallery_indices = indices[:split_point]  # First 60% indices
    probe_indices = indices[split_point:]  # Remaining 40% indices

    labels_gallery = labels[gallery_indices]
    labels_probe = labels[probe_indices]
    # Step 2: Extract sub-matrices
    probe_vs_gallery = filtered_matrix[np.ix_(probe_indices, gallery_indices)]

    # Step 3: Perform identification test
    thresholds, DIR, GRR, FAR, FRR = identification_test_probe_vs_gallery(probe_vs_gallery, labels_probe, labels_gallery, log=True)

    # Step 4: Plot the results
    plot_far_frr_roc(thresholds, FAR, FRR, GRR, DIR=np.array(DIR))

    # Step 5: Perform verification test
    verification_probe_vs_gallery(probe_vs_gallery, labels_probe, labels_gallery, log=True)

    # Step 6: Plot the results
    plot_far_frr_roc(thresholds, FAR, FRR, GRR, DIR=None)



