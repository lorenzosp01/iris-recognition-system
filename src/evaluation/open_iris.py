import json
import os

from src.lib.cnn_utils import identification_test_all_vs_all, identification_test_probe_vs_gallery, \
    verification_probe_vs_gallery, verification_all_vs_all
from src.utils.plotting import plot_far_frr_roc


def save_results(results, filename="results.json"):
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

def load_results(filename="results.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    import numpy as np

    matrix = np.loadtxt('distance_matrix_3724.csv', delimiter=',')

    with open('labels.txt', 'r') as f:
        labels = f.readlines()

    labels = [label.strip() for label in labels]

    rows_to_keep = np.where(np.diag(matrix) == 0)[0]

    if len(rows_to_keep) == 0:
        filtered_matrix = np.zeros((0, 0), dtype=matrix.dtype)
    else:
        filtered_matrix = matrix[rows_to_keep, :]
        filtered_matrix = filtered_matrix[:, rows_to_keep]

    labels = np.array([labels[i] for i in rows_to_keep])

    thresholds, DIR, GRR, FAR, FRR = identification_test_all_vs_all(filtered_matrix, labels, log=True)

    plot_far_frr_roc(thresholds, FAR, FRR, GRR, DIR=np.array(DIR))

    GARs, FARs, FRRs, GRRs = verification_all_vs_all(filtered_matrix, labels, log=True)

    plot_far_frr_roc(thresholds, FARs, FRRs, GRRs, DIR=None)

    N = len(filtered_matrix)
    indices = np.arange(N)
    np.random.shuffle(indices)

    split_point = int(0.6 * N)
    gallery_indices = indices[:split_point]
    probe_indices = indices[split_point:]

    labels_gallery = labels[gallery_indices]
    labels_probe = labels[probe_indices]

    probe_vs_gallery = filtered_matrix[np.ix_(probe_indices, gallery_indices)]

    thresholds, DIR, GRR, FAR, FRR = identification_test_probe_vs_gallery(probe_vs_gallery, labels_probe, labels_gallery, log=True)

    plot_far_frr_roc(thresholds, FAR, FRR, GRR, DIR=np.array(DIR))

    verification_probe_vs_gallery(probe_vs_gallery, labels_probe, labels_gallery, log=True)

    plot_far_frr_roc(thresholds, FAR, FRR, GRR, DIR=None)



