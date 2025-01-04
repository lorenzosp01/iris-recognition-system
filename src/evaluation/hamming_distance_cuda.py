import json
import os
from typing import Tuple

import numpy as np
import torch
from iris import IrisTemplate
from tqdm import tqdm


def simple_hamming_distance(
        template_probe: IrisTemplate,
        template_gallery: IrisTemplate,
        rotation_shift: int = 15,
        normalise: bool = False,
        norm_mean: float = 0.45,
        norm_nb_bits: float = 12288,
) -> Tuple[float, int]:
    """Compute Hamming distance with tensors, without bells and whistles.

    Args:
        template_probe (IrisTemplate): Iris template from probe (tensors).
        template_gallery (IrisTemplate): Iris template from gallery (tensors).
        rotation_shift (int): Rotations allowed in matching, in columns. Defaults to 15.
        normalise (bool): Flag to normalize HD. Defaults to False.
        norm_mean (float): Peak of the non-match distribution. Defaults to 0.45.
        norm_nb_bits (float): Average number of bits visible in 2 randomly sampled iris codes. Defaults to 12288.

    Returns:
        Tuple[float, int]: Minimum Hamming distance and corresponding rotation shift.
    """
    # Ensure tensors are on the correct device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move iris_codes and mask_codes to the GPU if not already
    probe_codes = [torch.from_numpy(code).to(device) for code in template_probe.iris_codes]
    gallery_codes = [torch.from_numpy(code).to(device) for code in template_gallery.iris_codes]
    probe_mask_codes = [torch.from_numpy(mask).to(device) for mask in template_probe.mask_codes]
    gallery_mask_codes = [torch.from_numpy(mask).to(device) for mask in template_gallery.mask_codes]

    best_dist = 1
    rot_shift = 0

    for current_shift in range(-rotation_shift, rotation_shift + 1):
        irisbits = [
            torch.roll(probe_code, current_shift, dims=1) != gallery_code
            for probe_code, gallery_code in zip(probe_codes, gallery_codes)
        ]
        maskbits = [
            torch.roll(probe_code, current_shift, dims=1) & gallery_code
            for probe_code, gallery_code in zip(probe_mask_codes, gallery_mask_codes)
        ]

        irisbitcount = sum(torch.sum(irisbit & maskbit) for irisbit, maskbit in zip(irisbits, maskbits))
        maskbitcount = sum(torch.sum(maskbit) for maskbit in maskbits)

        if maskbitcount == 0:
            continue

        current_dist = irisbitcount.item() / maskbitcount.item()

        if normalise:
            current_dist = max(0,
                               norm_mean - (norm_mean - current_dist) * torch.sqrt(maskbitcount.float() / norm_nb_bits))

        if (current_dist < best_dist) or (current_dist == best_dist and current_shift == 0):
            best_dist = current_dist
            rot_shift = current_shift

    return best_dist, rot_shift

def compute_distance_matrix(
        probe_templates: list,  # List of IrisTemplate objects from the probe set
        gallery_templates: list,  # List of IrisTemplate objects from the gallery set
        rotation_shift: int = 15,
        normalise: bool = False,
        norm_mean: float = 0.45,
        norm_nb_bits: float = 12288
) -> torch.Tensor:
    """Compute a distance matrix for all pairs of probe and gallery iris templates.

    Args:
        probe_templates (list): List of IrisTemplate objects from the probe set.
        gallery_templates (list): List of IrisTemplate objects from the gallery set.
        rotation_shift (int): Maximum rotation shift to allow during matching.
        normalise (bool): Whether to normalize Hamming distance.
        norm_mean (float): Peak of the non-match distribution.
        norm_nb_bits (float): Average number of bits visible in randomly sampled iris codes.

    Returns:
        torch.Tensor: Distance matrix where each entry (i, j) is the Hamming distance between
                      the i-th probe template and the j-th gallery template.
    """
    # Initialize the distance matrix with zeros
    distance_matrix = torch.zeros(len(probe_templates), len(gallery_templates),
                                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Total number of distance calculations (probe_templates * gallery_templates)
    total_comparisons = len(probe_templates) * len(gallery_templates)

    # Initialize the progress bar
    with tqdm(total=total_comparisons, desc="Computing Distance Matrix", unit="comparison") as pbar:
        # Compute Hamming distance for all pairs of probe and gallery templates
        for i, probe_template in enumerate(probe_templates):
            for j, gallery_template in enumerate(gallery_templates):
                # Call the simple_hamming_distance function to compute the distance for this pair
                best_dist, _ = simple_hamming_distance(
                    probe_template,
                    gallery_template,
                )
                # Store the computed distance in the matrix
                distance_matrix[i, j] = best_dist

                # Update progress bar after each comparison
                pbar.update(1)

    return distance_matrix


def get_label_template(json_file, label, eye_side):
    with open(json_file, 'r') as f:
        data = json.load(f)
    full_label = f"{label}_{eye_side}"  # Combine label and eye side for clarity
    return IrisTemplate.deserialize(data), full_label


def get_dataset(dataset_dir):
    # Traverse the dataset and return templates and labels
    templates = []
    labels = []
    label_to_n_samples = {}
    samples_for_eye_side = 5

    for label_dir in os.listdir(dataset_dir):  # First level: labels
        label_path = os.path.join(dataset_dir, label_dir)
        if not os.path.isdir(label_path):  # Skip non-directory files
            continue

        for eye_side_dir in os.listdir(label_path):  # Second level: eye side (e.g., left, right)
            eye_side_path = os.path.join(label_path, eye_side_dir)
            if not os.path.isdir(eye_side_path):  # Skip non-directory files
                continue

            for file in os.listdir(eye_side_path):  # Third level: JSON files
                if file.endswith(".json"):  # Only process JSON files
                    current_label_samples_counter = label_to_n_samples.get(f'{label_dir}_{eye_side_dir}', 0)
                    if current_label_samples_counter == samples_for_eye_side:
                        break
                    json_path = os.path.join(eye_side_path, file)
                    template, full_label = get_label_template(json_path, label_dir, eye_side_dir)
                    templates.append(template)
                    labels.append(full_label)
                    label_to_n_samples[f'{label_dir}_{eye_side_dir}'] = current_label_samples_counter + 1

    return templates, labels


# Usage Example
if __name__ == "__main__":

    # Assuming templates is a pre-loaded list of iris templates
    # Check for valid data
    dataset_dir = "../Casia/CASIA-Iris-Thousand-Encoding"
    templates, labels = get_dataset(dataset_dir)
    num_templates = len(templates)
    if num_templates == 0:
        raise ValueError("No valid templates found. Check dataset structure.")

    print(f"Loaded {num_templates} templates.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Compute distance matrix in parallel
    distance_matrix = compute_distance_matrix(templates, templates)

    # Save results
    np.savetxt("distance_matrix.csv", distance_matrix, delimiter=",")
    print("Distance matrix saved as 'distance_matrix.csv'.")

    # Save results
    np.savetxt("distance_matrix.csv", distance_matrix, delimiter=",")
    with open("labels.txt", "w") as f:
        f.write("\n".join(labels))

    print("Distance matrix saved as 'distance_matrix.csv'.")
    print("Labels saved as 'labels.txt'.")
