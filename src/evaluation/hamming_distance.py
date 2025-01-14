import json
import multiprocessing

import numpy as np
from iris import IrisTemplate
from iris.nodes.matcher.utils import simple_hamming_distance
from tqdm import tqdm

from src.data.CasiaIrisDataset import CasiaIrisDataset
from src.data.datasetUtils import splitDataset
# Define dataset root directory

# Function to load iris templates and their labels
def get_label_template(json_file, label, eye_side):
    with open(json_file, 'r') as f:
        data = json.load
    full_label = f"{label}_{eye_side}"  # Combine label and eye side for clarity
    return IrisTemplate.deserialize(data), full_label

# def get_dataset(dataset_dir):
#     # Traverse the dataset and return templates and labels
#     templates = []
#     labels = []
#     label_to_n_samples = {}
#     samples_for_eye_side = 5
#
#     for i, label_dir in enumerate(os.listdir(dataset_dir)):  # First level: labels
#         label_path = os.path.join(dataset_dir, label_dir)
#         if not os.path.isdir(label_path):  # Skip non-directory files
#             continue
#
#         if random.randint(0, 1) == 0:
#             eye_side = 'L'
#         else:
#             eye_side = 'R'
#
#         eye_side_path = f"{label_path}/{eye_side}"
#         for file in os.listdir(eye_side_path):  # Third level: JSON files
#             if file.endswith(".json"):  # Only process JSON files
#                 current_label_samples_counter = label_to_n_samples.get(f'{label_dir}_{eye_side}', 0)
#                 if current_label_samples_counter == samples_for_eye_side:
#                     break
#                 json_path = os.path.join(eye_side_path, file)
#                 template, full_label = get_label_template(json_path, label_dir, eye_side)
#                 templates.append(template)
#                 labels.append(full_label)
#                 label_to_n_samples[f'{label_dir}_{eye_side}'] = current_label_samples_counter + 1
#
#     return templates, labels

def task(start_idx, end_idx, templates):
    n_rows = end_idx - start_idx
    sub_distance_matrix = np.zeros((n_rows, len(templates)))
    try:
        for i in range(start_idx, end_idx):
            for j in range(len(templates)):  # Confronto con tutti i template della gallery
                dist = simple_hamming_distance(templates[i], templates[j])[0]
                sub_distance_matrix[i % n_rows, j] = dist
    except Exception as e:
        print(f"Error in task: {e}")
    return sub_distance_matrix

def calculate_distance_matrix(templates, num_cores=10):
    num_samples = len(templates)
 # Matrice delle distanze

    # Suddividere il lavoro in blocchi per ogni worker
    chunk_size = num_samples // num_cores

    # Progress bar
    total_tasks = num_samples * num_samples
    # Creazione di un pool di processi
    results = []
    with tqdm(total=total_tasks) as pbar:
        with multiprocessing.Pool(processes=num_cores) as pool:
            # Assegnare un blocco di template a ciascun worker
            for core_idx in range(num_cores):
                start_idx = core_idx * chunk_size
                end_idx = (core_idx + 1) * chunk_size if core_idx < num_cores - 1 else num_samples
                result = pool.apply_async(task, args=(start_idx, end_idx, templates))
                results.append(result)

            for result in results:
                print(result.get())

    return np.concatenate([result.get() for result in results], axis=0)

def elaborate_batch(dataset):
    iris_templates = []
    labels = []
    for image, label, path in dataset:
        iris_templates.append(image)
        labels.append(label)
    return iris_templates, labels

# Usage Example
if __name__ == "__main__":

    # Assuming templates is a pre-loaded list of iris templates
    # Check for valid data
    dataset_dir = "F:\\Dataset\\Casia"
    dataset = CasiaIrisDataset(dataset_dir, transform=[], encoding=True)

    _, _, test_dataset = splitDataset(dataset, 0.2, 0.1)
    dataset.eval()

    # templates, labels = get_dataset(dataset_dir)
    templates, labels = elaborate_batch(test_dataset)
    num_templates = len(templates)
    if num_templates == 0:
        raise ValueError("No valid templates found. Check dataset structure.")

    # print(f"Loaded {num_templates} templates.")
    # matcher = HammingDistanceMatcher()
    # Compute distance matrix in parallel
    distance_matrix = calculate_distance_matrix(templates)

    print("Distance matrix computed.", distance_matrix)
    # Save results
    np.savetxt("distance_matrix_3724.csv", distance_matrix, delimiter=",")
    print("Distance matrix saved as 'distance_matrix.csv'.")

    with open("labels.txt", "w") as f:
        for label in labels:
            f.write(f"{label}\n")
