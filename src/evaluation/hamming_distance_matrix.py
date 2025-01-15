import json
import multiprocessing

import numpy as np
from iris import IrisTemplate
from iris.nodes.matcher.utils import simple_hamming_distance
from tqdm import tqdm

from src.data.CasiaIrisDataset import CasiaIrisDataset
from src.data.datasetUtils import splitDataset

distance_matrix = np.zeros((9984, 9984))

def get_label_template(json_file, label, eye_side):
    with open(json_file, 'r') as f:
        data = json.load
    full_label = f"{label}_{eye_side}"
    return IrisTemplate.deserialize(data), full_label


def task(start_idx, end_idx, templates):
    n_rows = end_idx - start_idx
    sub_distance_matrix = np.zeros((n_rows, len(templates)))
    try:
        for i in range(start_idx, end_idx):
            for j in range(len(templates)):
                dist = simple_hamming_distance(templates[i], templates[j])[0]
                sub_distance_matrix[i % n_rows, j] = dist
    except Exception as e:
        print(f"Error in task: {e}")
    return sub_distance_matrix


def calculate_distance_matrix(templates, num_cores=10):
    num_samples = len(templates)
    chunk_size = num_samples // num_cores
    total_tasks = num_samples * num_samples

    results = []
    with tqdm(total=total_tasks) as pbar:
        with multiprocessing.Pool(processes=num_cores) as pool:
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


if __name__ == "__main__":

    dataset_dir = "../Casia"
    dataset = CasiaIrisDataset(dataset_dir, transform=[], encoding=True)

    _, _, test_dataset = splitDataset(dataset, 0.2, 0.1)
    dataset.eval()

    templates, labels = elaborate_batch(test_dataset)
    num_templates = len(templates)
    if num_templates == 0:
        raise ValueError("No valid templates found. Check dataset structure.")

    distance_matrix = calculate_distance_matrix(templates)
    np.savetxt("distance_matrix_3724.csv", distance_matrix, delimiter=",")


    with open("labels.txt", "w") as f:
        for label in labels:
            f.write(f"{label}\n")
