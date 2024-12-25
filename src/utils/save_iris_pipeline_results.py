import json
import os
import cv2
import pandas as pd
from iris import IRISPipeline
from tqdm import tqdm
from src.data.CasiaIrisDataset import CasiaIrisDataset
from src.utils.irisExtractor import get_cropped_iris_image

# Leggere le immagini, trovare l'occhio e salvare immagine normalizzata e template in due cartelle separate seguendo la struttura originale del dataset
# Il dataset originale poi deve essere modificato in modo da utilizzare solamente le imm

originalPath = "F:\\Dataset\\Casia"

casia_dataset = CasiaIrisDataset(originalPath)
iris_pipeline = IRISPipeline()


def write_template(iris_template, file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, "w") as template_file:
        json.dump(iris_template.serialize(), template_file)

def write_img(file_path, img):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    cv2.imwrite(file_path, img)


labels_path = []
eyes_found_counter = 0
for anchor, label, path in tqdm(casia_dataset, desc="Processing items", total=len(casia_dataset)):
    output = iris_pipeline(img_data=anchor, eye_side="left" if label < 1000 else "right")
    if output['error'] is None:
        eyes_found_counter += 1
        label_file = str(label % 1000 if label > 999 else label).zfill(3)
        eye_side = "L" if label < 1000 else "R"
        fileName = path.split("\\")[-1].split('.')[0]
        file_path = f"{str(label_file).zfill(3)}\\{eye_side}\\{fileName}"
        labels_path.append((f"{label_file}-{eye_side}", file_path))

        # Save the cropped image
        write_img(originalPath + "\\CASIA-Iris-Thousand-Centered\\" + file_path + ".jpg", get_cropped_iris_image(anchor, output["metadata"]["eye_centers"]["iris_center"]))
        # Save the normalized image
        write_img(originalPath + "\\CASIA-Iris-Thousand-Normalized\\" + file_path + ".jpg", iris_pipeline.call_trace['normalization'].normalized_image)
        # save template to file
        write_template(output["iris_template"], originalPath + "\\CASIA-Iris-Thousand-Encoding\\" + file_path + ".json")


# Save the updated CSV file
updated_file = os.path.join(originalPath + "iris_thousand_updated.csv")
df = pd.DataFrame(labels_path, columns=["Label", "ImagePath"])
df.to_csv(updated_file)

print(f"Found {eyes_found_counter} eyes")