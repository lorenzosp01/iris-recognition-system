import json

import cv2
import torch
from iris import IRISPipeline
from torchvision import transforms as tt

from src.lib.cnn_utils import load_model
from src.utils.irisExtractor import get_cropped_iris_image


def process_image(path, eye_side):
    eye_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if eye_image is None:
        print("Error: Could not load image.")
    else:
        return process_eye_image(eye_image=eye_image, eye_side=eye_side, device="cpu")


def load_and_prepare_model(model_path, device="cpu"):
    model = load_model(model_path, to_cpu=True)
    model.to(device)
    model.eval()
    return model


def transform_image_to_tensor(image):
    transform = tt.Compose([tt.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def process_eye_image(eye_image, eye_side, device="cpu"):
    iris_pipeline = IRISPipeline()
    pipeline_output = iris_pipeline(img_data=eye_image, eye_side=eye_side)

    if pipeline_output['error'] is not None:
        return {"error": "An error occurred during processing."}

    try:
        iris_center = pipeline_output["metadata"]["eye_centers"]["iris_center"]

        cropped_iris_image = get_cropped_iris_image(eye_image, iris_center)

        normalized_iris_image = iris_pipeline.call_trace['normalization'].normalized_image

        full_eye_model = load_and_prepare_model("models/modelFullEyeMargin0.4Loss0.00457.pth", device)
        normalized_iris_model = load_and_prepare_model("models/modelNormalizedEyeMargin0.4Loss0.00521.pth", device)

        cropped_iris_tensor = transform_image_to_tensor(cropped_iris_image).to(device)
        normalized_iris_tensor = transform_image_to_tensor(normalized_iris_image).to(device)

        with torch.no_grad():
            full_eye_prediction = full_eye_model(cropped_iris_tensor)
            normalized_iris_prediction = normalized_iris_model(normalized_iris_tensor)

        full_eye_prediction = full_eye_prediction.cpu().numpy().tolist()
        normalized_iris_prediction = normalized_iris_prediction.cpu().numpy().tolist()

        iris_template = pipeline_output["iris_template"].serialize()

        output_dict = {"full_eye_prediction": json.dumps(full_eye_prediction),
                       "normalized_iris_prediction": json.dumps(normalized_iris_prediction),
                       "iris_template_output": json.dumps(iris_template)
                       }

        return output_dict

    except Exception as e:
        return {"error": f"An error occurred during processing: {str(e)}"}