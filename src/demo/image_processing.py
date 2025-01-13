import base64  # For encoding binary data (e.g., iris template)
import json  # Import the json module for serialization

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
    """Load a model and move it to the specified device."""
    model = load_model(model_path)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model


def transform_image_to_tensor(image):
    """Transform the image to a tensor and add a batch dimension."""
    transform = tt.Compose([tt.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def process_eye_image(eye_image, eye_side, device="cpu"):
    """Process the eye image using the IRISPipeline and CNN models."""
    iris_pipeline = IRISPipeline()
    pipeline_output = iris_pipeline(img_data=eye_image, eye_side=eye_side)

    if pipeline_output['error'] is not None:
        return {"error": pipeline_output['error']}  # Return error as a dictionary

    try:
        # Extract the iris center from the pipeline metadata
        iris_center = pipeline_output["metadata"]["eye_centers"]["iris_center"]

        # Crop the iris region from the original eye image
        cropped_iris_image = get_cropped_iris_image(eye_image, iris_center)

        # Get the normalized iris image from the pipeline
        normalized_iris_image = iris_pipeline.call_trace['normalization'].normalized_image

        # Load the CNN models
        full_eye_model = load_and_prepare_model("models/modelFullEyeMargin0.2.pth", device)
        normalized_iris_model = load_and_prepare_model("models/modelNormalizedEyeMargin0.4Loss0.00521.pth", device)

        # Transform images to tensors
        cropped_iris_tensor = transform_image_to_tensor(cropped_iris_image).to(device)
        normalized_iris_tensor = transform_image_to_tensor(normalized_iris_image).to(device)

        # Get predictions from the models
        with torch.no_grad():  # Disable gradient calculation for inference
            full_eye_prediction = full_eye_model(cropped_iris_tensor)
            normalized_iris_prediction = normalized_iris_model(normalized_iris_tensor)

        # Convert predictions to lists for JSON serialization
        full_eye_prediction = full_eye_prediction.cpu().numpy().tolist()
        normalized_iris_prediction = normalized_iris_prediction.cpu().numpy().tolist()

        # Serialize the iris template (if it's binary data, encode it as base64)
        iris_template = pipeline_output["iris_template"].serialize()

        # Prepare the output dictionary with separate keys
        output_dict = {"full_eye_prediction": json.dumps(full_eye_prediction),  # Serialize prediction
                       "normalized_iris_prediction": json.dumps(normalized_iris_prediction),  # Serialize prediction
                       "iris_template_output": json.dumps(iris_template)  # Already serialized or encoded
                       }

        return output_dict

    except Exception as e:
        return {"error": f"An error occurred during processing: {str(e)}"}


if __name__ == '__main__':
    # Load the eye image
    eye_image = cv2.imread("eye.jpg", cv2.IMREAD_GRAYSCALE)

    if eye_image is None:
        print("Error: Could not load image.")
    else:
        # Process the eye image
        output = process_eye_image(eye_image=eye_image, eye_side="right", device="cpu")

        # Print or store each output separately
        if "error" in output:
            print("Error:", output["error"])
        else:
            # print("Full Eye Prediction (JSON):", output["full_eye_prediction"])
            # print("Normalized Iris Prediction (JSON):", output["normalized_iris_prediction"])
            # print("Iris Template Output:", output["iris_template_output"])

            print(type(output["full_eye_prediction"]))
            print(type(output["normalized_iris_prediction"]))
            print(type(output["iris_template_output"]))
