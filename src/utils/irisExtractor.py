import cv2
import numpy as np


def get_cropped_iris_image(image, center, scale=1, output_size=(256, 256)):
    h, w = image.shape[:2]
    cx, cy = center

    translate_matrix = np.array([[1, 0, -(cx - w // 2)],
                                 [0, 1, -(cy - h // 2)]], dtype=np.float32)

    translated_image = cv2.warpAffine(image, translate_matrix, (w, h))

    scale_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), 0, scale)

    scaled_image = cv2.warpAffine(translated_image, scale_matrix, (w, h))

    output_width, output_height = output_size
    x_offset = max(0, (w - output_width) // 2)
    y_offset = max(0, (h - output_height) // 2)

    cropped_image = scaled_image[y_offset:y_offset + output_height, x_offset:x_offset + output_width]

    return cropped_image

