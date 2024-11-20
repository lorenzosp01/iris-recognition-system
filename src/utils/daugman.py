# Perform segmentation
import os
import cv2
from src.utils.fnc.segment import segment


def remove_glare(img):
    # Threshold to detect bright spots (glare regions)
    _, mask = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    # Load pretrained model
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    inpainted = cv2.inpaint(img, refined_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return inpainted


if __name__ == '__main__':
    dataset_path = "../data/datasets/CASIA-Iris-Thousand/CASIA-Iris-Thousand/001/L/"

    for image_path in os.listdir(dataset_path):
        # Read the image
        im = cv2.imread(os.path.join(dataset_path, image_path), cv2.IMREAD_GRAYSCALE)
        # Remove glare from the image
        im = remove_glare(im)
        # Perform segmentation
        ciriris, cirpupil, imwithnoise = segment(im)

        # Draw the pupil and iris boundaries
        cv2.circle(im, (ciriris[1], ciriris[0]), ciriris[2], (255, 255, 255), 2)

        cv2.imshow(f"Segmented Iris-{image_path}", im)
    cv2.waitKey(0)
