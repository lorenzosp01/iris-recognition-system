import os

import cv2
import numpy as np

def find_pupil(image):

    # Preprocessing: Smooth the image to reduce noise
    blurred = cv2.GaussianBlur(image, (9, 9), 2)

    # Use Hough Circle Transform to detect circles (approximates iris)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=50, param2=30, minRadius=20, maxRadius=60)

    # Check if any circle is detected
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        # Find the center of the image
        img_center = (image.shape[1] // 2, image.shape[0] // 2)

        # Calculate distances of each circle from the image center
        distances = [np.sqrt((x - img_center[0]) ** 2 + (y - img_center[1]) ** 2) for x, y, r in circles]

        # Get the circle with the smallest distance
        nearest_circle_idx = np.argmin(distances)
        return circles[nearest_circle_idx]
    else:
        print("No circles detected.")

def extract_iris(image, pupil_center, pupil_radius, iris_radius_multiplier=2.5):
    x, y = pupil_center
    iris_radius = int(pupil_radius * iris_radius_multiplier)  # Calculate the iris radius

    # Create a blank mask
    mask = np.zeros_like(image, dtype=np.uint8)

    # Draw the iris circle on the mask
    cv2.circle(mask, (x, y), iris_radius, 255, -1)

    # Extract the iris region
    iris_region = cv2.bitwise_and(image, image, mask=mask)

    return iris_region, mask

if __name__ == '__main__':
    folder = "../data/datasets/CASIA-Iris-Thousand/CASIA-Iris-Thousand/000/L/"
    for path in os.listdir(folder):
        image = cv2.imread(os.path.join(folder, path), cv2.IMREAD_GRAYSCALE)
        x, y, r = find_pupil(image)
        iris_region, mask = extract_iris(image, (x, y), r)

        cv2.imshow(path, iris_region)
    cv2.waitKey(0)

