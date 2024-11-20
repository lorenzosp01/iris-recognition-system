import os
import cv2
import numpy as np


def find_pupil(image):
    # Preprocessing: Smooth the image to reduce noise
    median_blurred = cv2.medianBlur(image, 5)  # blur image to remove noise
    gaussian_blurred = cv2.GaussianBlur(median_blurred, (9, 9), 2)

    # Use Hough Circle Transform to detect circles (approximates iris)
    circles = cv2.HoughCircles(gaussian_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
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


def extract_iris(image, colorImage, pupil_center, radius, kernel, distance=2, threshold=10):
    x_center, y_center = pupil_center
    h, w = image.shape
    kernel = kernel if kernel % 2 == 1 else kernel + 1

    def apply_local_filter(x, y):
        half_k = kernel // 2
        x_start, x_end = max(0, x - half_k), min(w, x + half_k + 1)
        y_start, y_end = max(0, y - half_k), min(h, y + half_k + 1)
        window = image[y_start:y_end, x_start:x_end]
        return np.mean(window)

    def find_edge_along_line(start, direction, limit):
        """
        Trova il bordo lungo una direzione data.
        """
        x, y = start
        dx, dy = direction
        x_start, y_start = x, y

        while 0 <= x < w and 0 <= y < h and x - x_center < limit and y - y_center < limit:

            current_value = apply_local_filter(x, y)
            center_value = apply_local_filter(x_start, y_start)
            diff = abs(current_value - center_value)
            drawImage = colorImage.copy()
            cv2.circle(drawImage, (x, y), 1, (0, 0, 255), 2)
            cv2.circle(drawImage, (x_start, y_start), 1, (0, 255, 0), 2)
            cv2.imshow(f"Difference:", drawImage)
            cv2.waitKey(250)

            # Confronta la differenza di intensità
            if diff > threshold:
                print(f"Diff: {diff}")
                if (direction[0] == 0):
                    return int(abs(y - y_center + kernel))
                else:
                    return int(abs(x - x_center + kernel))
            x += dx
            y += dy
            if (abs(x-x_start) > kernel*distance or abs(y-y_start) > kernel*distance):
                x_start = x_start + (kernel*dx)
                y_start = y_start + (kernel*dy)
        return int(limit)

    # Trova i bordi nelle quattro direzioni principali
    top = find_edge_along_line((x_center, y_center - radius - kernel), (0, -1), 2.5 * radius)
    bottom = find_edge_along_line((x_center, y_center + radius + kernel), (0, 1), 2.5 * radius)
    left = find_edge_along_line((x_center - radius - kernel, y_center), (-1, 0), 2.5 * radius)
    right = find_edge_along_line((x_center + radius + kernel, y_center), (1, 0), 2.5 * radius)

    return max(top, bottom, left, right)+(kernel//2)


if __name__ == '__main__':
    folder = "../data/datasets/CASIA-Iris-Thousand/CASIA-Iris-Thousand/000/L/"
    for path in os.listdir(folder):
        image = cv2.imread(os.path.join(folder, path), 1)
        x, y, r = find_pupil(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        cv2.circle(image, (x, y), r, (255, 255, 255), 2)
        cv2.circle(image, (x, y), 1, (255, 255, 255), 2)

        iris_region = extract_iris(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), image, (x, y), r,8, 1, 8)

        cv2.circle(image, (x, y), iris_region, (255, 255, 255), 1)

        cv2.imshow(f"Iris Region{path}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()