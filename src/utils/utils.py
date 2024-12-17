import cv2

def drawEye(pupil_center, iris_center, img):
    cv2.circle(img, (int(pupil_center[0]), int(pupil_center[1])), 1, (0, 0, 255), 2)
    cv2.circle(img, (int(iris_center[0]), int(iris_center[1])), 1, (0, 0, 255), 2)
    return img