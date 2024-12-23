import cv2
import numpy as np

def resize_with_center(image, center, scale, output_size):
    """
    Ridimensiona un'immagine mantenendo un determinato punto come centro.

    Args:
        image (np.ndarray): Immagine originale.
        center (tuple): Coordinata del punto centrale (x, y).
        scale (float): Fattore di ridimensionamento.
        output_size (tuple): Dimensioni finali dell'immagine (width, height).

    Returns:
        np.ndarray: Immagine trasformata.
    """
    # Ottieni le dimensioni originali
    h, w = image.shape[:2]
    cx, cy = center

    # Matrice di traslazione per spostare il centro desiderato al centro dell'immagine
    translate_matrix = np.array([[1, 0, -(cx - w // 2)],
                                 [0, 1, -(cy - h // 2)]], dtype=np.float32)

    # Applica la traslazione
    translated_image = cv2.warpAffine(image, translate_matrix, (w, h))

    # Matrice di scalatura
    scale_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), 0, scale)

    # Applica la scalatura
    scaled_image = cv2.warpAffine(translated_image, scale_matrix, (w, h))

    # Ritaglia o riempie l'immagine per adattarla alle dimensioni di output
    output_width, output_height = output_size
    x_offset = max(0, (w - output_width) // 2)
    y_offset = max(0, (h - output_height) // 2)

    # Ritaglio
    cropped_image = scaled_image[y_offset:y_offset + output_height, x_offset:x_offset + output_width]

    return cropped_image