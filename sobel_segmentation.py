import cv2
import numpy as np

def sobelOperator(img):
    container = np.copy(img)
    size = container.shape
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            gx = (img[i - 1][j - 1] + 2*img[i][j - 1] + img[i + 1][j - 1]) - (img[i - 1][j + 1] + 2*img[i][j + 1] + img[i + 1][j + 1])
            gy = (img[i - 1][j - 1] + 2*img[i - 1][j] + img[i - 1][j + 1]) - (img[i + 1][j - 1] + 2*img[i + 1][j] + img[i + 1][j + 1])
            container[i][j] = min(255, np.sqrt(gx**2 + gy**2))
    return container

def segmentImage(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = sobelOperator(img)

    # Seuillage pour la segmentation (ici 100)
    _, segmented = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)

    cv2.imshow('Original Image', img)
    cv2.imshow('Sobel Edges', edges)
    cv2.imshow('Segmented Image', segmented)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


segmentImage('zidane.jpg')
