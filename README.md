# ContourImage
Programme permettant de trouver les contours des objets présents sur un image de manière automatique.

import numpy as np
import cv2
from matplotlib import pyplot as plt

def sobelOperator(img):
    container = np.copy(img)
    size = container.shape
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            gx = (img[i - 1][j - 1] + 2*img[i][j - 1] + img[i + 1][j - 1]) - (img[i - 1][j + 1] + 2*img[i][j + 1] + img[i + 1][j + 1])
            gy = (img[i - 1][j - 1] + 2*img[i - 1][j] + img[i - 1][j + 1]) - (img[i + 1][j - 1] + 2*img[i + 1][j] + img[i + 1][j + 1])
            container[i][j] = min(255, np.sqrt(gx**2 + gy**2))
    return container
    pass

img = cv2.cvtColor(cv2.imread("zidane.jpg"), cv2.COLOR_BGR2GRAY)
img = sobelOperator(img)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
plt.imshow(img)
plt.show()

plt.hist(img.ravel(), bins=256, range=[0, 256])
plt.title("Histogramme des intensités de bord")
plt.xlabel("Intensité de bord")  
plt.ylabel("Nombre de Pixels")  
plt.show()

# Matrices Sobel

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Masque pour gx
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Masque pour gy
plt.figure(figsize=(12, 4)) #les afficher

#Les matrices de convolution
plt.subplot(1, 3, 1)
plt.imshow(sobel_x, cmap='gray')
plt.title("Matrice de convolution Sobel X")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(sobel_y, cmap='gray')
plt.title("Matrice de convolution Sobel Y")
plt.colorbar()


img = cv2.imread('zidane.jpg', cv2.IMREAD_GRAYSCALE) # lire l'image donnée en niveaux de gris

# Matrice de corrélation
correlation_matrix = np.corrcoef(img.ravel(), img.ravel())

plt.subplot(1, 3, 3)
plt.imshow(correlation_matrix, cmap='gray')
plt.title("Matrice de corrélation")
plt.colorbar()

plt.show()

# Sobel caméra

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
   

def processWebcam():
    cap = cv2.VideoCapture(0)  # Ouvrir la webcam (0 = premiere camera disponible)

    while True:
        ret, frame = cap.read()  # Capturer un frame de la video (c'est la d'ou vient l'effet 'segmenter')
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir en niveaux de gris
        edges = sobelOperator(gray)  # Appliquer le filtre de Sobel
        cv2.imshow('Sobel Edge Detection', edges)  # Afficher le resultat

        if cv2.waitKey(1) & 0xFF == ord('q'):  # pour arreter appuyer sur 'q'
            break

    cap.release()
    cv2.destroyAllWindows()

# Lancer le traitement en irl

processWebcam()

#Sobel Segmentation
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
