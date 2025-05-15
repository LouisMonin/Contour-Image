# -*- coding: utf-8 -*-

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
