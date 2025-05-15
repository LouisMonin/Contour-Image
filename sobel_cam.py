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

