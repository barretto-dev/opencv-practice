import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img1 = cv.imread('img2.png', 0)
img1_bgr = cv.imread('img2.png', 1)

# Aumentar constraste da imagem
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
const = clahe.apply(img1)

# Aplicar borrão para diminuir ruidos
blur = cv.GaussianBlur(const, (7,7), 0)

# Encontrar bordas
edge = cv.Canny(blur,70,200)

# O canny geralmente quebra as rachaduras
# para corrigir isso se usa o MORPH_CLOSE
kernel = np.ones((3,3), np.uint8)
closed = cv.morphologyEx(edge, cv.MORPH_CLOSE, kernel)

copy_closed = closed.copy()

#Encontrar contornos
contours, _ = cv.findContours(
    copy_closed,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_NONE
)

for i, contour in enumerate(contours):
    area = cv.contourArea(contour)
    if area <= 120:
        continue

    cv.drawContours(img1_bgr, [contour], -1, (0,255,0), 2)

img_rgb = cv.cvtColor(img1_bgr, cv.COLOR_BGR2RGB)

plt.subplot(241); plt.imshow(img1, cmap="gray");  plt.title("Original");
plt.subplot(242); plt.imshow(const, cmap="gray");  plt.title("Clahe");
plt.subplot(243); plt.imshow(blur, cmap="gray");  plt.title("Blur");
plt.subplot(244); plt.imshow(edge, cmap="gray");  plt.title("Canny");
plt.subplot(245); plt.imshow(closed, cmap="gray");  plt.title("CLOSED");
plt.subplot(246); plt.imshow(img_rgb);  plt.title("Contours");
plt.show()