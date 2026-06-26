import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("img2.png")

# 1. Escala de cinza
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2. CLAHE - melhora contraste local
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_clahe = clahe.apply(gray)

# 3. Redução de ruído
blur = cv.medianBlur(gray_clahe, 7)

# 4. Canny
edges = cv.Canny(blur, 70, 200)

# 5. Fechamento morfológico
kernel = np.ones((3, 3), np.uint8)

closed = cv.morphologyEx(
    edges,
    cv.MORPH_CLOSE,
    kernel,
)

# 6. Dilatação para conectar partes da rachadura
dilated = cv.dilate(closed, kernel)

# 7. Encontra contornos
contours, _ = cv.findContours(
    dilated,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_NONE
)

# 8. Filtra contornos pequenos
result = img.copy()
result = cv.cvtColor(result, cv.COLOR_BGR2RGB)

for cnt in contours:
    area = cv.contourArea(cnt)

    if area < 110:
        continue

    perimeter = cv.arcLength(cnt, False)

    # Desenha contorno
    cv.drawContours(result, [cnt], -1, (0, 255, 0), 2)

    # Caixa delimitadora rotacionada
    # rect = cv.minAreaRect(cnt)
    # box = cv.boxPoints(rect)
    # box = np.intp(box)

    # cv.drawContours(result, [box], 0, (0, 255, 255), 2)

    print("Área:", area)
    print("Perímetro:", perimeter)
    #print("Ângulo:", rect[2])


plt.subplot(241); plt.imshow(img, cmap="gray");  plt.title("Original");
plt.subplot(242); plt.imshow(gray_clahe, cmap="gray");  plt.title("CLAHE");
plt.subplot(243); plt.imshow(blur, cmap="gray");  plt.title("Blur");
plt.subplot(244); plt.imshow(edges, cmap="gray");  plt.title("Canny");
plt.subplot(245); plt.imshow(closed, cmap="gray");  plt.title("Morph Close");
plt.subplot(246); plt.imshow(dilated, cmap="gray");  plt.title("dilated");
plt.subplot(247); plt.imshow(result);  plt.title("Resultado");
plt.show()