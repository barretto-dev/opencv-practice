import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("img1.png")

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

_, img_mask = cv.threshold(img_gray, 25, 255, cv.THRESH_BINARY_INV)

kernel = np.ones((3,3), np.uint8)
closed = cv.morphologyEx(img_mask, cv.MORPH_CLOSE, kernel)

contours, _ = cv.findContours(
    closed,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_NONE
)

# 8. Filtra contornos pequenos
result = img_rgb.copy()
for cnt in contours:
    area = cv.contourArea(cnt)

    if area < 110:
        continue

    perimeter = cv.arcLength(cnt, False)

    # Desenha contorno
    cv.drawContours(result, [cnt], -1, (0, 255, 0), 2)

    print("Área:", area)


#plt.subplot(241); plt.imshow(img_rgb);  plt.title("Original");
#plt.subplot(242); plt.imshow(img_gray, cmap="gray");  plt.title("Gray");
#plt.subplot(243); plt.imshow(img_mask, cmap="gray");  plt.title("Mask");
#plt.subplot(244); plt.imshow(closed, cmap="gray");  plt.title("MORPH_CLOSE");
plt.subplot(111); plt.imshow(result);  plt.title("Final");
plt.show()

