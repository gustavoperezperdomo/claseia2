import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar la imagen en escala de grises
nombre_imagen = 'imagen_prueba.jpg'
imagen = cv2.imread(nombre_imagen, cv2.IMREAD_GRAYSCALE)

if imagen is None:
    print(f"Error: No se pudo cargar la imagen '{nombre_imagen}'. Revisa el nombre y la ruta.")
    exit()

# 2. Detección de Bordes con Sobel
# Sobel X (Bordes Verticales)
sobel_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
sobel_x_abs = cv2.convertScaleAbs(sobel_x)

# Sobel Y (Bordes Horizontales)
sobel_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
sobel_y_abs = cv2.convertScaleAbs(sobel_y)

# 3. Detección de Bordes con Canny (Umbrales estándar iniciales: 50 y 150)
bordes_canny_estandar = cv2.Canny(imagen, 50, 150)

# 4. Experimentación con Umbrales de Canny
bordes_canny_bajos = cv2.Canny(imagen, 10, 50)     # Umbrales bajos (captura más ruido)
bordes_canny_altos = cv2.Canny(imagen, 200, 250)   # Umbrales altos (solo bordes muy fuertes)

# 5. Visualización del Panel Comparativo General (Puntos 2 y 3 del taller)
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(imagen, cmap='gray')
plt.title('Imagen Original (Grises)')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(sobel_x_abs, cmap='gray')
plt.title('Sobel X (Bordes Verticales)')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(sobel_y_abs, cmap='gray')
plt.title('Sobel Y (Bordes Horizontales)')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(bordes_canny_estandar, cmap='gray')
plt.title('Algoritmo Canny (50, 150)')
plt.axis('off')

plt.tight_layout()
plt.savefig('panel_comparativo.png')
print("Panel comparativo guardado como 'panel_comparativo.png'")

# 6. Visualización de la Experimentación de Canny (Punto 4 del taller)
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.imshow(bordes_canny_bajos, cmap='gray')
plt.title('Canny Bajos (10, 50)\n[Mucho Ruido]')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(bordes_canny_estandar, cmap='gray')
plt.title('Canny Estándar (50, 150)\n[Balanceado]')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(bordes_canny_altos, cmap='gray')
plt.title('Canny Altos (200, 250)\n[Solo Bordes Fuertes]')
plt.axis('off')

plt.tight_layout()
plt.savefig('experimentacion_canny.png')
print("Experimentación guardada como 'experimentacion_canny.png'")