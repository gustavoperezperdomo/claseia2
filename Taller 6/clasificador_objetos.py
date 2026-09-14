import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar imagen original
nombre_imagen = 'objetos.jpg'
imagen_color = cv2.imread(nombre_imagen)

if imagen_color is None:
    print(f"Error: No se pudo cargar '{nombre_imagen}'. Verifica que esté en la carpeta 'Taller 6'.")
    exit()

# Copia para dibujar los resultados
imagen_resultado = imagen_color.copy()

# 2. Pipeline de Procesamiento Digital de Imágenes
# a. Conversión a escala de grises
grises = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)

# b. Umbralización (Binarización Otsu para separar objeto de fondo)
# Invertimos con THRESH_BINARY_INV para que los objetos sean blancos (255) y el fondo negro (0)
_, imagen_binaria = cv2.threshold(grises, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# c. Limpieza Morfológica (Elimina ruido pequeño dentro y fuera del objeto)
kernel = np.ones((5, 5), np.uint8)
limpia = cv2.morphologyEx(imagen_binaria, cv2.MORPH_OPEN, kernel)

# d. Detección de Contornos
contornos, _ = cv2.findContours(limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"--- TOTAL DE OBJETOS DETECTADOS: {len(contornos)} ---")

# Umbral definido para clasificar objeto Grande o Pequeño (puedes ajustarlo según tu imagen)
UMBRAL_AREA_GRANDE = 1500 

# 3. Iterar y clasificar los objetos
for i, cnt in enumerate(contornos):
    area = cv2.contourArea(cnt)
    
    # Filtrar ruido extremadamente pequeño
    if area < 100:
        continue

    # Imprimir área en consola
    print(f"Objeto #{i+1}: Área = {area:.2f} px")

    # Obtener Bounding Box (x, y, ancho, alto)
    x, y, w, h = cv2.boundingRect(cnt)

    # Lógica de clasificación por tamaño
    if area > UMBRAL_AREA_GRANDE:
        color_box = (255, 0, 0)  # BGR -> Azul (Objeto Grande)
        etiqueta = "Grande"
    else:
        color_box = (0, 0, 255)  # BGR -> Rojo (Objeto Pequeño)
        etiqueta = "Pequeno"

    # Dibujar Bounding Box en la imagen
    cv2.rectangle(imagen_resultado, (x, y), (x + w, y + h), color_box, 2)

    # Dibujar Centroide
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(imagen_resultado, (cx, cy), 4, (0, 255, 0), -1)  # Punto Verde

    # Escribir etiqueta y área sobre la imagen
    text = f"{etiqueta} ({int(area)}px)"
    cv2.putText(imagen_resultado, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

# 4. Guardar y visualizar resultados
# Convertir BGR a RGB para mostrar correctamente en Matplotlib
imagen_resultado_rgb = cv2.cvtColor(imagen_resultado, cv2.COLOR_BGR2RGB)
limpia_rgb = cv2.cvtColor(limpia, cv2.COLOR_GRAY2RGB)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(limpia_rgb)
plt.title('Imagen Binarizada y Limpia')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(imagen_resultado_rgb)
plt.title('Clasificación por Bounding Box\n(Azul = Grande, Rojo = Pequeño)')
plt.axis('off')

plt.tight_layout()
plt.savefig('resultado_clasificacion.png')
print("\nImagen guardada exitosamente como 'resultado_clasificacion.png'")