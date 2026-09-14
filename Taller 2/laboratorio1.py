import cv2
import numpy as np

# 1. Crear un píxel BGR de prueba completamente amarillo intenso
pixel = np.array([0, 255, 255], dtype=np.float32)

# 2. Calcular su valor en escala de grises usando la fórmula ponderada con NumPy
# Pesos ajustados al orden BGR: B=0.114, G=0.587, R=0.299
pesos_bgr = np.array([0.114, 0.587, 0.299])
valor_gris = np.sum(pixel * pesos_bgr)

# 3. Imprimir el resultado e interpretación
print("=== RESPUESTAS TALLER 1 ===")
print(f"Píxel BGR original: {pixel.astype(int)}")
print(f"Valor matemático exacto: {valor_gris:.2f}")
print(f"Intensidad de gris (0 a 255): {int(round(valor_gris))}")
print("Explicación: Arroja un gris muy claro (226) porque el amarillo combina verde y rojo, los canales a los que el ojo humano es más sensible.\n")

# 4. Usar la función de OpenCV sobre una imagen real
nombre_imagen = 'muestra.jpg'
imagen_color = cv2.imread(nombre_imagen)

if imagen_color is not None:
    img_gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('resultado_opencv_gris.jpg', img_gris)
    print("[OK] Imagen real convertida y guardada como 'resultado_opencv_gris.jpg'.")
else:
    print(f"[!] Recuerda tener una imagen llamada '{nombre_imagen}' en la carpeta 'Taller 2'.")