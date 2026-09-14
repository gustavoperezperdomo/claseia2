import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar la imagen en escala de grises
nombre_imagen = 'documento.jpg'
imagen_grises = cv2.imread(nombre_imagen, cv2.IMREAD_GRAYSCALE)

if imagen_grises is None:
    print(f"Error: No se encontró '{nombre_imagen}' en la carpeta 'Taller 3'.")
    exit()

# 2. Umbralización Estática con Ruido Intencional (T = 110)
# Convertimos a binario (píxeles sobre el umbral quedan blancos 255)
_, imagen_binaria = cv2.threshold(imagen_grises, 110, 255, cv2.THRESH_BINARY)

# 3. Construir Elemento Estructurante (Kernel 3x3 de unos)
kernel = np.ones((3, 3), np.uint8)

# 4. Operación de APERTURA (Opening = Erosión + Dilatación)
# Útil para eliminar ruido blanco aislado del fondo (ruido de sal)
apertura = cv2.morphologyEx(imagen_binaria, cv2.MORPH_OPEN, kernel)

# 5. Operación de CIERRE (Closing = Dilatación + Erosión)
# Útil para rellenar huecos negros dentro del objeto (ruido de pimienta)
cierre = cv2.morphologyEx(imagen_binaria, cv2.MORPH_CLOSE, kernel)

# 6. Guardar y visualizar los 3 resultados con Matplotlib
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(imagen_binaria, cmap='gray')
plt.title('1. Binarizada (Con Ruido)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(apertura, cmap='gray')
plt.title('2. Apertura (Opening)\nErosión + Dilatación')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cierre, cmap='gray')
plt.title('3. Cierre (Closing)\nDilatación + Erosión')
plt.axis('off')

plt.tight_layout()
plt.savefig('resultado_morfologia.png')
print("[OK] Imagen de resultados guardada exitosamente como 'resultado_morfologia.png'\n")

# --- CONCLUSIÓN AUTOMÁTICA EN CONSOLA ---
print("=== ANÁLISIS Y CONCLUSIÓN DE LAS OPERACIONES ===")
print("• Apertura (Opening): Elimina pequeñas imperfecciones y puntos blancos aislados en el fondo.")
print("• Cierre (Closing): Elimina agujeros negros dentro de las estructuras blancas principales.")
print("• Elección según la escena: Si el problema principal eran puntos blancos flotando en el fondo,")
print("  la APERTURA es más efectiva. Si el problema eran grietas/huecos dentro de los objetos, el CIERRE fue el mejor.")