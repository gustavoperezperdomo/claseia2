import cv2
import matplotlib.pyplot as plt

# 1. Cargar una imagen RGB/BGR de prueba
nombre_imagen = 'muestra.jpg'
imagen = cv2.imread(nombre_imagen)

if imagen is None:
    print(f"Error: No se encontró la imagen '{nombre_imagen}'. Verifica que esté en la carpeta 'Taller 2'.")
    exit()

# 2. Separar la imagen en sus 3 canales (B, G, R)
canal_azul = imagen[:, :, 0]
canal_verde = imagen[:, :, 1]
canal_rojo = imagen[:, :, 2]

# 3. Calcular el histograma de cada canal por separado (256 niveles de 0 a 255)
hist_azul = cv2.calcHist([imagen], [0], None, [256], [0, 256])
hist_verde = cv2.calcHist([imagen], [1], None, [256], [0, 256])
hist_rojo = cv2.calcHist([imagen], [2], None, [256], [0, 256])

# 4. Graficar los tres histogramas superpuestos usando Matplotlib
plt.figure(figsize=(10, 6))
plt.plot(hist_azul, color='blue', label='Canal Azul (B)')
plt.plot(hist_verde, color='green', label='Canal Verde (G)')
plt.plot(hist_rojo, color='red', label='Canal Rojo (R)')

plt.title('Histograma Comparativo de Canales BGR')
plt.xlabel('Nivel de Intensidad del Píxel (0 - 255)')
plt.ylabel('Cantidad de Píxeles (Frecuencia)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Guardar el gráfico como archivo de imagen en la carpeta
plt.savefig('histograma_analisis.png')
print("[OK] Histograma generado y guardado como 'histograma_analisis.png'.")

# 5. Cálculo automático del canal dominante para apoyar la conclusión
promedio_b = canal_azul.mean()
promedio_g = canal_verde.mean()
promedio_r = canal_rojo.mean()

print("\n=== ANÁLISIS ESTADÍSTICO DE INTENSIDAD PROMEDIO ===")
print(f"Intensidad promedio Canal Azul:  {promedio_b:.2f}")
print(f"Intensidad promedio Canal Verde: {promedio_g:.2f}")
print(f"Intensidad promedio Canal Rojo:  {promedio_r:.2f}")

if promedio_r > promedio_g and promedio_r > promedio_b:
    color_dominante = "ROJO"
elif promedio_g > promedio_r and promedio_g > promedio_b:
    color_dominante = "VERDE"
else:
    color_dominante = "AZUL"

print(f"\nConclusión: El color dominante en la iluminación general es el {color_dominante}.")