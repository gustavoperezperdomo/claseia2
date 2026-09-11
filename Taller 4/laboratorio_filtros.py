import cv2         #respuesta 3 taller 4
import numpy as np

def generar_imagen_con_ruido_sal_pimienta(alto=300, ancho=300, cantidad_ruido=0.05):
    """
    Crea una imagen sintética con figuras y le añade ruido de Sal y Pimienta.
    """
    # 1. Crear una imagen base con un rectángulo blanco en el centro
    imagen = np.zeros((alto, ancho), dtype=np.uint8)
    cv2.rectangle(imagen, (50, 50), (250, 250), 180, -1)
    cv2.circle(imagen, (150, 150), 50, 255, -1)
    
    # 2. Agregar Ruido de Sal y Pimienta
    imagen_ruidosa = imagen.copy()
    num_ruido = int(cantidad_ruido * alto * ancho)
    
    # Sal (píxeles blancos)
    coords_sal = [np.random.randint(0, i - 1, int(num_ruido / 2)) for i in imagen.shape]
    imagen_ruidosa[tuple(coords_sal)] = 255
    
    # Pimienta (píxeles negros)
    coords_pimienta = [np.random.randint(0, i - 1, int(num_ruido / 2)) for i in imagen.shape]
    imagen_ruidosa[tuple(coords_pimienta)] = 0
    
    return imagen_ruidosa

def main():
    # 1. Generar la imagen con ruido
    img_ruidosa = generar_imagen_con_ruido_sal_pimienta()
    
    # Tamaño de Kernel agresivo (7x7) como lo solicita el taller
    ksize = 7
    
    # 2. Aplicar los tres filtros
    blur_media = cv2.blur(img_ruidosa, (ksize, ksize))
    blur_gauss = cv2.GaussianBlur(img_ruidosa, (ksize, ksize), 0)
    blur_mediana = cv2.medianBlur(img_ruidosa, ksize)
    
    # 3. Concatenar las imágenes en una sola tira horizontal para fácil visualización
    # Fila 1: Ruidosa | Media
    # Fila 2: Gaussiano | Mediana
    top_row = np.hstack((img_ruidosa, blur_media))
    bottom_row = np.hstack((blur_gauss, blur_mediana))
    resultado_final = np.vstack((top_row, bottom_row))
    
    # Agregar etiquetas de texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(resultado_final, "1. Ruidosa", (10, 30), font, 0.7, (255), 2)
    cv2.putText(resultado_final, "2. Media (7x7)", (310, 30), font, 0.7, (255), 2)
    cv2.putText(resultado_final, "3. Gaussiano (7x7)", (10, 330), font, 0.7, (255), 2)
    cv2.putText(resultado_final, "4. Mediana (7x7)", (310, 330), font, 0.7, (255), 2)
    
    # 4. Guardar en disco (Compatible con Github Codespaces)
    nombre_archivo = "resultado_comparativo.png"
    cv2.imwrite(nombre_archivo, resultado_final)
    print(f"✅ Procesamiento completado. Imagen guardada como '{nombre_archivo}'.")

if __name__ == "__main__":
    main()