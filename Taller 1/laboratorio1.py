import numpy as np

# 1. Crear matriz de prueba 5x5 simulando radiografía sobreexpuesta (valores entre 200 y 255)
matriz_original = np.random.randint(200, 255, (5, 5), dtype=np.uint8)

# 2. Parámetros: Reducción de contraste del 50% (alpha = 0.5) y disminución de brillo en 50 (beta = -50)
alpha = 0.5
beta = -50.0

# Convertir a float32 para realizar cálculos matemáticos precisos
matriz_float = matriz_original.astype(np.float32)

# Aplicar la ecuación lineal: Anueva = alpha * A + beta
matriz_procesada = (alpha * matriz_float) + beta

# 3. Saturación (Clipping) para mantener rango [0, 255] y reconversión a uint8
matriz_procesada = np.clip(matriz_procesada, 0, 255).astype(np.uint8)

# 4. Mostrar resultados
print("=== MATRIZ ORIGINAL (SOBREEXPUESTA) ===")
print(matriz_original)
print("\n=== MATRIZ PROCESADA (CONTRASTE Y BRILLO AJUSTADOS) ===")
print(matriz_procesada)