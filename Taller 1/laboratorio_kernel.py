import numpy as np

# 1. Definir la sección de la imagen I (3x3) y el Kernel K de Realce (3x3)
I = np.array([
    [100, 100, 100],
    [100, 200, 100],
    [100, 100, 100]
], dtype=np.float32)

K = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)

# 2. Multiplicación Hadamard (elemento a elemento) y suma total de los elementos
producto_hadamard = I * K
poxel_central = np.sum(producto_hadamard)

# 3. Imprimir el resultado
print("=== MATRIZ SECCIÓN DE IMAGEN (I) ===")
print(I)
print("\n=== KERNEL DE REALCE (K) ===")
print(K)
print("\n=== RESULTADO PRODUCTO HADAMARD ===")
print(producto_hadamard)
print(f"\nValor del píxel central calculado: {poxel_central}")