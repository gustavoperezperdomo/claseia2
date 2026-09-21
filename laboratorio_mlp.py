import numpy as np

# 1. Función de Activación Sigmoide
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

print("=== PARTE 1 & 2: INFERENCIA DE 1 CLIENTE ===")
X_1 = np.array([0.5, 0.8, 0.2])

W1 = np.array([
    [0.1,  0.2,  0.3,  0.4],
    [-0.5, 0.6,  0.7, -0.8],
    [0.9, -0.1,  0.2,  0.3]
])
b1 = np.array([0.1, 0.2, 0.3, 0.4])

# Capa Oculta (1 Cliente)
Z1_single = np.dot(X_1, W1) + b1
A1_single = sigmoide(Z1_single)

print("Valores Z1 (Combinación Lineal Pura):\n", Z1_single)
print("Valores A1 (Transformación Sigmoide [0, 1]):\n", np.round(A1_single, 4))

# Capa de Salida (1 Cliente)
W2 = np.array([0.5, 0.6, 0.7, 0.8])
b2 = np.array([-0.1])

Z2_single = np.dot(A1_single, W2) + b2
Salida_single = sigmoide(Z2_single)
print("Predicción (1 Cliente):", np.round(Salida_single[0], 4))


print("\n" + "="*50)
print("=== PARTE 3 & 4: RETO DIMENSIONAL (BATCH DE 2 CLIENTES) ===")

# Matriz X de 2x3 (2 Clientes, 3 Características cada uno)
X_batch = np.array([
    [0.5, 0.8, 0.2],  # Cliente 1
    [0.1, 0.9, 0.9]   # Cliente 2
])

# Capa Oculta (2 Clientes al tiempo)
Z1_batch = np.dot(X_batch, W1) + b1
A1_batch = sigmoide(Z1_batch)

# Capa de Salida (2 Clientes al tiempo)
Z2_batch = np.dot(A1_batch, W2) + b2
Salida_batch = sigmoide(Z2_batch)

print("Dimensiones de la entrada (X_batch):", X_batch.shape)
print("Dimensiones de la salida oculta (A1_batch):", A1_batch.shape)
print("Dimensiones de las salidas finales:", Salida_batch.shape)
print("\nProbabilidad de Aprobación por Cliente:")
for i, prob in enumerate(Salida_batch):
    print(f" • Cliente {i+1}: {np.round(prob, 4)} ({np.round(prob*100, 2)}%)")