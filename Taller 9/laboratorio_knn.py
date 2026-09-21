import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# 1 y 2. Dataset ampliado a 10 puntos con 3 columnas: [Edad, Salario (miles), Hijos]
X_entrenamiento = np.array([
    [20, 30, 0],  # 0: NO COMPRA
    [40, 50, 2],  # 1: COMPRA
    [35, 45, 1],  # 1: COMPRA
    [18, 25, 0],  # 0: NO COMPRA
    [22, 32, 0],  # 0: NO COMPRA
    [45, 60, 3],  # 1: COMPRA
    [50, 65, 2],  # 1: COMPRA
    [25, 28, 1],  # 0: NO COMPRA
    [38, 52, 2],  # 1: COMPRA
    [29, 35, 0]   # 0: NO COMPRA
])

# 3. Etiquetas (0 = NO COMPRA, 1 = COMPRA)
Y_entrenamiento = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0])

# 4. Experimento variando K (n_neighbors)
nuevo_cliente = np.array([[30, 40, 1]])  # [Edad=30, Salario=40, Hijos=1]

# Modelo con K=1
modelo_k1 = KNeighborsClassifier(n_neighbors=1)
modelo_k1.fit(X_entrenamiento, Y_entrenamiento)
pred_k1 = modelo_k1.predict(nuevo_cliente)

# Modelo con K=5
modelo_k5 = KNeighborsClassifier(n_neighbors=5)
modelo_k5.fit(X_entrenamiento, Y_entrenamiento)
pred_k5 = modelo_k5.predict(nuevo_cliente)

print("=== CLASIFICACIÓN DE CLIENTE NUEVO [30 años, $40k, 1 hijo] ===")
print(f"Predicción con K=1: {pred_k1[0]} ({'COMPRA' if pred_k1[0]==1 else 'NO COMPRA'})")
print(f"Predicción con K=5: {pred_k5[0]} ({'COMPRA' if pred_k5[0]==1 else 'NO COMPRA'})")