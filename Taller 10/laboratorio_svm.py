import numpy as np
from sklearn.svm import SVC

print("=== PARTE 1: MODELO CON KERNEL LINEAL ===")
# 1. Dataset Original
X = np.array([
    [2, 2],  # Clase 0
    [3, 3],  # Clase 0
    [4, 2],  # Clase 0
    [6, 6],  # Clase 1
    [7, 8],  # Clase 1
    [8, 7]   # Clase 1
])
Y = np.array([0, 0, 0, 1, 1, 1])

# Entrenar modelo lineal
modelo_lineal = SVC(kernel='linear')
modelo_lineal.fit(X, Y)

# Vectores de Soporte detectados
vectores_lineal = modelo_lineal.support_vectors_
print("Vectores de Soporte (Kernel Lineal):\n", vectores_lineal)

pred_lineal = modelo_lineal.predict([[5, 4]])
print("Predicción para el punto [5, 4]:", pred_lineal[0], ("(Clase A)" if pred_lineal[0]==0 else "(Clase B)"))


print("\n=== PARTE 2: EXPERIMENTO DE 'ENGAÑO' AL KERNEL LINEAL ===")
# 2. Agregar punto [5, 5] etiquetado como Clase 0
X_mod = np.vstack([X, [5, 5]])
Y_mod = np.append(Y, 0)

# 3. Re-entrenar modelo lineal con el punto complejo
modelo_lineal_engano = SVC(kernel='linear')
modelo_lineal_engano.fit(X_mod, Y_mod)
print("Predicción del modelo lineal para [5, 5]:", modelo_lineal_engano.predict([[5, 5]])[0])


print("\n=== PARTE 3: SOLUCIÓN CON KERNEL RBF (KERNEL TRICK) ===")
# 4. Cambiar a Kernel RBF
modelo_rbf = SVC(kernel='rbf')
modelo_rbf.fit(X_mod, Y_mod)

pred_rbf = modelo_rbf.predict([[5, 5]])
print("Predicción con Kernel RBF para [5, 5]:", pred_rbf[0], ("(Correcto: Clase A)" if pred_rbf[0]==0 else "(Incorrecto)"))