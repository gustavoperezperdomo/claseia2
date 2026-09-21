import numpy as np

# 1. Función de Activación (Escalón)
def funcion_escalon(z):
    if z >= 0:
        return 1
    else:
        return 0

# 2. Estructura del Perceptrón
def perceptron(X, W, b):
    # Combinación lineal (Producto punto)
    Z = np.dot(X, W) + b
    # Activación
    salida = funcion_escalon(Z)
    return salida

# --- COMPROBACIÓN DEL TALLER ANALÍTICO ---
print("=== TALLER ANALÍTICO: CRÉDITO BANCARIO ===")
X_credito = np.array([50, 20])
W_credito = np.array([0.8, -0.5])
b_credito = -10

Z_calculado = np.dot(X_credito, W_credito) + b_credito
resultado_credito = perceptron(X_credito, W_credito, b_credito)

print(f"Valor Z calculado: {Z_calculado}")
print(f"Resultado de la neurona: {resultado_credito} ({'Aprobado' if resultado_credito == 1 else 'Rechazado'})\n")


# --- TALLER DE LABORATORIO: COMPUERTA LOGICA OR ---
print("=== TALLER DE LABORATORIO: COMPUERTA LOGICA OR ===")

# Pesos y sesgo ajustados manualmente para cumplir la tabla de verdad OR
pesos_or = np.array([0.5, 0.5])
sesgo_or = -0.2

# Posibles combinaciones de entrada [X1, X2]
entradas_or = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

print("Entrada | Salida Esperada | Salida Obtenida | Estado")
print("-" * 50)

tabla_correcta = [0, 1, 1, 1]

for idx, x in enumerate(entradas_or):
    salida = perceptron(x, pesos_or, sesgo_or)
    esperada = tabla_correcta[idx]
    estado = "OK" if salida == esperada else "ERROR"
    print(f" {x}  |        {esperada}        |        {salida}        |  {estado}")

print("\n[OK] Pesos configurados exitosamente: W =", pesos_or, "Sesgo b =", sesgo_or)