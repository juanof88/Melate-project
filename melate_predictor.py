# -*- coding: utf-8 -*-
"""
Sistema Predictor Melate Retro
Estrategia: el modelo aprende la distribución de probabilidad de cada número
y genera 100 combinaciones diversas que maximizan la cobertura estadística.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ── Constantes del juego ──────────────────────────────────────────────────────
N_NUMS        = 39    # Números posibles: 1–39
N_ADICIONAL   = 10    # Adicional: 1–10
N_PRINCIPALES = 6
VENTANA       = 10    # Sorteos pasados que el modelo observa
N_JUEGOS      = 100   # Boletos a generar

# ── Carga de datos ────────────────────────────────────────────────────────────
def cargar_datos():
    try:
        from google.colab import files
        uploaded = files.upload()
        filename = list(uploaded.keys())[0]
        df = pd.read_csv(filename)
    except Exception:
        print("Modo local: leyendo melate.csv")
        df = pd.read_csv("melate.csv")

    columnas = ["F1", "F2", "F3", "F4", "F5", "F6", "F7"]
    if not all(c in df.columns for c in columnas):
        raise ValueError(f"El CSV debe tener las columnas: {columnas}")
    return df[columnas].dropna().reset_index(drop=True)


def datos_ejemplo():
    np.random.seed(42)
    rows = []
    for _ in range(300):
        nums = sorted(np.random.choice(range(1, 40), 6, replace=False))
        adicional = np.random.randint(1, 11)
        rows.append(nums + [adicional])
    return pd.DataFrame(rows, columns=["F1","F2","F3","F4","F5","F6","F7"])


try:
    datos = cargar_datos()
    print(f"Datos cargados: {len(datos)} sorteos")
except Exception as e:
    print(f"  {e}. Usando datos de ejemplo.")
    datos = datos_ejemplo()

# ── Codificación one-hot por sorteo ──────────────────────────────────────────
# Cada sorteo → vector binario de 39 bits (1 = número salió) + 10 bits adicional
def sorteo_a_vector(row):
    v_main = np.zeros(N_NUMS, dtype=np.float32)
    for col in ["F1","F2","F3","F4","F5","F6"]:
        v_main[int(row[col]) - 1] = 1.0

    v_adic = np.zeros(N_ADICIONAL, dtype=np.float32)
    v_adic[int(row["F7"]) - 1] = 1.0

    return np.concatenate([v_main, v_adic])   # tamaño 49


vectores = np.array([sorteo_a_vector(datos.iloc[i]) for i in range(len(datos))],
                    dtype=np.float32)

# ── Secuencias de entrada/salida ──────────────────────────────────────────────
def crear_secuencias(data, ventana):
    X, y = [], []
    for i in range(len(data) - ventana):
        X.append(data[i : i + ventana])
        y.append(data[i + ventana])
    return np.array(X), np.array(y)


X, y = crear_secuencias(vectores, VENTANA)

split = int(len(X) * 0.85)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print(f"Entrenamiento: {len(X_train)}  |  Validación: {len(X_val)}")

# ── Modelo: LSTM → distribución de probabilidad ───────────────────────────────
# Salidas: 39 probs para principales + 10 probs para adicional (softmax/sigmoid)
def construir_modelo(ventana, n_features=49):
    inp = layers.Input(shape=(ventana, n_features))

    x = layers.LSTM(256, return_sequences=True)(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(128)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)

    # Cabeza para los 39 números principales (sigmoid: prob independiente por número)
    out_main = layers.Dense(N_NUMS, activation="sigmoid", name="principales")(x)
    # Cabeza para el adicional (softmax: solo 1 número adicional sale)
    out_adic = layers.Dense(N_ADICIONAL, activation="softmax", name="adicional")(x)

    return models.Model(inp, [out_main, out_adic])


modelo = construir_modelo(VENTANA)
modelo.summary()

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss={
        "principales": "binary_crossentropy",   # prob por número
        "adicional":   "categorical_crossentropy"
    },
    metrics={"principales": "accuracy", "adicional": "accuracy"}
)

# ── Preparar targets separados ────────────────────────────────────────────────
y_main_train = y_train[:, :N_NUMS]
y_adic_train = y_train[:, N_NUMS:]
y_main_val   = y_val[:, :N_NUMS]
y_adic_val   = y_val[:, N_NUMS:]

cbs = [
    callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6),
]

history = modelo.fit(
    X_train,
    {"principales": y_main_train, "adicional": y_adic_train},
    validation_data=(X_val, {"principales": y_main_val, "adicional": y_adic_val}),
    epochs=300,
    batch_size=32,
    callbacks=cbs,
    verbose=1
)

# ── Obtener distribución de probabilidad ──────────────────────────────────────
ultima_ventana = vectores[-VENTANA:].reshape(1, VENTANA, 49)
prob_main, prob_adic = modelo.predict(ultima_ventana, verbose=0)

prob_main = prob_main[0]   # shape (39,)  — prob de cada número 1-39
prob_adic = prob_adic[0]   # shape (10,)  — prob de cada adicional 1-10

# Normalizar probs principales para usarlas como pesos de muestreo
pesos = prob_main / prob_main.sum()

print("\nTop 15 números más probables (modelo):")
ranking = np.argsort(pesos)[::-1]
for pos, idx in enumerate(ranking[:15], 1):
    print(f"  {pos:2}. Número {idx+1:2d}  —  prob {pesos[idx]:.4f}")

# ── Generación de 100 combinaciones diversas ──────────────────────────────────
def generar_juegos(pesos_main, pesos_adic, n_juegos=100, semillas=5):
    """
    Genera `n_juegos` combinaciones únicas y diversas.

    Estrategia:
    1. Muestreo ponderado: favorece números más probables pero no los fija.
    2. Temperatura variable: algunas tiradas más 'calientes' (concentradas)
       y otras más frías (exploración), aumentando diversidad.
    3. Deduplicación: descarta combinaciones repetidas.
    """
    numeros = np.arange(1, N_NUMS + 1)
    adicionales = np.arange(1, N_ADICIONAL + 1)
    temperaturas = np.linspace(0.5, 2.0, semillas)   # 0.5=concentrado, 2.0=exploratorio

    juegos = set()
    intentos = 0

    while len(juegos) < n_juegos and intentos < n_juegos * 50:
        # Variar temperatura cíclicamente para asegurar diversidad
        temp = temperaturas[intentos % len(temperaturas)]
        pesos_temp = pesos_main ** (1.0 / temp)
        pesos_temp = pesos_temp / pesos_temp.sum()

        combinacion = tuple(sorted(
            np.random.choice(numeros, size=N_PRINCIPALES, replace=False, p=pesos_temp)
        ))

        if combinacion not in juegos:
            juegos.add(combinacion)

        intentos += 1

    # Si no se completaron 100, rellenar con muestreo uniforme
    while len(juegos) < n_juegos:
        combinacion = tuple(sorted(
            np.random.choice(numeros, size=N_PRINCIPALES, replace=False)
        ))
        juegos.add(combinacion)

    # Número adicional: muestreo ponderado por prob del modelo
    juegos_lista = [[int(n) for n in c] for c in juegos]
    adics = np.random.choice(adicionales, size=n_juegos, p=pesos_adic)

    return juegos_lista, [int(a) for a in adics]


juegos, adics = generar_juegos(pesos, prob_adic)

# ── Métricas de la muestra ────────────────────────────────────────────────────
todos_numeros = [n for j in juegos for n in j]
cobertura = len(set(todos_numeros))
# Números cubiertos al menos 1 vez vs 39 posibles
pct_cobertura = cobertura / N_NUMS * 100

print(f"\n{'='*50}")
print(f"  100 JUEGOS GENERADOS — MELATE RETRO")
print(f"{'='*50}")
print(f"  Cobertura de números 1-39:  {cobertura}/39  ({pct_cobertura:.1f}%)")
print(f"  Adicionales distintos usados: {len(set(adics))}/10")
print(f"{'='*50}")

for i, (juego, adic) in enumerate(zip(juegos, adics), 1):
    print(f"  {i:3}. {juego}  +  [{adic}]")

# ── Exportar a CSV ────────────────────────────────────────────────────────────
df_juegos = pd.DataFrame(juegos, columns=["F1","F2","F3","F4","F5","F6"])
df_juegos["F7"] = adics
df_juegos.index = range(1, N_JUEGOS + 1)
df_juegos.to_csv("mis_100_juegos.csv")
print("\nArchivo guardado: mis_100_juegos.csv")

# Descarga automática en Colab
try:
    from google.colab import files
    files.download("mis_100_juegos.csv")
except Exception:
    pass
