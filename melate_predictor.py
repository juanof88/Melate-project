# -*- coding: utf-8 -*-
"""Sistema Predictor Melate Retro - Versión Mejorada"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ── Constantes del juego ──────────────────────────────────────────────────────
RANGO_PRINCIPALES = (1, 39)   # Melate Retro: 6 números del 1 al 39
RANGO_ADICIONAL   = (1, 10)   # Número adicional
N_PRINCIPALES     = 6
VENTANA           = 5         # Sorteos pasados que el modelo "ve" a la vez

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

    columnas = ["N1", "N2", "N3", "N4", "N5", "N6", "Adicional"]
    if not all(c in df.columns for c in columnas):
        raise ValueError(f"El CSV debe tener las columnas: {columnas}")
    return df[columnas].dropna().reset_index(drop=True)


def datos_ejemplo():
    """Datos sintéticos para pruebas sin archivo."""
    np.random.seed(42)
    rows = []
    for _ in range(200):
        nums = sorted(np.random.choice(range(1, 40), 6, replace=False))
        adicional = np.random.randint(1, 11)
        rows.append(nums + [adicional])
    return pd.DataFrame(rows, columns=["N1","N2","N3","N4","N5","N6","Adicional"])


try:
    datos = cargar_datos()
    print(f"Datos cargados: {len(datos)} sorteos")
except Exception as e:
    print(f"⚠ {e}. Usando datos de ejemplo.")
    datos = datos_ejemplo()

# ── Normalización ─────────────────────────────────────────────────────────────
valores = datos.values.astype(np.float32)

# Normalización manual (min-max) para poder invertirla fácilmente
mins  = valores.min(axis=0)
maxs  = valores.max(axis=0)
rango = maxs - mins + 1e-8

def normalizar(x):
    return (x - mins) / rango

def desnormalizar(x):
    return x * rango + mins

norm = normalizar(valores)

# ── Creación de secuencias (ventana deslizante) ───────────────────────────────
def crear_secuencias(data, ventana):
    X, y = [], []
    for i in range(len(data) - ventana):
        X.append(data[i : i + ventana])        # últimos `ventana` sorteos
        y.append(data[i + ventana])             # siguiente sorteo
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


X, y = crear_secuencias(norm, VENTANA)

split = int(len(X) * 0.85)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print(f"Entrenamiento: {len(X_train)} muestras  |  Validación: {len(X_val)} muestras")

# ── Arquitectura: LSTM + cabeza densa ─────────────────────────────────────────
def construir_modelo(ventana, n_features=7):
    inp = layers.Input(shape=(ventana, n_features))

    x = layers.LSTM(128, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation="relu")(x)

    # Salida separada: 6 principales + 1 adicional
    out = layers.Dense(n_features, activation="sigmoid")(x)  # sigmoid → [0,1] como la normalización min-max

    return models.Model(inp, out)


modelo = construir_modelo(VENTANA)
modelo.summary()

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"]
)

# ── Callbacks ─────────────────────────────────────────────────────────────────
cbs = [
    callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
]

# ── Entrenamiento ─────────────────────────────────────────────────────────────
history = modelo.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=16,
    callbacks=cbs,
    verbose=1
)

# ── Postprocesamiento de predicciones ─────────────────────────────────────────
def prediccion_valida(pred_norm):
    """
    Convierte la salida normalizada del modelo a números válidos del juego:
    - 6 principales únicos en [1, 39]
    - 1 adicional en [1, 10]
    """
    pred = desnormalizar(pred_norm)

    # Números principales: redondear, clampar al rango y asegurar únicos
    principales_raw = np.clip(np.round(pred[:N_PRINCIPALES]).astype(int),
                              RANGO_PRINCIPALES[0], RANGO_PRINCIPALES[1])

    # Resolver duplicados: si hay repetidos, reemplazar por números faltantes
    unicos = list(dict.fromkeys(principales_raw))           # mantiene orden, elimina dupes
    if len(unicos) < N_PRINCIPALES:
        usados   = set(unicos)
        faltantes = [n for n in range(RANGO_PRINCIPALES[0], RANGO_PRINCIPALES[1] + 1)
                     if n not in usados]
        np.random.shuffle(faltantes)
        unicos += faltantes[: N_PRINCIPALES - len(unicos)]

    principales = sorted(unicos[:N_PRINCIPALES])

    # Número adicional
    adicional = int(np.clip(round(pred[N_PRINCIPALES]),
                            RANGO_ADICIONAL[0], RANGO_ADICIONAL[1]))

    return principales, adicional


# Tomar los últimos `VENTANA` sorteos como entrada
ultima_ventana = norm[-VENTANA:].reshape(1, VENTANA, 7)
pred_norm = modelo.predict(ultima_ventana, verbose=0)[0]

principales, adicional = prediccion_valida(pred_norm)

print("\n" + "="*45)
print("  PREDICCIÓN PRÓXIMO SORTEO MELATE RETRO")
print("="*45)
print(f"  Números: {principales}")
print(f"  Adicional: {adicional}")
print("="*45)
print("\n⚠  Recordatorio: la lotería es aleatoria por diseño.")
print("   Este modelo encuentra patrones estadísticos,")
print("   pero no puede garantizar resultados.")
