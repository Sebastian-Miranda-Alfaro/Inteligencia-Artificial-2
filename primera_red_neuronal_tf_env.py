# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 09:05:02 2025

@author: sebas
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Datos de entrada (peso, color)
X = np.array([
    [150, 1],
    [130, 1],
    [180, 1],
    [120, 0],
    [110, 0],
    [100, 0]
])

# Etiquetas (1 = manzana, 0 = pera)
y = np.array([1, 1, 1, 0, 0, 0])

# Crear modelo secuencial
model = Sequential()
model.add(Dense(3, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilar
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entrenar
model.fit(X, y, epochs=100, verbose=0)

# Predicción de nueva fruta
nueva_manzana = np.array([[160, 1]])
prediccion = model.predict(nueva_manzana)

# Convertir predicción en texto
if prediccion >= 0.5:
    print("Es una manzana")
else:
    print("Es una pera")

print("Valor predicho:", prediccion)
