# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 11:09:44 2025

@author: sebas
"""

#Red neuronal que clasifique peras y manzanas

#importar librerias
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np

#crear datos simulados
#peras(peso mas bajo, color mas verdoso)
#manzana (peso mas alto, color mas rojizo)

# Características: [peso, color]
# 1 representa "más pesado" o "más rojo", O representa "más liviano" o "más verde"
X_train = np.array([[150, 1], # manzana
                    [130, 1], # manzana
                    [180, 1], # manzana
                    [120, 0], # pera
                    [110, 0], # pera
                    [100, 0]]) # pera

# Etiquetas: 0 es pera, 1 es manzana
y_train = np.array([1, 1, 1, 0, 0, 0])


#crear la red neuronal
# Definir el modelo secuencial
model = Sequential()

# Capa oculta con 3 neuronas y activación relu
model. add (Dense(3, input_shape=(2,), activation='relu'))
# Capa de salida con una neurona y activación sigmoide 
model.add(Dense(1, activation='sigmoid'))
# Compilar el modelo con un optimizador y función de pérdida binaria 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Entrenar la red neuronal
# Entrenar el modelo
model.fit(X_train, y_train, epochs=100)

#Hacer predicciones
# Predecir para una nueva fruta
new_fruit = np.array([[160, 1]]) # Fruta con peso de 160 y más rojiza
prediction = model.predict (new_fruit)

# Convertir la predicción en 0 o 1 
if prediction > 0.5:
    print("Es una manzana")
else:
    print("Es una pera")