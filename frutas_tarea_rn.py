# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 11:30:38 2025

@author: sebas
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#características [peso y color]
#0 representa más color de mango y 1 más color de naranja
X_train = np.array([[150,1], #naranja
                    [130,1], #naranja
                    [180,1], #naranja
                    [120,0], #mango
                    [110,0], #mango
                    [100,0]]) #mango

#etiqueta 0 es para mango y 1 para naranja
y_train = np.array([1,1,1,0,0,0])

#crear la red neuronal definiendo el modelo secuencial
model = Sequential()

#Capa oculta con 5 neuronas y activación relu
model.add(Dense(5, input_shape=(2,), activation='relu'))

#Capa de salida con una neurona y activación sigmoide
model.add(Dense(1, activation='sigmoid'))

#compilar el modelo con un optimizador y función de pérdida binaria
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#entrenar la red neuronal y el modelo
model.fit(X_train, y_train, epochs=200)

#Hacer predicción
#Predecir para un nuevo mango
new_fruit = np.array([[110,0]])
prediction = model.predict(new_fruit)

if prediction > 0.5:
    print("Es una naranja")
else:
    print("Es un mango")
