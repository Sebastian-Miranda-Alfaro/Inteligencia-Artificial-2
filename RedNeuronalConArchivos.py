# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 08:56:47 2025

@author: sebas
"""

# Importar librerias necesarias
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.utils import to_categorical


# Cargar los datos desde el archivo CSV
data = pd.read_csv("zoo.csv")

# Ver las primeras filas del conjunto de datos
print(data.head())

# Separar características (X) y etiquetas (Y)
X = data.drop(['animal_name','class_type'], axis=1) # Quitamos el nombre del animal y la clase objetivo
y = data['class_type'] # Esta es la columna que contiene las clases (objetivo)

# Codificar las etiquetas de las clases
encoder = LabelEncoder ()
y_encoded = encoder.fit_transform(y)

# Convertir etiquetas a formato categórico (One-hot encoding)
y_categorical = to_categorical(y_encoded)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Normalizar los datos (escalado)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear el modelo de la red neuronal
model = Sequential()

# Añadir la capa de entrada (y primera capa oculta)
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu')) #64 neuronas

# Añadir una segunda capa oculta
model.add(Dense(32, activation='relu')) # 32 neuronas

# Añadir la capa de salida con tantas neuronas como clases (7 en este caso) y softmax para multiclase
model.add(Dense(y_categorical.shape[1], activation='softmax'))

# Compilar el modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.1)

# Evaluar el modelo en el conjunto de prueba
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Pérdida en el conjunto de prueba: {loss}")
print(f"Precisión en el conjunto de prueba: {accuracy * 100:.2f}%")

# Hacer predicciones con el conjunto de prueba
predicciones = model.predict(X_test)

# Convertir las predicciones a clases (la clase con la mayor probabilidad)
predicciones_clase = np.argmax(predicciones, axis=1)

#Mostrar algunas predicciones
print(f"Predicciones (indice de clase): {predicciones_clase}")