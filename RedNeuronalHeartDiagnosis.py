# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 09:29:51 2025

@author: sebas
"""

#Red neuronal para problemas del corazón
# Librerías necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import seaborn as sns
#1.	Leer el set de datos
data = pd.read_excel("Heart_diagnosis.xlsx") 
data.columns = data.columns.str.strip()

#2.	Encabezados de las 10 primeras filas
data.head(10)

#3.	Revisar la información
data.info()

#4.	Procesamiento de datos
#•	Investigar si hay valores nulos
data.isnull().sum()
#•	Eliminar variables que tienen un alto porcentaje de valores nulos
#Calcular el porcentaje de valores nulos por columna
porcentaje_nulos = data.isnull().mean() * 100

#Definir un umbral
umbral = 50

#Identificar columnas con más del 50% de nulos
columnas_a_eliminar = porcentaje_nulos[porcentaje_nulos > umbral].index

df= data.drop(columns=columnas_a_eliminar)

#5.	Eliminar datos na
df = df.dropna()


#6.	Verificar que no hay valores erroneos ni nulos
print(df.isnull().sum())
df[(df == '') | (df == ' ') | (df.isin(['NA', 'NaN', 'None']))].count()
print(df.columns)

#7.	Descripcion estadistica
df.describe()

#8.	Visualizar los datos con un heatmap de la correlación de los datos
corr = df.corr()

# Configurar el tamaño del gráfico
plt.figure(figsize=(10, 8))

# Crear el mapa de calor (heatmap)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")

# Título del gráfico
plt.title('Mapa de correlación de las variables')
plt.show()

#restecg, fbs, chol, slope. podriamos leiminar esas columnas ya que no tienen tanta correlacion y evitar un sobreajuste 

#9.	Modelo
#•	Separa las variables X y Y
X = df.drop(columns=['target','restecg', 'fbs', 'chol', 'slope'])
y = df['target']

#•	Dividir set de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#•	Escalado de datos entre 0  y 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#•	Modelo de tipo secuencial: 3 capas,
model = Sequential()

#laprimera densa con la cantidad de neuronas como variables y función de activación Relu,
model.add(Dense(9, activation='relu', input_shape=(X.shape[1],)))

#la segunda densa, dividir la cantidad de neuronas en 2, y función de activación Relu, 
model.add(Dense(4, activation='relu'))

#la tercera será densa, con una neurona y funci[on de activacion sigmoid
model.add(Dense(1, activation='sigmoid'))

#•	Compilar el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#•	Entrenar el modelo con 600 epocas
history = model.fit(X_train, y_train, epochs=600, batch_size=10, validation_split=0.1)

#•	Validar el modelo:almacenar el historico en un dataframe y plotear las perdidas
hist_df = pd.DataFrame(history.history)

# Graficar la pérdida
plt.figure(figsize=(8,5))
plt.plot(hist_df['loss'], label='Pérdida de entrenamiento')
plt.plot(hist_df['val_loss'], label='Pérdida de validación')
plt.title('Evolución de la pérdida del modelo')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.show()

#•	Hacer las predicciones
predicciones = model.predict(X_test)
#Para redondear y evitar numeros decimales
predicciones_binarias = (predicciones > 0.5).astype(int)

#•	Crear la matriz de confusión para y_test y predicciones
matriz = confusion_matrix(y_test, predicciones_binarias)

# Mostrarla en pantalla
print("Matriz de confusión:\n", matriz)

# Visualizarla gráficamente
disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=[0, 1])
disp.plot(cmap='Blues')
plt.title("Matriz de confusión del modelo")
plt.show()

# Precision de la prediccion
print("\nReporte de clasificación:")
print(classification_report(y_test, predicciones_binarias))
