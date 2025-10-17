import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

#CARGA DE DATOS
X_train = np.array([
    [55, 180], [58, 178], [60, 182],   # (Delgado, Alto) -> clase 0
    [52, 160], [50, 158], [54, 165],   # (Delgado, Chaparro) -> clase 1
    [90, 185], [95, 190], [88, 178],   # (Gordo, Alto) -> clase 2
    [85, 165], [92, 170], [80, 160],   # (Gordo, Chaparro) -> clase 3
], dtype=float)

y_train = np.array([
    0,0,0,  # Delgado-Alto
    1,1,1,  # Delgado-Chaparro
    2,2,2,  # Gordo-Alto
    3,3,3   # Gordo-Chaparro
], dtype=int)

#ESCALADO
X_min = X_train.min(axis=0)
X_max = X_train.max(axis=0)
X_train_scaled = (X_train - X_min) / (X_max - X_min + 1e-8)

#One-hot de las etiquetas (4 clases)
num_clases = 4
y_train_oh = to_categorical(y_train, num_classes=num_clases)


#DEFINICIÓN DEL MODELO
model = Sequential()
# Capa oculta: ajusta neuronas si ves under/overfitting (8–16 suele ir bien)
model.add(Dense(8, input_shape=(2,), activation='relu'))

#Capa de salida: 4 neuronas (una por clase) + softmax
model.add(Dense(num_clases, activation='softmax'))

#Compilación: entropía cruzada categórica para clasificación multiclase
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#ENTRENAMIENTO
hist = model.fit(X_train_scaled, y_train_oh, epochs=300, verbose=0)

#PREDICCIÓN PARA UN NUEVO ALUMNO
nuevo_peso_kg = 68.500
nueva_estatura_cm = 180.0
x_new = np.array([[nuevo_peso_kg, nueva_estatura_cm]], dtype=float)

#Escala con los mismos parámetros del entrenamiento
x_new_scaled = (x_new - X_min) / (X_max - X_min + 1e-8)

probas = model.predict(x_new_scaled, verbose=0)[0]   # vector de 4 probabilidades
pred_clase = int(np.argmax(probas))

clase_a_texto = {
    0: "Delgado y Alto",
    1: "Delgado y Chaparro",
    2: "Gordo y Alto",
    3: "Gordo y Chaparro"
}

print(f"Probabilidades por clase (softmax): {probas}")
print(f"Predicción: {pred_clase} → {clase_a_texto[pred_clase]}")


