# -- coding: utf-8 --
"""
Created on Tue Oct 14 10:43:41 2025
@author: sebas
"""

# Librerías necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# 1.	Lee el set de datos
data = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
data.columns = data.columns.str.strip()

# 2. Revisa los primeros 5 datos
data.head()

# 3. Revisa la información del set de datos
data.info()

# 4. Elimina las columnas necesarias y los datos nulos
dataDrop= [
    'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'CWE Flag Count',
    'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate'
]
data = data.drop(columns=dataDrop, errors='ignore')

rate_cols = ['Flow Bytes/s', 'Flow Packets/s']
for c in rate_cols:
    if c in data.columns:
        # Reemplaza ±inf -> NaN
        data[c].replace([np.inf, -np.inf], np.nan, inplace=True)
        # Tasa negativa no tiene sentido / clamp a 0
        data.loc[data[c] < 0, c] = 0
        # Imputa NaN con P99.9 y recorta a P99.9 para evitar colas extremas
        p999 = data[c].quantile(0.999)
        data[c].fillna(p999, inplace=True)
        data[c] = data[c].clip(upper=p999)
        # Estabiliza escala
        data[c] = np.log1p(data[c])

heavy_cols = [
    'Flow Duration',
    'Fwd IAT Total','Bwd IAT Total',
    'Flow IAT Mean','Flow IAT Max','Flow IAT Min',
    'Fwd IAT Mean','Fwd IAT Max','Fwd IAT Min',
    'Bwd IAT Mean','Bwd IAT Max','Bwd IAT Min',
    'Idle Mean','Idle Max','Idle Min',
    'Active Mean','Active Max','Active Min'
]
for c in heavy_cols:
    if c in data.columns:
        data.loc[data[c] < 0, c] = 0
        data[c] = np.log1p(data[c])


data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
        
# 5. Codifica la etiqueta LABEL con 0 para benigno y 1 para maligno
for c in data.columns:
    if c != 'Label':
        data[c] = pd.to_numeric(data[c], errors='coerce')

encoder = LabelEncoder()
data['Label'] = encoder.fit_transform(data['Label'])  # BENIGN -> 0, DDoS -> 1

# 6 Separar en X y Y
X = data.drop(columns=['Label'])
y = data['Label']

# 7. Divide el set de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 8. Normaliza las características
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 9. Define el modelo de tipo secuencial con:
model = Sequential()

#a.	Una capa oculta densa de 32 neuronas y funcionde activación Relu
model.add(Dense(32, activation='relu', input_shape=(X.shape[1],)))

#b.	Una segunda capa oculta con 16 neuronas y función de activación Relu
model.add(Dense(16, activation='relu'))

#c.	Una capa de saluda con una neurona y función de activación sigmoid
model.add(Dense(1, activation='sigmoid'))

# 10. Compila el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 11. Muestra la estructura del modelo
model.summary()

# 12. Entrena el modelo con 20 épocas
historial = model.fit(X_train, y_train, epochs=20, batch_size=10, validation_data=(X_test, y_test)) ## Se utiliza validation_data en lugar de validation_split porque ya se realizó una separación previa
# entre entrenamiento y prueba con train_test_split. Esto permite evaluar el modelo con un conjunto
# completamente independiente y reproducible, evitando que el modelo vea datos de validación durante el entrenamiento.

# 13. Evalúa el modelo con los valores de prueba
# Evaluar el modelo en el conjunto de prueba
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Pérdida en el conjunto de prueba: {loss}")
print(f"Precisión en el conjunto de prueba: {accuracy * 100:.2f}%")

# 14.Grafica la precisión y la pérdida durante el entrenamiento 
plt.plot(historial.history['accuracy'], label='Entrenamiento')
plt.plot(historial.history['val_accuracy'], label='Validación')
plt.title('Precisión')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend(); plt.grid(True); plt.show()

plt.plot(historial.history['loss'], label='Entrenamiento')
plt.plot(historial.history['val_loss'], label='Validación')
plt.title('Pérdida')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend(); plt.grid(True); plt.show()

# 15. Realizar predicciones en X_test
predicciones = model.predict(X_test)
# 16.	Muestra las primeras 10 predicciones y compara con los valore reales.
pred_bin = (predicciones > 0.5).astype(int).ravel()
y_test_reset = y_test.reset_index(drop=True).to_numpy()

for i in range(10):
    print(f"Muestra {i+1}: Predicción = {pred_bin[i]}, Real = {y_test_reset[i]}")
    
