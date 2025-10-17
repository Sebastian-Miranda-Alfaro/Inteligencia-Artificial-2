# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 22:32:52 2024

@author: mayte
"""

import cv2

# Cargar el clasificador en cascada de Haar para la detección de rostros
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capturar video desde la cámara web
cap = cv2.VideoCapture(0)

while True:
    # Leer el cuadro de video
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convertir el cuadro a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostros en la imagen en escala de grises
    faces = faceClassif.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Procesar cada rostro detectado
    for (x, y, w, h) in faces:
        # Extraer el área del rostro en la imagen original
        face_region = gray[y:y+h, x:x+w]
        
        # Aplicar umbralización binaria a la región del rostro
        _, face_binary = cv2.threshold(face_region, 128, 255, cv2.THRESH_BINARY)
        
        # Convertir la imagen binaria a BGR para combinarla con la imagen original
        face_binary_bgr = cv2.cvtColor(face_binary, cv2.COLOR_GRAY2BGR)
        
        # Reemplazar la región del rostro en la imagen original con la versión binaria
        frame[y:y+h, x:x+w] = face_binary_bgr
    
    # Mostrar el video con los rostros en binario
    cv2.imshow('Detección de Rostros en Binario', frame)
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el objeto de captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()