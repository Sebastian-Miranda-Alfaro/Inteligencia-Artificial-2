# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 07:31:05 2025

@author: sebas
"""

import cv2

# Clasificador para detección de rostros
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Captura de video
cap = cv2.VideoCapture(0)

# Opción inicial (1 = binario, 2 = gris)
opcion = 1
foto_bn_guardada = False
foto_gris_guardada = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceClassif.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_region = gray[y:y+h, x:x+w]

        if opcion == 1:  # Rostro en blanco y negro
            _, face_processed = cv2.threshold(face_region, 128, 255, cv2.THRESH_BINARY)
        else:  # Rostro en gris
            face_processed = face_region

        face_final = cv2.cvtColor(face_processed, cv2.COLOR_GRAY2BGR)
        frame[y:y+h, x:x+w] = face_final

    cv2.imshow('Video en tiempo real', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):
        opcion = 1
        print("Modo: Blanco y Negro ")
    elif key == ord('2'):
        opcion = 2
        print("Modo: Escala de Grises")
    elif key == ord('b') and opcion == 1:
        cv2.imwrite("foto_bn.jpg", frame)
        foto_bn_guardada = True
        print("Guardada: foto_bn.jpg")
    elif key == ord('g') and opcion == 2:
        cv2.imwrite("foto_gris.jpg", frame)
        foto_gris_guardada = True
        print("Guardada: foto_gris.jpg")
    elif key == ord('q'):  # Salir
        break

cap.release()
cv2.destroyAllWindows()

# ========= Rompecabezas =========
if foto_bn_guardada and foto_gris_guardada:
    img1 = cv2.imread("foto_bn.jpg")
    img2 = cv2.imread("foto_gris.jpg")

    if img1 is not None and img2 is not None:
        
        img1 = cv2.resize(img1, (400, 400))
        img2 = cv2.resize(img2, (400, 400))

        h, w, _ = img1.shape
        mitad_h, mitad_w = h // 2, w // 2

        # Cuadrantes de la imagen 1
        q1 = img1[0:mitad_h, 0:mitad_w]
        q4 = img1[mitad_h:h, mitad_w:w]

        # Insertar q1 y q4 en img2
        img2[0:mitad_h, 0:mitad_w] = q1
        img2[mitad_h:h, mitad_w:w] = q4

        cv2.imshow("Rompecabezas", img2)
        cv2.imwrite("rompecabezas.jpg", img2)
        print("Rompecabezas guardado como rompecabezas.jpg")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error al leer las imágenes guardadas.")
else:
    print("No se generó rompecabezas: asegúrate de guardar foto_bn.jpg y foto_gris.jpg")
