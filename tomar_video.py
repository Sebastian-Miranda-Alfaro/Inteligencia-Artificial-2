# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 22:28:16 2025

@author: sebas
"""

import cv2

cap = cv2.VideoCapture(0)

# Propiedades del video
ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20

# Crear VideoWriter para MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter("video_output.mp4", fourcc, fps, (ancho, alto))

# Contador de frames
num_frames = 0
max_frames = 200

while num_frames < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Escribir el frame en el archivo de video
    out.write(frame)
    
    # Mostrar el video en tiempo real
    cv2.imshow("Grabando...", frame)
    
    num_frames += 1

    # Letra para interrumpir la condicion
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar todo
cap.release()

print("Video guardado como video_output.mp4")
