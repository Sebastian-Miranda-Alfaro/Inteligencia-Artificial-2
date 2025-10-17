# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 14:07:31 2025

@author: sebas
"""

#===================TOMAR UNA FOTO CON LA CAMARA========================#
import cv2
#import os 

#Crear video
cap=cv2.VideoCapture(0)

#Tomar una sola foto
ret, frame=cap.read()

if ret:
    cv2.imwrite("foto.png",frame)
    print("foto tomada correctamente")
else:
    print("Error al acceder a la camara")
    
#liberar la camara
cap.release()

fotoTomada=cv2.imread("foto.png")
cv2.imshow("foto tomada", fotoTomada)