# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 07:47:53 2025

@author: sebas
"""

import cv2
import numpy as np 
import  matplotlib.pyplot as plt

ima=cv2.imread("imagen2.jpg",0)
cv2.imshow("original",ima)

ima.shape

#Generar histogramas
hist=cv2.calcHist([ima],[0],None,[256],[0,256])
plt.plot(hist)
#Otra forma de obtener el histograma
plt.hist(ima.ravel(),256,[0,256])
plt.show()

ima2=cv2.imread("imagen2.jpg")

#Crear una figura para mostrarla
fig, ax=plt.subplots(2,2)
ax[0,0].imshow(ima2,cmap="gray")
ax[0,0].set_title("original")
ax[0,0].axis('off')

ax[0,1].plot(hist)
ax[0,1].set_title("Histograma")

ax[1,0].imshow(ima2,cmap="gray")
ax[1,0].set_title("original")
ax[1,0].axis('off')

ax[1,1].hist(ima.ravel(),256,[0,256])
ax[1,1].set_title("Histograma")

#Operaciones con imágenes
img=cv2.imread("imagen2.jpg")
rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
red1=cv2.resize(rgb,None,fx=2,fy=2)
cv2.imshow("Original",red1)
suma=red1+5
cv2.imshow("suma",suma)

resta=red1-5
cv2.imshow("resta",resta)

multi=red1*2
cv2.imshow("multiplicacion",multi)

Division=red1//2
cv2.imshow("division",Division)

#Recorte ojos
ima3=cv2.imread("imagen2.jpg")
recorte = ima3[135-20:135+20, 420:470]
plt.imshow(recorte,cmap='gray')
plt.axis('off')
plt.show()


suma = recorte + 50
cv2.imshow("Suma", suma)

resta = recorte - 50 
cv2.imshow("Resta", resta)

multi = recorte * 2   
cv2.imshow("Multiplicacion", multi)

division = recorte // 2   
cv2.imshow("Division", division)

hist1=cv2.calcHist([ima3],[0],None,[256],[0,256])
plt.plot(hist1)
