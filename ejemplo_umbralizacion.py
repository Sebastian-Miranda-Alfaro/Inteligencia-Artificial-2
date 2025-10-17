# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 13:23:48 2025

@author: sebas
"""

#============================UMBRALIZACION=========================#
import cv2
import numpy as np 
import matplotlib.pyplot as plt

#leer imagen
img = cv2.imread("bluey.jpg")

#Convertir a gris
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

matriz = np.ones(gris.shape, dtype='uint8')*50

#=========IMAGEN BRILLANTE==================#
#Aumentar el brillo 
brillanteGray = cv2.add(gris, matriz)

#Usando threshold
_, imgthres1 = cv2.threshold(brillanteGray, 160, 255, cv2.THRESH_BINARY)
_, imgthres2 = cv2.threshold(brillanteGray, 160, 255, cv2.THRESH_BINARY_INV)

#USANDO EL ADAPTATIVO
imaadaptative = cv2.adaptiveThreshold(brillanteGray, 255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 11, 7)

#==============IMAGEN OSCURA====================#
oscuraGray = cv2.subtract(gris, matriz)

_, imgthres3 = cv2.threshold(oscuraGray, 50, 255, cv2.THRESH_BINARY)
_, imgthres4 = cv2.threshold(oscuraGray, 50, 255, cv2.THRESH_BINARY_INV)

#==================USANDO EL ADAPTATIVO========================#
imaadaptative2 = cv2.adaptiveThreshold(oscuraGray, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 11, 7)

#===================MOSTRAR RESULTADOS========================#
fig = plt.figure(figsize=(12,8))

# Original y gris
ax = fig.add_subplot(3,4,1)
ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax.set_title("Original")
ax.axis("off")

ax = fig.add_subplot(3,4,2)
ax.imshow(gris, cmap="gray")
ax.set_title("Gris")
ax.axis("off")

# Brillante y umbrales
ax = fig.add_subplot(3,4,3)
ax.imshow(brillanteGray, cmap="gray")
ax.set_title("Brillante")
ax.axis("off")

ax = fig.add_subplot(3,4,4)
ax.imshow(imgthres1, cmap="gray")
ax.set_title("Brillante THRESH_BINARY")
ax.axis("off")

ax = fig.add_subplot(3,4,5)
ax.imshow(imgthres2, cmap="gray")
ax.set_title("Brillante THRESH_BINARY_INV")
ax.axis("off")

ax = fig.add_subplot(3,4,6)
ax.imshow(imaadaptative, cmap="gray")
ax.set_title("Brillante Adaptativo")
ax.axis("off")

# Oscura y umbrales
ax = fig.add_subplot(3,4,7)
ax.imshow(oscuraGray, cmap="gray")
ax.set_title("Oscura")
ax.axis("off")

ax = fig.add_subplot(3,4,8)
ax.imshow(imgthres3, cmap="gray")
ax.set_title("Oscura THRESH_BINARY")
ax.axis("off")

ax = fig.add_subplot(3,4,9)
ax.imshow(imgthres4, cmap="gray")
ax.set_title("Oscura THRESH_BINARY_INV")
ax.axis("off")

ax = fig.add_subplot(3,4,10)
ax.imshow(imaadaptative2, cmap="gray")
ax.set_title("Oscura Adaptativo")
ax.axis("off")

plt.tight_layout()
plt.show()
