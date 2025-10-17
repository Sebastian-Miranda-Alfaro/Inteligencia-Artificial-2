# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 08:00:33 2025

@author: sebas
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

#================== LEER IMAGEN ==================#
imagen = cv2.imread("fotoex.jpg")
rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

#=============Imagen para sacar coordenadas================#
plt.imshow(rgb)
plt.title("Imagen completa")
plt.axis("on")
plt.show()

#================== RECORTE DE OJOS ==================#
# (y1:y2, x1:x2)
ojos = rgb[214:276, 609:802]

#================== AGREGAR RUIDO SAL Y PIMIENTA ==================#
def ruido_sal_pimienta(img, prob):
    salida = np.copy(img)
    rnd = np.random.rand(img.shape[0], img.shape[1])
    
    # píxeles negros (pimienta)
    salida[rnd < (prob/2)] = [0, 0, 0]
    
    # píxeles blancos (sal)
    salida[rnd > 1 - (prob/2)] = [255, 255, 255]

    return salida

ojos_ruido = ruido_sal_pimienta(ojos, 0.1)

#================== FILTRO DE MEDIANA ==================#
ojos_mediana = cv2.medianBlur(ojos_ruido, 3)

#================== MOSTRAR RESULTADOS ==================#
fig = plt.figure(figsize=(10,5))

ax1 = fig.add_subplot(1,2,1)
ax1.imshow(ojos_ruido)
ax1.set_title("Ojos con ruido S y P")
ax1.axis("off")

ax2 = fig.add_subplot(1,2,2)
ax2.imshow(ojos_mediana)
ax2.set_title("Filtro de mediana")
ax2.axis("off")

plt.show()
