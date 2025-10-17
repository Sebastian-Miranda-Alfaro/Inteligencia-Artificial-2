import numpy as np 
import matplotlib.pyplot as plt 
import cv2

#leer imagen
ima = cv2.imread("imagenedu.jpg")

#Mostrar la imagen original en BGR
cv2.imshow("Imagen original (BGR)", ima)

#Convertir la imagen de BGR a RGB
rgb = cv2.cvtColor(ima, cv2.COLOR_BGR2RGB)
cv2.imshow("Imagen RGB", rgb)

#Convertir la imagen a escala de grises
gris = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
cv2.imshow("Imagen gris", gris)

#Crear una imagen con una matriz
img2 = 50 * np.ones((729, 759, 3), dtype=np.uint8)
plt.imshow(img2)

#Operaciones matriz con imagen 
#Subir el brillo
suma = rgb + img2

#Disminuir el brillo
resta = rgb - img2

#Multiplicacion
multi = rgb * img2

#Divicion
div = rgb / img2

fig=plt.figure()
#suma
ax1=fig.add_subplot(2,2,1)
ax1.imshow(suma)
plt.title("Suma")
#resta
ax1=fig.add_subplot(2,2,2)
ax1.imshow(resta)
plt.title("Resta")
#multi
ax1=fig.add_subplot(2,2,3)
ax1.imshow(multi)
plt.title("Multi")
#Div
ax1=fig.add_subplot(2,2,4)
ax1.imshow(div)
plt.title("Division")