import numpy as np
import matplotlib.pyplot as plt
import cv2

ima = cv2.imread("imagen.jpg")

#comando reservado para extraer los canales rgb
R,G,B = cv2.split(ima)
#crear una imagen negra
black = np.zeros(ima.shape[:2], dtype="uint8")

#combinar imagen negra con los canales rgb
#canal rojo
rojo = cv2.merge([black, black, R])

#canal verde
verde = cv2.merge([black, G, black])

#canal azul
azul = cv2.merge([B, black, black])

#Mostrar imagenes 

#Canal rojo
Rojo=cv2.merge([black,black,R])
#cv2.imshow('Rojo',Rojo)

#Canal verde
Verde=cv2.merge([black,G,black])
#cv2.imshow('Verde',Verde)

#Canal azul
Azul=cv2.merge([B,black,black])
#cv2.imshow('Azul',Azul)

#Recortes de una imagen
tam = ima.shape
print(tam)

#Canal 1, sacar el canal rojo
c1=ima[0:63,0:63,:]
R,G,B=cv2.split(c1)
black=np.zeros(R.shape[:2],dtype="uint8")
Rojo=cv2.merge([black,black,R])
cv2.imshow("Rojo_recorte",Rojo)

#Canal verde del segundo cuadrante
c2=ima[0:63,63:126,:]
R,G,B=cv2.split(c2)
black=np.zeros(G.shape[:2],dtype="uint8")
Verde=cv2.merge([black,G,black])
cv2.imshow("Verde_recorte",Verde)

#Canal azul del cuadrante 3
c3=ima[63:126,0:63,:]
R,G,B=cv2.split(c3)
black=np.zeros(B.shape[:2],dtype="uint8")
Azul=cv2.merge([R,black,black])
cv2.imshow("Azul_recorte",Azul)

#Gris para cuadrante 4
c4=ima[63:126,63:126,:]
gris=cv2.cvtColor(c4,cv2.COLOR_BGR2GRAY)
cv2.imshow("Recorte_gris",gris)

fig=plt.figure()
#canal rojo
ax1=fig.add_subplot(2,2,1)
ax1.imshow(Rojo)
#Canal verde
ax1=fig.add_subplot(2,2,2)
ax1.imshow(Verde)
#Canal azul
ax1=fig.add_subplot(2,2,3)
ax1.imshow(Azul)
#Gris
ax1=fig.add_subplot(2,2,4)
ax1.imshow(gris,cmap="gray")

#rompecabezas con los canales y el 
ima2=ima
ima2[0:63,0:63,0]=gris
ima2[0:63,0:63,1]=gris
ima2[0:63,0:63,2]=gris
ima2[0:63,63:126,:]=Azul
ima2[63:126,0:63,:]=Verde
ima2[63:126,63:126,:]=Rojo
cv2.imshow("rompecabezas",ima2)