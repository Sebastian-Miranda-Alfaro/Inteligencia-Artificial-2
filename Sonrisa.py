import cv2

# Clasificadores de Haar
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smileClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# Capturar video desde la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros
    faces = faceClassif.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detectar sonrisas dentro del rostro
        smiles = smileClassif.detectMultiScale(
            roi_gray,
            scaleFactor=1.7,
            minNeighbors=22,
            minSize=(25, 25)
        )

        for (sx, sy, sw, sh) in smiles:
            # Extraer la región de la sonrisa en escala de grises
            smile_region = roi_gray[sy:sy+sh, sx:sx+sw]

            # Convertir a binario (blanco y negro)
            _, smile_binary = cv2.threshold(smile_region, 128, 255, cv2.THRESH_BINARY)

            # Convertir a BGR para poder pegarlo de nuevo en el frame a color
            smile_binary_bgr = cv2.cvtColor(smile_binary, cv2.COLOR_GRAY2BGR)

            # Reemplazar la región de la sonrisa en la imagen original
            roi_color[sy:sy+sh, sx:sx+sw] = smile_binary_bgr

    cv2.imshow("Deteccion de Sonrisas", frame)

    # Salir con tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
