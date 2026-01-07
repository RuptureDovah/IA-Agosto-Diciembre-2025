import cv2 as cv
import os
from deepface import DeepFace

# 1. Carga tu reconocedor de rostros y la lista de nombres
faceRecognizer = cv.face.FisherFaceRecognizer_create()
faceRecognizer.read('FisherFace.xml')
faces = ['carlos', 'gabs', 'solrac'] # Asegúrate que el orden es el mismo con el que entrenaste

cap = cv.VideoCapture(0)
rostro = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Crea una copia en escala de grises para el reconocimiento facial
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    rostros = rostro.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in rostros:
        # --- Parte 1: Reconocimiento Facial (con FisherFace) ---
        # Recorta el rostro de la imagen en escala de grises
        cara_gris = gray[y:y+h, x:x+w]
        cara_gris = cv.resize(cara_gris, (100, 100), interpolation=cv.INTER_CUBIC)
        
        # Realiza la predicción para saber quién es
        result = faceRecognizer.predict(cara_gris)
        
        # --- Parte 2: Detección de Emociones (con DeepFace) ---
        # Recorta el rostro de la imagen a color
        cara_color = frame[y:y+h, x:x+w]
        
        try:
            # Analiza la emoción del rostro
            resultado_emocion = DeepFace.analyze(cara_color, actions=['emotion'], enforce_detection=False)
            emocion_dominante = resultado_emocion[0]['dominant_emotion']
            
            # Diccionario para traducir emociones
            emociones_es = {
                'angry': 'Enojo', 'disgust': 'Disgusto', 'fear': 'Miedo',
                'happy': 'Feliz', 'sad': 'Triste', 'surprise': 'Sorpresa', 'neutral': 'Neutral'
            }
            emocion_traducida = emociones_es.get(emocion_dominante, emocion_dominante)
            
            # Muestra la emoción en la pantalla
            cv.putText(frame, emocion_traducida, (x, y - 40), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            
        except Exception as e:
            # Si no detecta emoción, no muestra nada
            pass

        # --- Parte 3: Mostrar los Resultados Combinados ---
        # Comprueba la confianza de la predicción de reconocimiento
        if result[1] < 500: # Ajusta este umbral si es necesario
            nombre = faces[result[0]]
            color_rect = (0, 255, 0) # Verde para conocido
        else:
            nombre = "Desconocido"
            color_rect = (0, 0, 255) # Rojo para desconocido

        # Dibuja el rectángulo y el nombre
        cv.rectangle(frame, (x, y), (x+w, y+h), color_rect, 2)
        cv.putText(frame, nombre, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, color_rect, 2)

    cv.imshow('frame', frame)
    k = cv.waitKey(1)
    if k == 27:
        break
        
cap.release()
cv.destroyAllWindows()