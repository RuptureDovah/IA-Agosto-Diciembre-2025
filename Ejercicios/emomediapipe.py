import cv2
import mediapipe as mp
import math

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Inicializar dibujador de MediaPipe
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(234, 255, 233))  # Puntos verdes

# Índices de puntos faciales importantes para emociones
LIPS_TOP = 13        # Labio superior
LIPS_BOTTOM = 14     # Labio inferior
LEFT_EYEBROW_TOP = 65    # Ceja izquierda arriba
RIGHT_EYEBROW_TOP = 295  # Ceja derecha arriba
LEFT_EYE_TOP = 159       # Ojo izquierdo arriba
LEFT_EYE_BOTTOM = 145    # Ojo izquierdo abajo
RIGHT_EYE_TOP = 386      # Ojo derecho arriba
RIGHT_EYE_BOTTOM = 374   # Ojo derecho abajo

def calcular_distancia(punto1, punto2):
    """Calcula la distancia euclidiana entre dos puntos"""
    return math.sqrt((punto1.x - punto2.x)**2 + (punto1.y - punto2.y)**2)

def detectar_emocion(face_landmarks):
    """Detecta la emoción basándose en las posiciones de los puntos faciales"""
    
    # Obtener puntos relevantes
    labio_superior = face_landmarks.landmark[LIPS_TOP]
    labio_inferior = face_landmarks.landmark[LIPS_BOTTOM]
    ceja_izq = face_landmarks.landmark[LEFT_EYEBROW_TOP]
    ceja_der = face_landmarks.landmark[RIGHT_EYEBROW_TOP]
    ojo_izq_arriba = face_landmarks.landmark[LEFT_EYE_TOP]
    ojo_izq_abajo = face_landmarks.landmark[LEFT_EYE_BOTTOM]
    ojo_der_arriba = face_landmarks.landmark[RIGHT_EYE_TOP]
    ojo_der_abajo = face_landmarks.landmark[RIGHT_EYE_BOTTOM]
    
    # Calcular medidas
    apertura_boca = calcular_distancia(labio_superior, labio_inferior)
    apertura_ojo_izq = calcular_distancia(ojo_izq_arriba, ojo_izq_abajo)
    apertura_ojo_der = calcular_distancia(ojo_der_arriba, ojo_der_abajo)
    altura_ceja_izq = ceja_izq.y
    altura_ceja_der = ceja_der.y
    
    # Umbrales (pueden necesitar ajuste según la cámara y distancia)
    UMBRAL_BOCA_FELIZ = 0.08
    UMBRAL_BOCA_TRISTE = 0.03
    UMBRAL_OJOS_SERIO = 0.02
    UMBRAL_CEJAS_ENOJADO = 0.35
    
    # Detectar emociones
    if apertura_boca > UMBRAL_BOCA_FELIZ:
        return "FELIZ", (0, 255, 0)  # Verde
    elif apertura_boca < UMBRAL_BOCA_TRISTE and (altura_ceja_izq > UMBRAL_CEJAS_ENOJADO or altura_ceja_der > UMBRAL_CEJAS_ENOJADO):
        return "TRISTE", (255, 255, 0)  # Amarillo
    elif altura_ceja_izq < UMBRAL_CEJAS_ENOJADO and altura_ceja_der < UMBRAL_CEJAS_ENOJADO and apertura_ojo_izq < UMBRAL_OJOS_SERIO and apertura_ojo_der < UMBRAL_OJOS_SERIO:
        return "ENOJADO", (0, 0, 255)  # Rojo
    elif apertura_ojo_izq < UMBRAL_OJOS_SERIO and apertura_ojo_der < UMBRAL_OJOS_SERIO:
        return "SERIO", (255, 0, 0)  # Azul
    else:
        return "NEUTRO", (255, 255, 255)  # Blanco

# Captura de video
cap = cv2.VideoCapture(0)

dato = True

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Espejo para mayor naturalidad
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, drawing_spec, drawing_spec)
            
            # Detectar emoción
            emocion, color = detectar_emocion(face_landmarks)
            
            # Mostrar emoción en pantalla
            cv2.putText(frame, emocion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
            if dato:
                for idx, face_ptx in enumerate(face_landmarks.landmark):
                    print(f'Punto {idx}: (x: {face_ptx.x}, y: {face_ptx.y}, z: {face_ptx.z})')
                dato = False

    cv2.imshow('Deteccion de Emociones', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()