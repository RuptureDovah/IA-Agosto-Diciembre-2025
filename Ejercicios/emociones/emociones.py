import sys
print("Iniciando script...")
print(f"Usando Python en: {sys.executable}") # Para verificar que usas el .venv

try:
    import cv2
    print("Importado cv2 (OpenCV) exitosamente.")
    import mediapipe as mp
    print("Importado mediapipe exitosamente.")
    from deepface import DeepFace
    print("Importado deepface exitosamente.")
except ImportError as e:
    print(f"--- ERROR DE IMPORTACION ---")
    print(f"Fallo al importar: {e}")
    print("El script se detendra. Revisa tu instalacion.")
    print("Si el error es de tensorflow, protobuf o mediapipe,")
    print("intenta el 'Plan B' (entorno limpio) del mensaje anterior.")
    exit() # Detiene el script si las importaciones fallan

# --- Tu código de inicialización de MediaPipe (sin cambios) ---
print("Inicializando MediaPipe...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(
    thickness=1, 
    circle_radius=1, 
    color=(255, 255, 255)
)
PUNTOS_ESPECIFICOS = [1, 152, 133, 33, 362, 263, 61, 291, 13, 14, 53, 65, 55, 283, 295, 285, 159, 145, 386, 374]
print("MediaPipe inicializado.")
# -----------------------------------------------------------

# Diccionario para traducir las emociones
traduccion_emociones = {
    'happy': 'Feliz', 'sad': 'Triste', 'angry': 'Enojado',
    'neutral': 'Neutral', 'fear': 'Miedo', 'disgust': 'Asco',
    'surprise': 'Sorpresa'
}

# Captura de video
print("Intentando abrir camara (index 0)...")
cap = cv2.VideoCapture(0)

# --- ¡VERIFICACION IMPORTANTE! ---
if not cap.isOpened():
    print("---------------------------------------------------------")
    print("¡ERROR CRITICO! No se pudo abrir la camara (cap.isOpened() es Falso).")
else:
    print("¡Camara abierta exitosamente! Entrando al bucle principal...")

emocion_detectada = "Calculando..."
frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error al leer frame de la camara. Saliendo del bucle.")
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if frame_counter % 10 == 0:
        try:
            analisis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emocion_en_ingles = analisis[0]['dominant_emotion']
            emocion_detectada = traduccion_emociones.get(emocion_en_ingles, emocion_en_ingles)
        except Exception as e:
            emocion_detectada = "No detectada"
    
    frame_counter += 1

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx in PUNTOS_ESPECIFICOS:
                lm = face_landmarks.landmark[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), cv2.FILLED)

    cv2.putText(
        frame, 
        emocion_detectada, 
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imshow('PuntosFacialesMediaPipe', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Tecla 'q' presionada. Saliendo...")
        break

print("Bucle terminado.")
cap.release()
cv2.destroyAllWindows()
print("Recursos liberados. Script finalizado.")
