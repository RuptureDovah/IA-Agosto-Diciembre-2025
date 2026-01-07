import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
from tensorflow.keras.models import load_model

# --- CONFIGURACIÓN ---
MODELO_PATH = 'modelo_animales.h5'
DATASET_PATH = './dataset' # Usamos la misma carpeta para sacar ejemplos
ANCHO = 64 # DEBE SER 64 (Igual que el entrenamiento)
ALTO = 64

# 1. CARGAR MODELO
if not os.path.exists(MODELO_PATH):
    print(f"❌ Error: No encuentro el archivo {MODELO_PATH}")
    exit()

print(f"Cargando modelo: {MODELO_PATH}...")
riesgo_model = load_model(MODELO_PATH)

# 2. SELECCIONAR IMÁGENES DE PRUEBA
# Vamos a buscar una imagen real de cada carpeta para probar
imagenes_prueba = []
nombres_archivos = []
etiquetas_reales = []

# Definir las clases en ORDEN ALFABÉTICO 
# Si se guardó el archivo 'clases.npy' lo ideal es cargarlo, si no, lo definimos manual:
clases = ['ant', 'cat', 'dog', 'ladybug', 'turtle']

print("Preparando imágenes de prueba...")

# Buscamos una imagen de ejemplo dentro de tu carpeta dataset para cada animal
for clase in clases:
    ruta_clase = os.path.join(DATASET_PATH, clase)
    if os.path.exists(ruta_clase):
        # Tomar la primera imagen que encuentre en esa carpeta
        archivos = [f for f in os.listdir(ruta_clase) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if archivos:
            ruta_img = os.path.join(ruta_clase, archivos[0]) # Tomamos la primera
            
            try:
                # Cargar y preprocesar igual que en el entrenamiento
                img = imread(ruta_img)
                if len(img.shape) == 3 and img.shape[2] == 4: img = img[:,:,:3] # Quitar transparencia
                img_resized = resize(img, (ALTO, ANCHO), anti_aliasing=True, preserve_range=True)
                
                imagenes_prueba.append(img_resized)
                nombres_archivos.append(archivos[0])
                etiquetas_reales.append(clase)
            except Exception as e:
                print(f"Error cargando {ruta_img}: {e}")

# Convertir a numpy array y normalizar
X_test = np.array(imagenes_prueba, dtype=np.float32) / 255.0

# 3. PREDICCIÓN
print("La IA está pensando...")
predicciones = riesgo_model.predict(X_test)

# 4. RESULTADOS
ncorrect = 0
print("\n" + "="*40)
print("RESULTADOS DE LA PRUEBA")
print("="*40)

for i, pred in enumerate(predicciones):
    # Obtener el índice con mayor probabilidad
    indice_predicho = np.argmax(pred)
    clase_predicha = clases[indice_predicho]
    confianza = np.max(pred) * 100
    
    real = etiquetas_reales[i]
    
    if real == clase_predicha:
        icono = "✅"
        ncorrect += 1
    else:
        icono = "❌"
    
    print(f"{icono} Real: {real.ljust(10)} | Predicción: {clase_predicha.ljust(10)} ({confianza:.1f}%)")

# 5. ESTADÍSTICAS FINALES
total = len(imagenes_prueba)
if total > 0:
    print("-" * 40)
    print(f"Aciertos: {ncorrect} de {total}")
    print(f"Precisión: {ncorrect / total:.1%}")
else:
    print("No se encontraron imágenes para probar.")