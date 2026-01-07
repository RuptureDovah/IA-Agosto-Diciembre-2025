import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from skimage.transform import resize
from skimage.io import imread

# Librerías para abrir la ventana de selección de archivos
import tkinter as tk
from tkinter import filedialog

# --- CONFIGURACIÓN ---
MODELO = 'modelo_animales.h5'
ANCHO = 64
ALTO = 64
# Clases en orden
CLASES = ['hormiga', 'gato', 'perro', 'mariquita', 'tortuga'] 

# --- 1. CARGAR MODELO ---
print(f"Cargando cerebro ({MODELO})...")
try:
    model = load_model(MODELO)
    print("Modelo cargado y listo para trabajar.")
except:
    print(f"ERROR: No encuentro el archivo '{MODELO}'. Asegúrate de estar en la carpeta correcta.")
    exit()

# Preparamos la ventanita oculta de Tkinter
root = tk.Tk()
root.withdraw() # Ocultamos la ventana principal para solo mostrar el explorador de archivos

# --- 2. BUCLE INFINITO DE PRUEBAS ---
while True:
    print("\n" + "="*50)
    print("Abriendo selector de archivos... (Dale a 'Cancelar' para salir)")
    print("="*50)

    # Abre la ventana para seleccionar archivos (permite seleccionar varios a la vez)
    rutas_archivos = filedialog.askopenfilenames(
        title="Selecciona las fotos de animales para analizar",
        filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Todos los archivos", "*.*")]
    )

    # Si el usuario da "Cancelar" o no selecciona nada, rompemos el ciclo
    if not rutas_archivos:
        print("¡Prueba finalizada por el usuario!")
        break

    # Procesar cada imagen seleccionada
    for ruta_completa in rutas_archivos:
        nombre_archivo = os.path.basename(ruta_completa)
        
        try:
            # A. Leer imagen
            img = imread(ruta_completa)
            
            # B. Limpiar canales extra (si es PNG transparente)
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = img[:, :, :3]

            # C. Redimensionar a 64x64
            img_resized = resize(img, (ALTO, ANCHO), anti_aliasing=True, preserve_range=True)
            
            # D. Preparar para la IA (Normalizar y Lote)
            img_input = np.array([img_resized], dtype=np.float32) / 255.0

            # E. PREDECIR
            prediccion = model.predict(img_input, verbose=0)
            idx = np.argmax(prediccion) 
            animal_detectado = CLASES[idx]
            confianza = np.max(prediccion) * 100

            # F. MOSTRAR RESULTADO EN TEXTO
            print(f"Foto: {nombre_archivo}")
            print(f"Resultado: {animal_detectado.upper()} (Seguridad: {confianza:.1f}%)")
            print("-" * 30)

            # G. MOSTRAR RESULTADO VISUAL
            # Esto pausará el código hasta que cierres la ventanita de la foto
            plt.figure(figsize=(4, 4))
            plt.imshow(img)
            
            # Color del título: Verde si está muy seguro, Rojo si duda
            color_texto = 'green' if confianza > 70 else 'red'
            
            plt.title(f"Es un: {animal_detectado}\n({confianza:.1f}%)", color=color_texto, fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.show() 

        except Exception as e:
            print(f"Error procesando {nombre_archivo}: {e}")

    print("Lote terminado. ¿Quieres probar más?")