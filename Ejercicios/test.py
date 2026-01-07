import cv2 as cv
import numpy as np 

# Cargar la imagen
img = cv.imread("C:/Users/gabri/Documents/9no/IA/saber.jpg")

# Verificar que la imagen se cargó correctamente
if img is None:
    print("Error: No se pudo cargar la imagen. Revisa la ruta.")
    exit()

# Crear una matriz vacía del mismo tamaño que la imagen, en escala de grises
img2 = np.zeros(img.shape[:2], np.uint8)

# Separar canales
b, g, r = cv.split(img)

# Canales individuales
b1 = cv.merge([b, img2, img2])
g1 = cv.merge([img2, g, img2])
r1 = cv.merge([img2, img2, r])

# Reordenar canales (ejemplo: g, r, b)
res1 = cv.merge([g, r, b])

# Mostrar resultados
cv.imshow('res1', res1)
cv.imshow('b', b)
cv.imshow('g', g)
cv.imshow('r', r)
cv.imshow('b1', b1)
cv.imshow('g1', g1)
cv.imshow('r1', r1)
cv.imshow('marco2', img2)
cv.imshow('marco', img)

cv.waitKey(0)
cv.destroyAllWindows()
