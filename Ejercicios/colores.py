import cv2 as cv



def encontrar_y_dibujar_centroides(imagen, mascara, color_bgr):
    contornos, _ = cv.findContours(mascara, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for c in contornos:

        # Calcular los momentos y el centroide del contorno
        M = cv.moments(c)
        if M["m00"] != 0:
            centro_x = int(M["m10"] / M["m00"])
            centro_y = int(M["m01"] / M["m00"])
        else:
            centro_x, centro_y = 0, 0

        
        # Dibujar un pequeño círculo negro en el centro para que resalte
        cv.circle(imagen, (centro_x, centro_y), 7, (0, 0, 0), -1)
        
        # Escribir el nombre del color detectado
        # (Esto es un extra para que se vea mejor)
        nombre_color = ""
        if color_bgr == (0, 0, 255): nombre_color = "Rojo"
        elif color_bgr == (0, 255, 0): nombre_color = "Verde"
        elif color_bgr == (255, 0, 0): nombre_color = "Azul"
        elif color_bgr == (0, 255, 255): nombre_color = "Amarillo"
        
        cv.putText(imagen, nombre_color, (centro_x - 30, centro_y - 15), cv.QT_FONT_NORMAL, 0.5, (255, 255, 255), 2)







img = cv.imread("C:/Users/gabri/Documents/9no/IA/figura.png")
img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

ubbR = (0, 60, 60)
ubaR = (10, 255, 255)
ubb1R = (170, 60, 60)
uba1R = (180, 255, 255)

mascara1R = cv.inRange(hsv, ubbR, ubaR)
mascara2R = cv.inRange(hsv, ubb1R, uba1R)

mascaraR = mascara1R + mascara2R
resultadoR = cv.bitwise_and(img, img, mask=mascaraR)


ubbG = (35, 60, 60)
ubaG = (85, 255, 255)

mascaraG = cv.inRange(hsv, ubbG, ubaG)
resultadoG = cv.bitwise_and(img, img, mask=mascaraG)


ubbB = (90, 60, 60)
ubaB = (130, 255, 255)

mascaraB = cv.inRange(hsv, ubbB, ubaB)
resultadoB = cv.bitwise_and(img, img, mask=mascaraB)

ubbY = (20, 60, 60)
ubaY = (35, 255, 255)

mascaraY = cv.inRange(hsv, ubbY, ubaY)
resultadoY = cv.bitwise_and(img, img, mask=mascaraY)

encontrar_y_dibujar_centroides(resultadoR, mascaraR, (0, 0, 255))
encontrar_y_dibujar_centroides(resultadoG, mascaraG, (0, 255, 0))
encontrar_y_dibujar_centroides(resultadoB, mascaraB, (255, 0, 0))
encontrar_y_dibujar_centroides(resultadoY, mascaraY, (0, 255, 255))

cv.imshow('resultadoR', resultadoR)
cv.imshow('resultadoG', resultadoG)
cv.imshow('resultadoB', resultadoB)
cv.imshow('resultadoY', resultadoY)

cv.imshow('img', img)
#cv.imshow('img2', img2)
#cv.imshow('hsv', hsv)

cv.waitKey(0)
cv.destroyAllWindows()