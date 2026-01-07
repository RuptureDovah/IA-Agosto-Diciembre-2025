import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
while(True):
    res, img = cap.read()
    if res:
        cv.imshow('Fotograma', img)
        img2 = np.zeros(img.shape[:2], np.uint8)
        b,g,r=cv.split(img)
        b1=cv.merge([b, img2, img2])
        g1=cv.merge([img2, g, img2])
        r1=cv.merge([img2, img2, r])
        res1 = cv.merge([g,r,b])
        cv.imshow('res1', res1)
        cv.imshow('b', b)
        cv.imshow('g', g)
        cv.imshow('r', r)
        cv.imshow('b1', b1)
        cv.imshow('g1', g1)
        cv.imshow('r1', r1)
        cv.imshow('marco2', img2)
        cv.imshow('marco', img)

        k = cv.waitKey(1)
        if k == 27:
            break
cap.release()
cv.destroyAllWindows()