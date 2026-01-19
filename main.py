import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('picture/black (171).jpg')
cv.imshow('img',img)

if img is None:
    raise ValueError("图像读取错误")

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray,(5,5),0)

_,binary = cv.threshold(blur,127,255,cv.THRESH_BINARY)

contours , _ = cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

img_contour = cv.drawContours(img.copy(),contours,-1,(0,255,0),2)

plt.figure(figsize=(12, 6))
plt.subplot(131), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)), plt.title("original image")
plt.subplot(132), plt.imshow(binary, cmap="gray"), plt.title("binary")
plt.subplot(133), plt.imshow(cv.cvtColor(img_contour, cv.COLOR_BGR2RGB)), plt.title("contour")
plt.axis("off"), plt.tight_layout(), plt.show()
