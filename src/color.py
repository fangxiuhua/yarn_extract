import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img_colors = [
    'picture/purple-ge (167).jpg',
    'picture/red (42).jpg'
]

for img_color in img_colors:
    img = cv.imread(img_color)

    if img is None:
        print(f"图像错误{img_color}")
        continue

    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(5,5),0)

    _,binary = cv.threshold(blur,127,255,cv.THRESH_BINARY)

    contours , _ = cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    img_contour = cv.drawContours(img.copy(),contours,-1,(0,255,0),2)

    areas = []
    for c in contours:
        areas.append(cv.contourArea(c))
    max_contour = contours[np.argmax(areas)]

    (x,y),radius = cv.minEnclosingCircle(max_contour)

    center = (int(x),int(y))
    radius = int(radius)
    color_circle = (255,0,0)
    color_point = (0,0,255)

    # 圆的轮廓
    cv.circle(img_contour,center,radius,color_circle,10)

    # 圆心的位置
    cv.circle(img_contour,center,10,color_point,-1)

    plt.figure(figsize=(6,6))
    plt.imshow(cv.cvtColor(img_contour,cv.COLOR_BGR2RGB))
    plt.title(img_color)
    plt.axis("off")
    plt.show()
    plt.close()