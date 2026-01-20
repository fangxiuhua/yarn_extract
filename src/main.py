import cv2 as cv
import matplotlib.pyplot as plt

imgs_paths = [
        'picture/black (171).jpg',
        'picture/black-cha (72).jpg',
        'picture/black-dian (127).jpg',
        'picture/black-tiao (192).jpg',
        'picture/Pic_20241016123223968.jpg',
        'picture/black-xing (154).jpg'
    ]

for img_path in imgs_paths:
    img = cv.imread(img_path)

    if img is None:
        print(f"图像错误{img_path}")

    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(5,5),0)

    _,binary = cv.threshold(blur,127,255,cv.THRESH_BINARY)

    # RETR_CCOMP 提取所有轮廓，建立两层层级
    # RETR_EXTERNAL：仅提取最外层轮廓
    contours , _ = cv.findContours(binary,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)

    img_contour = cv.drawContours(img.copy(),contours,-1,(0,255,0),2)

    plt.figure(figsize=(12, 6))
    plt.suptitle(img_path)
    plt.subplot(131), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)), plt.title("original image")
    plt.subplot(132), plt.imshow(binary, cmap="gray"), plt.title("binary")
    plt.subplot(133), plt.imshow(cv.cvtColor(img_contour, cv.COLOR_BGR2RGB)), plt.title("contour")
    plt.axis("off"), plt.tight_layout(), plt.show(),plt.close()
