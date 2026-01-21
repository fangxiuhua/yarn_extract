import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import mask as mk

def main():
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
            continue

        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray,(5,5),0)

        # 生成二值图
        _,binary = cv.threshold(blur,127,255,cv.THRESH_BINARY)

        # RETR_CCOMP 提取所有轮廓，建立两层层级
        contours , hierarchy = cv.findContours(binary,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)

        result, mask = mk.extract_ring(img, binary)

        plt.figure(figsize=(6,6))
        plt.imshow(cv.cvtColor(result,cv.COLOR_BGR2RGB))
        plt.title(img_path)
        plt.axis("off")
        plt.show()
        plt.close()

main()