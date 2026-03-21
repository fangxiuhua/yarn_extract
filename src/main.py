import cv2 as cv
import matplotlib.pyplot as plt
import mask as mk

def main():
    # 载入图像
    imgs_paths = [
            'picture/black (171).jpg',
            'picture/black-cha (72).jpg',
            'picture/black-dian (127).jpg',
            'picture/black-tiao (192).jpg',
            'picture/Pic_20241016123223968.jpg',
            'picture/black-xing (154).jpg',
            'picture/purple-ge (167).jpg',
            'picture/red (42).jpg'
        ]

    for img_path in imgs_paths:
        img = cv.imread(img_path)
        if img is None:
            print(f"图像错误{img_path}")
            continue
        # 压缩图像
        h, w = img.shape[:2]
        scale = 640 / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_re = cv.resize(img, (new_w, new_h))

        # 灰度处理和高斯滤波
        gray = cv.cvtColor(img_re,cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray,(5,5),0)
        # 生成二值图
        _,binary = cv.threshold(blur,127,255,cv.THRESH_BINARY)

        # 识别纱线环区域
        yarn = mk.detect_yarn(binary)
        if yarn is None:
            print("未读取到类")
            continue 
        
        # 对图像进行极坐标展开
        img_ext = yarn.extract(gray)    
        rect_img = yarn.Circle2Rectangle(img_ext)
        
        plt.imshow(cv.cvtColor(rect_img,cv.COLOR_BGR2RGB))
        plt.title(img_path)
        plt.axis("off")
        plt.show()
        plt.close()
    
main()