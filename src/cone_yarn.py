import cv2 as cv
import numpy as np
import math

class yarn:
    def __init__(self, center_in, center_out, ri, ro, mask):
      self.center_in = center_in
      self.center_out = center_out
      self.ri = ri
      self.ro = ro 
      self.mask = mask

    def extract(self,img):
       return cv.bitwise_and(img, img, mask=self.mask)
    
    def Circle2Rectangle(self, img):
        cx,cy = self.center_out
        radius_width = self.ro - self.ri

        out_width = int(2 * math.pi * self.ro)
        out_height = radius_width

        rect_img = np.zeros((out_height, out_width, 3), dtype=np.uint8)

        for row in range(out_height):
            rho = self.ro - row - 1
            for col in range(out_width):
                # 当前角度
                theta = 2 * math.pi * col / out_width
                # 极坐标 → 笛卡尔坐标
                x = int(cx + rho * math.cos(theta) + 0.5)
                y = int(cy - rho * math.sin(theta) + 0.5)
                # 边界检查
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    rect_img[row, col] = img[y, x]
        return rect_img
       
    def draw(self, img, thickness=2):
        vis = img.copy()
        cv.circle(vis, self.center_out, self.ro, (0, 255, 0), thickness)
        cv.circle(vis, self.center_in, self.ri, (0, 255, 0), thickness)
        cv.circle(vis, self.center_out, 10, (0, 0, 255), -1)
        return vis
