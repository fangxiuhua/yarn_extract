import  cv2 as cv
import  numpy as np
from jinja2.nodes import Continue


def extract_ring(img, binary):
    """
    从二值图像中提取圆环区域

    参数：
        img    : 原始彩色图像 (BGR)
        binary : 二值图像 (uint8, 0/255)

    返回：
        result : 提取出的圆环彩色图
        mask   : 圆环掩模（单通道）
    """

    # 1. 查找轮廓及层级
    contours, hierarchy = cv.findContours(binary, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        print("未找到外轮廓")

    hierarchy = hierarchy[0]

    # 2. 找最大外轮廓（父轮廓）
    max_outer_area = 0
    outer_idx = -1

    for i, cnt in enumerate(contours):
        # 父轮廓
        if hierarchy[i][3] == -1:
            area = cv.contourArea(cnt)
            if area > max_outer_area:
                max_outer_area = area
                outer_idx = i

    if outer_idx == -1:
        print("未找到外轮廓")

    # 3. 在该外轮廓中找最大的内轮廓
    max_inner_area = 0
    inner_idx = -1

    for i, cnt in enumerate(contours):
        # 子轮廓
        if hierarchy[i][3] == outer_idx:
            area = cv.contourArea(cnt)
            if area > max_inner_area:
                max_inner_area = area
                inner_idx = i

    # 4. 构建圆环掩模
    mask = np.zeros(binary.shape, dtype=np.uint8)

    # 填充外轮廓
    cv.drawContours(mask, [contours[outer_idx]], -1, 255, -1)

    # 挖掉内轮廓（如果存在）
    if inner_idx != -1:
        cv.drawContours(mask, [contours[inner_idx]], -1, 0, -1)

    # 5. 用 mask 提取圆环
    result = cv.bitwise_and(img, img, mask=mask)

    return result, mask