import  cv2 as cv
import  numpy as np
from jinja2.nodes import Continue


def extract_ring(img, binary):
    """从二值图像中提取圆环区域"""

    # 1. 查找轮廓及层级
    contours, hierarchy = cv.findContours(binary, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        print("未找到外轮廓")

    hierarchy = hierarchy[0]

    # 2. 找最大外轮廓
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


    # 拟合外圆
    (xo, yo), ro = cv.minEnclosingCircle(contours[outer_idx])

    # 拟合内圆
    (xi, yi), ri = cv.minEnclosingCircle(contours[inner_idx])
    # 4. 构建圆环掩模
    mask = np.zeros(binary.shape, dtype=np.uint8)

    # 填充外轮廓
    cv.circle(mask, (int(xo), int(yo)), int(ro), 255, -1)

    # 挖掉内轮廓
    if inner_idx != -1:
        cv.circle(mask, (int(xi), int(yi)), int(ri), 0, -1)

    # 5. 用 mask 提取圆环
    result = cv.bitwise_and(img, img, mask=mask)

    return result,xo,yo