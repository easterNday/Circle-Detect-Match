import cv2 as cv
import numpy as np


def GetGravityCenter(img):
    h, w = img.shape
    project_img = np.zeros(shape=(img.shape), dtype=np.uint8) + 255
    x_total = 0
    y_total = 0
    xx_total = 0
    yy_total = 0
    for j in range(w):
        for i in range(h):
            if img[i][j] == 0:
                x_total += 1
                y_total += 1
                xx_total += i
                yy_total += j
    return int(yy_total / y_total),int(xx_total / x_total)


if __name__ == '__main__':
    src = cv.imread("./1111.jpg")
    # 灰度图像
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # 二值化
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    x, y = GetGravityCenter(binary)
    print(x, y)
    print(src.shape)
    image = cv.circle(src, (y, x), 5, (255, 0, 0), -1)
    cv.imwrite("111.jpg", src)
