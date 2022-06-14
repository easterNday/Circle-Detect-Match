# This is a sample Python script.
from types import NoneType

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

Detect_circles = []

# 霍夫圆检测算法
def HoughDetect(img, m, n):
    # 切割后的图像
    split_img = img[m:n, 0:4800]
    # 基于Hough函数法进行圆形提取
    gray = cv2.cvtColor(split_img, cv2.COLOR_BGR2GRAY)
    circles = []
    # 此处是8mm对应的圆柱，检测对应大小160，给定区间150-170
    circle1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 320, param1=100, param2=10, minRadius=150,
                               maxRadius=170)  # 160
    if type(circle1) != NoneType:
        circles.extend(circle1[0, :, :])  # 提取为二维
    # 此处是5mm对应的圆柱，检测对应大小100，给定区间90-110
    circle2 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 200, param1=100, param2=10, minRadius=90,
                               maxRadius=110)  # 100
    if type(circle2) != NoneType:
        circles.extend(circle2[0, :, :])  # 提取为二维
    # 此处是10mm对应的圆柱，检测对应大小200，给定区间190-210
    circle3 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 400, param1=100, param2=10, minRadius=190,
                               maxRadius=210)  # 200
    if type(circle3) != NoneType:
        circles.extend(circle3[0, :, :])  # 提取为二维

    circles = np.uint16(np.around(circles))  # 四舍五入，取整
    for i in range(0, len(circles)):
        circles[i][1] = circles[i][1] + m
    return circles

def GetCircleList(img_path):
    # 返回的数组
    circle_list = []

    # 基于Hough函数法进行圆形提取
    img_detect = cv2.imread(img_path, 1)
    img_detect = cv2.cvtColor(img_detect, cv2.COLOR_BGR2RGB)

    # Detect_circles.extend(HoughDetect(img, 300, 2700))
    circle_list.extend(HoughDetect(img_detect, 600, 2700))
    circle_list.extend(HoughDetect(img_detect, 2500, 4900))
    circle_list.extend(HoughDetect(img_detect, 4900, 7100))
    circle_list.extend(HoughDetect(img_detect, 6900, 9000))
    # Detect_circles.extend(HoughDetect(img, 6900, 9300))

    return img_detect, circle_list

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    img, Detect_circles = GetCircleList('Data.imgs/1.jpg')

    df = pd.DataFrame(Detect_circles, columns=['圆心x', '圆心y', '圆半径'])
    df.sort_values(by=['圆心y', '圆心x'], inplace=True)
    print(df)

    i = 0
    for index, row in df.iterrows():
        i = i + 1
        cv2.circle(img, (row['圆心x'], row['圆心y']), row['圆半径'], (255, 0, 0), 5)  # 画圆
        cv2.circle(img, (row['圆心x'], row['圆心y']), 2, (255, 0, 0), 10)  # 画圆心
        cv2.putText(img, str(i), (row['圆心x'], row['圆心y']), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 100, 255), 2,
                    cv2.LINE_AA)
    # 保存到本地excel
    df.reset_index(drop=True, inplace=True)
    df.to_excel("存储数据.xlsx",index=False)
    cv2.imwrite("圆识别.jpg", img)
    plt.imshow(img)
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
