import cv2
import numpy as np
import pandas as pd


# 获取两张img
def GetLabelImg(img_path, excel_path):
    img = cv2.imread(img_path, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    white_img = np.zeros(img.shape)
    ret, white_img = cv2.threshold(white_img, 5, 255, cv2.THRESH_BINARY_INV)

    img = cv2.bitwise_not(img)

    circle_infos = pd.read_excel(excel_path)
    for i in range(0, len(circle_infos)):
        current_x = circle_infos.loc[i, "圆心x"];
        current_y = circle_infos.loc[i, "圆心y"];
        current_r = circle_infos.loc[i, "圆半径"]
        current_xx = circle_infos.loc[i, "质心x"];
        current_yy = circle_infos.loc[i, "质心y"];
        cv2.circle(img, (current_x, current_y), current_r, (255, 0, 0), 15)  # 画圆
        cv2.circle(img, (current_x, current_y), 2, (255, 0, 0), 10)  # 画圆心
        cv2.arrowedLine(img, (current_x, current_y), (current_xx, current_yy), (255, 0, 0), 15, current_r, 0,
                        0.3)  # 画箭头
        cv2.putText(img, str(i), (current_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 15,
                    cv2.LINE_AA)

        cv2.circle(white_img, (current_x, current_y), current_r, (255, 0, 0), 15)  # 画圆
        cv2.circle(white_img, (current_x, current_y), 2, (255, 0, 0), 10)  # 画圆心
        cv2.arrowedLine(white_img, (current_x, current_y), (current_xx, current_yy), (255, 0, 0), 15, current_r, 0,
                        0.3)  # 画箭头
        cv2.putText(white_img, str(i), (current_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 15,
                    cv2.LINE_AA)
    return img, white_img


if __name__ == '__main__':
    pic1, pic2 = GetLabelImg('./Data.imgs/1.jpg', '圆心及质心数据.xlsx')
    cv2.imwrite("划线展示1.jpg", pic1)
    cv2.imwrite("划线展示2.jpg", pic2)
