import matplotlib.pyplot as plt
import numpy as np


# 计算角度
def dot_product_angle(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        print("Zero magnitude vector!")
    else:
        vector_dot_product = np.dot(v1, v2)
        arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle = np.degrees(arccos)
        return angle
    return 0


# 计算顺时针旋转角度
def clockwise_angle(v1, v2):
    x1, y1 = v1
    x2, y2 = v2
    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2
    theta = np.arctan2(det, dot)
    theta = theta if theta > 0 else 2 * np.pi + theta
    return theta * 180 / np.pi


# 向量标准化
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# 获取角度
def GetTheta(v1, v2):
    circle_slide = eval('%.5f' % dot_product_angle(v1, v2))
    circle_clockwise_slide = eval('%.5f' % clockwise_angle(v1, v2))
    return -circle_slide if circle_slide == circle_clockwise_slide else +circle_slide


if __name__ == '__main__':
    circle_slide = eval('%.5f' % dot_product_angle(np.array([1, 2]), np.array([0, 2])))
    circle_clockwise_slide = eval('%.5f' % clockwise_angle(np.array([1, 2]), np.array([0, 2])))
    if circle_slide == circle_clockwise_slide:
        print(-circle_slide)
    else:
        print(circle_slide)
