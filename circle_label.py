# -*- coding:utf-8 -*-
import os
import tkinter
import tkinter.filedialog
from time import sleep
from tkinter import *
from tkinter import filedialog
from tkinter.messagebox import showwarning

import cv2
import numpy as np
import pandas as pd
import scipy.spatial
from PIL import Image, ImageTk
from PIL import ImageGrab

from circle_detect import GetCircleList
from circle_show import GetLabelImg
from circle_calculate import GetTheta
from circle_gcenter import GetGravityCenter

# 创建tkinter主窗口
root = tkinter.Tk()
root.title('质心标注')

# 指定主窗口位置与大小
root.geometry('800x600+100+50')

# 不允许改变窗口大小
root.resizable(False, False)
root.focusmodel()

# 对话框参数设置
img_path = StringVar()  # 图片路径
xy_text = StringVar()  # 定义坐标显示位置

# 圆形参数
current_x = 0
current_y = 0
current_r = 0
gravity_x = 0
gravity_y = 0

# 保存列表
circle_infos = []
tmp_circle_infos = []

# 设置截取坐标全局变量， 元组类型
CapturePosition = ()

# 当前读取的图片序号
pic_index = 0

img_detect = 0


# 图片路径选取
def SelectIMGPath():
    # 全局变量
    global img_detect, tmp_circle_infos, circle_infos
    # 每次重新选取都清空数组
    tmp_circle_infos = []
    circle_infos = []
    # 选取路径
    file_path = filedialog.askopenfilename(title='请选择你要读取的图片', filetypes=[('JPEG图像', '*.jpg')])
    img_path.set(file_path)
    # 路径选取结束后，进行图片的读取
    img_detect, tmp_circle_infos = GetCircleList(img_path.get())
    pic_index = 0
    # 展示切割后的图片
    showPic()


tkinter.Label(root, text='图片路径:').place(x=0, y=0)
tkinter.Entry(root, textvariable=img_path).place(x=60, y=0)
tkinter.Button(root, text="①选取图片并进行Hough识别", command=SelectIMGPath).place(x=0, y=20)


# 点击坐标后的操作
def buttonCaptureClick():
    info_show = "圆心坐标(%d,%d)\n直线指向坐标(%d,%d)\n圆半径%d" % (
        current_x, current_y, gravity_x, gravity_y, current_r)
    # xy_text.set(str(w.selectPosition))
    circle_infos.append((current_x, current_y,
                         gravity_x, gravity_y,
                         current_r))
    xy_text.set(info_show)
    # 更换下一张
    showPic()
    # save2xlsx()


# 保存文件到excel
def save2xlsx():
    # 保存Excel文件
    file_path = filedialog.asksaveasfilename(title=u'保存Excel文件', defaultextension='xlsx')
    df = pd.DataFrame(circle_infos, columns=['圆心x', '圆心y', '质心x', '质心y', '圆半径'])
    df.sort_values(by=['圆心y', '圆心x'], inplace=True)
    df.to_excel(file_path, index=False)
    # 保存图片
    img_label, img_white_label = GetLabelImg(img_path.get(), file_path)
    cv2.imwrite(file_path.replace(".xlsx", "_原图_标注.jpg"), img_label)
    cv2.imwrite(file_path.replace(".xlsx", "_白底_标注.jpg"), img_white_label)


# 打开图片文件并显示
def showPic():
    # 设置全局变量
    global tmp_circle_infos, pic_index, img_detect, current_x, current_y, current_r, gravity_x, gravity_y
    # 判断图片是否读取完了
    if pic_index >= len(tmp_circle_infos):
        showwarning(title="警告",
                    message="全部检视完成！")
        return
    # 设定参数
    current_x = tmp_circle_infos[pic_index][0]
    current_y = tmp_circle_infos[pic_index][1]
    current_r = tmp_circle_infos[pic_index][2]

    try:
        # 图片裁剪
        sub_img = cv2.cvtColor(img_detect[current_y - current_r:
                                          current_y + current_r,
                               current_x - current_r:
                               current_x + current_r], cv2.COLOR_BGR2RGB)
        # 黑色画布
        image_bg = np.zeros(sub_img.shape, dtype="uint8")
        cv2.circle(image_bg, (current_r, current_r), int(current_r*0.95), (255, 255, 255), -1)
        sub_img = sub_img & image_bg
        # 灰度图像
        gray = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
        # 二值化
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        image_bg = np.zeros(binary.shape, dtype="uint8")
        cv2.circle(image_bg, (current_r, current_r), int(current_r*0.95), (255, 255, 255), -1)
        binary = binary & image_bg
        ret, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        x, y = GetGravityCenter(binary)
        gravity_x = x - current_r + current_x
        gravity_y = y - current_r + current_y

        cv2.circle(sub_img, (x, y), 10, (255, 255, 0), -1)

        cv2.circle(sub_img, (current_r, current_r), current_r, (255, 0, 0), 5, 1)
        cv2.imwrite("tmp_subimg.jpg", sub_img)
        cv2.imwrite("tmp_subimg_binary.jpg", binary)

        # 图片序号增加
        pic_index = pic_index + 1

        # 更新图片
        img_open = Image.open("tmp_subimg.jpg")
        img_show = ImageTk.PhotoImage(img_open)
        image_label.config(image=img_show)
        image_label.image = img_show

        # 删除文件
        os.remove("tmp_subimg.jpg")
    except:
        # 图片序号增加
        pic_index = pic_index + 1
        showPic()


# 图片显示区域
image_label = Label(root, bg='gray')
image_label.place(x=0, y=50, width=400, height=400)

# 完成图片读取识别之后展示图片
# showPic()

# 添加截图按钮功能
buttonCapture = tkinter.Button(root, text='②保留', command=buttonCaptureClick)
buttonCapture.place(x=10, y=450, width=100, height=40)

# 下一张
tkinter.Button(root, text='②舍弃', command=showPic).place(x=175, y=530, width=100, height=40)

# 坐标名称及显示坐标
tkinter.Label(root, textvariable=xy_text, fg='blue').place(x=125, y=450)

# 保存数据
save = tkinter.Button(root, text="③保存数据", command=save2xlsx, width=10, height=2)
save.place(x=300, y=530, width=100, height=40, anchor=NW)

# 右边的操作显示
tkinter.Label(root,
              text='当标注完两张图片并保存相关信息之后，\n请在下面的选择框里选择对应的两个Excel，\n可以进行对应的最近距离查找，\n节省一点匹配时间').place(
    x=500, y=0)

# Excel参数设置
excel_1_path = StringVar()  # 图片路径
excel_2_path = StringVar()  # 图片路径


# Excel路径选取
def SelectExcel1():
    # 选取路径
    file_path = filedialog.askopenfilename(title='请选择你要读取的Excel', filetypes=[('Excel', '*.xlsx')])
    excel_1_path.set(file_path)


def SelectExcel2():
    # 选取路径
    file_path = filedialog.askopenfilename(title='请选择你要读取的Excel', filetypes=[('Excel', '*.xlsx')])
    excel_2_path.set(file_path)


tkinter.Label(root, text='圆心数据1:').place(x=500, y=100)
tkinter.Entry(root, textvariable=excel_1_path).place(x=600, y=100)
tkinter.Button(root, text="①选取数据1", command=SelectExcel1).place(x=700, y=100)

tkinter.Label(root, text='圆心数据2:').place(x=500, y=150)
tkinter.Entry(root, textvariable=excel_2_path).place(x=600, y=150)
tkinter.Button(root, text="①选取数据2", command=SelectExcel2).place(x=700, y=150)

# Pic参数设置
pic_1_path = StringVar()  # 图片路径
pic_2_path = StringVar()  # 图片路径


# Pic路径选取
def SelectPic1():
    # 选取路径
    file_path = filedialog.askopenfilename(title='请选择你要读取的图片', filetypes=[('图片', '*.jpg')])
    pic_1_path.set(file_path)


def SelectPic2():
    # 选取路径
    file_path = filedialog.askopenfilename(title='请选择你要读取的图片', filetypes=[('图片', '*.jpg')])
    pic_2_path.set(file_path)


tkinter.Label(root, text='圆心数据1:').place(x=500, y=100)
tkinter.Entry(root, textvariable=excel_1_path).place(x=600, y=100)
tkinter.Button(root, text="①选取数据1", command=SelectExcel1).place(x=700, y=100)

tkinter.Label(root, text='圆心数据2:').place(x=500, y=150)
tkinter.Entry(root, textvariable=excel_2_path).place(x=600, y=150)
tkinter.Button(root, text="①选取数据2", command=SelectExcel2).place(x=700, y=150)


# 计算最近距离匹配
def CalculateNearInfo():
    # 读取文件
    circle_infos_1 = pd.read_excel(excel_1_path.get())
    circle_infos_2 = pd.read_excel(excel_2_path.get())
    # 求距离
    mat = scipy.spatial.distance.cdist(circle_infos_1[['圆心x', '圆心y']],
                                       circle_infos_2[['圆心x', '圆心y']], metric='euclidean')
    distance_df = pd.DataFrame(mat, index=circle_infos_1.index, columns=circle_infos_2.index)
    # 标号
    circle_infos_1['标号'] = circle_infos_1.index
    circle_infos_2['标号'] = circle_infos_2.index
    # 自然连接
    circle_infos_1['join'] = distance_df.idxmin(axis=1)
    circle_infos_1['距离'] = distance_df.min(axis=1)
    circle_infos_2['join'] = circle_infos_2.index
    map_circle_infos = pd.merge(circle_infos_1, circle_infos_2, on='join', suffixes=['_1', '_2'], how='inner')
    # 删除无用列
    del map_circle_infos['join']

    # 保存Excel文件
    file_path = filedialog.asksaveasfilename(title=u'保存合并后Excel文件', defaultextension='xlsx')
    map_circle_infos.to_excel(file_path, index=False)

    file_path = filedialog.asksaveasfilename(title=u'保存合并后前一张图的Excel文件', defaultextension='xlsx')
    tmp_df = map_circle_infos.iloc[:, :5]
    tmp_df.columns = ['圆心x', '圆心y', '质心x', '质心y', '圆半径']
    tmp_df.to_excel(file_path, index=False)
    # 保存图片
    img_label, img_white_label = GetLabelImg(
        filedialog.askopenfilename(title='请选择你要读取的图片（前一张图片的原图）', filetypes=[('JPEG图像', '*.jpg')]),
        file_path)
    cv2.imwrite(file_path.replace(".xlsx", "_原图_标注.jpg"), img_label)
    cv2.imwrite(file_path.replace(".xlsx", "_白底_标注.jpg"), img_white_label)

    file_path = filedialog.asksaveasfilename(title=u'保存合并后后一张图的Excel文件', defaultextension='xlsx')
    tmp_df = map_circle_infos.iloc[:, 7:12]
    tmp_df.columns = ['圆心x', '圆心y', '质心x', '质心y', '圆半径']
    tmp_df.to_excel(file_path, index=False)
    # 保存图片
    img_label, img_white_label = GetLabelImg(
        filedialog.askopenfilename(title='请选择你要读取的图片（后一张图片的原图）', filetypes=[('JPEG图像', '*.jpg')]),
        file_path)
    cv2.imwrite(file_path.replace(".xlsx", "_原图_标注.jpg"), img_label)
    cv2.imwrite(file_path.replace(".xlsx", "_白底_标注.jpg"), img_white_label)


tkinter.Button(root, text="②计算最近的数据并保存文件到本地", command=CalculateNearInfo).place(x=500, y=200)


# Excel路径选取
def SelectFinalExcel():
    # 选取路径
    file_path = filedialog.askopenfilename(title='请选择你要读取的Excel', filetypes=[('Excel', '*.xlsx')])
    # 获取数据
    final_circle_infos = pd.read_excel(file_path)
    # 计算角度
    final_circle_infos['偏转角度'] = final_circle_infos.apply(
        lambda x: int(GetTheta((x[2] - x[0], x[3] - x[1]), (x[9] - x[7], x[10] - x[8]))), axis=1)
    # 保存Excel文件
    file_path = filedialog.asksaveasfilename(title=u'保存Excel文件', defaultextension='xlsx')
    final_circle_infos.to_excel(file_path, index=False)


tkinter.Label(root, text='最后匹配完成的Excel:').place(x=500, y=270)
tkinter.Button(root, text="①选取Excel并计算输出角度", command=SelectFinalExcel).place(x=500, y=300)

# 启动消息主循环
root.update()
root.mainloop()
