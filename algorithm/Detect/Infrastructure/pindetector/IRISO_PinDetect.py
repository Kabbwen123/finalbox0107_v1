import os
import cv2
import time
import math
import glob
import numpy as np
import Infrastructure.align_preprocess.GS_CV_Lib as gscv
import Infrastructure.align_preprocess.GS_Common as gscommon

'''1217'''
def detect_pin(image_gray, pincount=35, offset_up=40, offset_down=60, pinsizethreshold=500, errsizethreshold=30):
    # 初始化返回值
    errBoxs = []
    # 取得图像信息
    height, width = image_gray.shape[:2]
    top, bottom = offset_up, height - offset_down
    step = float(width) / pincount
    # 裁剪ROI区域
    image_pin = image_gray[top:bottom, 0:width]
    height_ROI, width_ROI = image_pin.shape
    # 归一化处理
    image_normal = cv2.normalize(image_pin, None, 0.0, 255.0, cv2.NORM_MINMAX).astype(np.uint8)
    # 均值自适应阈值
    image_2Value = cv2.adaptiveThreshold(image_normal, 255, cv2.ADAPTIVE_THRESH_MEAN_C,  # 使用邻域均值
                                         cv2.THRESH_BINARY,121, 5 )
    # 降噪处理
    image_2Value, _ = gscv.clear_noise(image_2Value, area_threshold=100, AreaorEdge=True)
    image_2Value = cv2.dilate(image_2Value, (5, 5), iterations=1)
    image_result = cv2.cvtColor(image_2Value, cv2.COLOR_GRAY2BGR)
    # 开运算，提取针脚轮廓
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    image_2Value = cv2.morphologyEx(image_2Value, cv2.MORPH_OPEN, kernel)
    # 去除两侧阴影
    cv2.rectangle(image_2Value, (0, 0), (5, height_ROI), 255, -1)
    cv2.rectangle(image_2Value, (width_ROI - 5, 0), (width_ROI, height_ROI), 255, -1)
    # 计算针脚宽度
    pincount_actual, pinwidth_sum, centerx_sum = 0, 0.0, 0.0
    for i in range(pincount):
        # 取得当pin的ROI
        left, right = max(0, int(step * (i - 0.2))), min(width, int(step * (i + 1.2)))
        image_ROI = cv2.bitwise_not(image_2Value[0:height_ROI, left:right])
        image_ROI = cv2.erode(image_ROI, (3, 3), iterations=2)
        # 查找轮廓
        contours, _ = cv2.findContours(image_ROI, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 如果没有找到轮廓，返回None
        isError = False
        if contours is not None:
            # 找到面积最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) >= pinsizethreshold:
                x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                pincount_actual = pincount_actual + 1
                centerx_sum = centerx_sum + int(x + w /2)
                pinwidth_sum = pinwidth_sum + min(w, h)
            else:
                isError = True
        else:
            isError = True
        if isError:
            left_b, right_b = int(step * i), int(step * (i + 1))
            errBoxs.append([(left_b, top), (right_b, bottom)])
    pinwidth = pinwidth_sum / pincount_actual + 3
    pincenter_x = int(centerx_sum / pincount_actual)
    # 将针脚的合理位置用白色掩掉
    for i in range(pincount):
        centerx = max(0, int(step * (i - 0.2) + pincenter_x))
        left, right = int(centerx - pinwidth / 2) , int(centerx + pinwidth / 2)
        cv2.rectangle(image_2Value, (left, 0), (right, height_ROI), 255, -1)
    # 逐pin进行分析
    for i in range(pincount):
        # 取得当pin的ROI
        left, right = max(0, int(step * (i - 0.2))), min(width, int(step * (i + 1.2)))
        image_ROI = cv2.bitwise_not(image_2Value[0:height_ROI, left:right])
        image_ROI = cv2.erode(image_ROI, (3, 3), iterations=2)
        # 查找轮廓
        contours, hierarchy = cv2.findContours(image_ROI, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 如果没有找到轮廓，返回None
        isError = False
        if contours is not None and len(contours) > 0:
            # 找到面积最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            # 取得最小矩形
            _, _, w, h = cv2.boundingRect(largest_contour)
            area = cv2.contourArea(largest_contour)
            isError = cv2.contourArea(largest_contour) >= errsizethreshold and min(w, h) > 8
        # 可视化检测结果
        if isError:
            left_b, right_b = int(step * i), int(step * (i + 1))
            errBoxs.append([(left_b, top), (right_b, bottom)])

    # # Debug用
    # for i in range(pincount):
    #     centerx = max(0, int(step * (i - 0.2) + pincenter_x))
    #     left, right = int(centerx - pinwidth / 2), int(centerx + pinwidth / 2)
    #     cv2.rectangle(image_gray, (left, top), (right, bottom), (0, 0, 255), 2)
    # gscv.imageShow("org", image_gray, 1400)
    # gscv.imageShow("closed", image_2Value, 1400)
    # cv2.waitKey(0)
    print("当前pinsizethreshold是 ", pinsizethreshold)
    # 返回针脚部分二值化图像和不良针脚位置
    return image_result, errBoxs

if __name__ == "__main__":
    # 搜索当前文件夹
    pattern = os.path.join(r"C:\Users\Kabbw\Desktop\Projects\1203_iriso_anomalib\DATA\1204_trainingDATA\OK\testpinsmall", "*.jpg")
    jpg_files = glob.glob(pattern)
    # jpg_files = [r"D:\IRISO\1204\NG\09060073_2_alt.jpg"]
    # 逐文件进行分析处理
    for index, image_path in enumerate(jpg_files):
        # 取得图像名称
        filename = gscommon.get_filename_without_extension(image_path)
        image_org = cv2.imread(rf"C:\Users\Kabbw\Desktop\Projects\1203_iriso_anomalib\DATA\1204_trainingDATA\OK\testpinsmall\{filename}.jpg", cv2.IMREAD_COLOR_BGR)
        # image_pin = image_org[40:240, 310:2740]
        image_pin =  image_org[20:120, 154:1364]

        image_gray = cv2.cvtColor(image_pin, cv2.COLOR_BGR2GRAY)
        # image_visible = image_org[40:240, 310:2740]
        image_visible = image_org[20:120, 154:1364]
        # image_pin = image_org[20:120, 154:1364] 这个是在宽1520，高468的换算。
        
        
        # 针脚检测
        start = time.time()
        image_pin, errBoxs = detect_pin(image_gray, pincount=35, offset_up=35, offset_down=45)
        print(f"Spend time = {time.time() - start}")
        # 可视化
        for errBox in errBoxs:
            # 透明度（0~1，越大越不透明）
            alpha = 0.2
            [left, top], [right, bottom] = errBox[0], errBox[1]
            # 取出 ROI
            image_ROI = image_visible[top:bottom, left:right]
            # 拷贝一层作为叠加层
            image_overlay = image_ROI.copy()
            # 在 overlay 上画实心红色矩形（BGR: 红色为 (0, 0, 255)）
            cv2.rectangle(image_overlay, (0, 0), (right - left, bottom - top), (0, 0, 255), thickness=-1)
            # 将 overlay 按 alpha 与原 ROI 融合
            cv2.addWeighted(image_overlay, alpha, image_ROI, 1 - alpha, 0, image_ROI)
            cv2.rectangle(image_visible, [left + 2, top], [right - 2, bottom], (0, 0, 255), 2)
        # image_result = cv2.vconcat([image_org, image_pin, image_visible])
        output_image_path = rf"D:\IRISO\1204\Pin\{filename}.jpg"
        if len(errBoxs) > 0:
            cv2.imwrite(output_image_path, image_org)
        # gscv.imageShow("Result", image_org, 1400)
        # cv2.waitKey(0)