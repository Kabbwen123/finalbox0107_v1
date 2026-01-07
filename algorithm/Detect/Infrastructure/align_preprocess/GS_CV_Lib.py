import cv2
import numpy as np


# 窗口图像显示
def imageShow(imagename, imagemat, win_width = 800):
    height, width = imagemat.shape[0:2]
    height = int(height * win_width / width)
    img = cv2.resize(imagemat, (win_width, height), interpolation=cv2.INTER_AREA)
    cv2.imshow(imagename, img)


# 亮度调整
def imageadjustBright(image, brightness):
    # 确保图像是8位无符号整数类型
    img = np.uint8(image)
    # 调整亮度
    return cv2.convertScaleAbs(img, alpha=brightness, beta=brightness)


# 图像锐度调整
def imageadjustsharpen(image, kernel_size=3):
    # 使用拉普拉斯算子或自定义卷积核来锐化图像
    # 创建一个拉普拉斯算子核，也可以使用其他锐化核
    kernel = np.array([[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]], dtype=np.float32)  # 强锐化核
    sharpened = cv2.filter2D(image, -1, kernel)
    # 锐化后的图像通常会有一个偏置，可以通过加上原始图像的一部分来修正
    # sharpened = cv2.addWeighted(image, 1.5, sharpened, -0.5, 0)
    # 确保图像数据类型和范围正确
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# 图像饱和度调整
def imageadjustColor(image, Saturation = 1.0):
    # 将图像从BGR转换到HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 分离HSV通道的H, S, V
    h, s, v = cv2.split(hsv)
    # 只对掩码指定的像素应用饱和度提升
    s = np.clip(s * Saturation, 0, 255).astype(np.uint8)
    # 合并修改后的S通道和原始的H, V通道
    hsv_modified = cv2.merge([h, s, v])
    # 将图像从HSV转换回BGR
    img_modified = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2BGR)
    return img_modified


# 图像对比度度调整
def imageadjustContrast(image, contrast):
    img_result = np.clip(image.astype(np.float32) * contrast, 0, 255).astype(np.uint8)
    # 调整对比度
    return img_result


def adaptive_sharpen(img):
    """适合低对比度图像的锐化"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(2,2))
    l_sharp = clahe.apply(l)
    lab = cv2.merge((l_sharp, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def find_sobel_edge(image_gray, ksize=3):
    sobelx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=ksize)
    grad = cv2.magnitude(sobelx, sobely)
    return cv2.convertScaleAbs(grad)


# 图像膨胀处理
def imageerode(image, kernal=5, iterations=1):
    # 创建一个5x5的结构元素，通常用于膨胀操作
    kernel = np.ones((kernal, kernal), np.uint8)
    return cv2.erode(image, kernel, iterations)


# 图像腐蚀处理
def imagedilate(image, kernal=5, iterations=1):
    # 创建一个5x5的结构元素，通常用于膨胀操作
    kernel = np.ones((kernal, kernal), np.uint8)
    return cv2.dilate(image, kernel, iterations)


# 图像归一化处理   辉度拉伸到0-255
def imageNormalize(image):
    # 计算图像中的最小和最大灰度值
    min_val = np.min(image)
    max_val = np.max(image)
    # 确保 min_val 和 max_val 不是相等的，以避免除以零的错误
    if min_val == max_val:
        # 如果所有像素值都相同，可以将其设置为一个常数，例如0到255的中间值
        normalized_image = np.full_like(image, 128, dtype=np.uint8)
    else:
        # 线性变换到0-255范围
        # 使用np.clip确保值在0-255之间（虽然理论上是安全的，但在极端情况下可能有帮助）
        normalized_image = np.uint8(np.clip((image - min_val) * (255.0 / (max_val - min_val)), 0, 255))
    # 返回归一化图像
    return normalized_image


def get_UniqueTopBrightness(image, percent = 50):
    bright_list = np.unique(image)
    return bright_list[int(len(bright_list) * (100 - percent) / 100)]


def merge_rects(rect1, rect2):
    """
    合并两个矩形 (x, y, w, h)，返回一个能同时包住两者的新矩形。
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # 左上角最小值
    x_min = min(x1, x2)
    y_min = min(y1, y2)
    # 右下角最大值
    x_max = max(x1 + w1, x2 + w2)
    y_max = max(y1 + h1, y2 + h2)

    # 新矩形
    w_new = x_max - x_min
    h_new = y_max - y_min
    return (x_min, y_min, w_new, h_new)



# 图像旋转
def rotate_image(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]
    # 如果中心点为None，则使用图像中心
    if center is None:
        center = (w / 2, h / 2)
    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # 执行仿射变换（旋转）
    rotated = cv2.warpAffine(image, M, (w, h))
    # 返回旋转后的图像
    return rotated


def image_rotate_full(image, angle, center=None, scale=1.0, border_value=(0,0,0)):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    # === Step 1. 计算旋转矩阵 ===
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # === Step 2. 计算旋转后图像尺寸（避免裁切） ===
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    # === Step 3. 调整旋转矩阵中的平移部分 ===
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    # === Step 4. 仿射变换（旋转+平移） ===
    rotated = cv2.warpAffine(image, M, (new_w, new_h),
                             flags=cv2.INTER_LINEAR,
                             borderValue=border_value)
    return rotated


# 图像多边形比例变换
# 参数：
#    expand_ratio   扩展比例
#    expand_pixels  扩展后延申像素数
def expand_rect(rect, expand_ratio, expand_pixels):
    # 扩大矩形比例
    if expand_ratio != 1:
        rect = (int(rect[0]*expand_ratio), int(rect[1]*expand_ratio), int(rect[2]*expand_ratio), int(rect[3]*expand_ratio))
    # 假设rect是一个包含(x, y, width, height)的元组或列表
    x, y, width, height = rect
    # 扩展矩形的四个边
    new_x = max(0, x - expand_pixels)  # 防止x坐标变为负数
    new_y = max(0, y - expand_pixels)  # 防止y坐标变为负数
    new_width = width + 2 * expand_pixels
    new_height = height + 2 * expand_pixels
    # 返回新的矩形（RectB）
    return (new_x, new_y, new_width, new_height)


# 判断图像清晰度
# 检测方式：一般拍摄照片	  Laplacian 方差	100~300
#         工业相机实时检测	  Tenengrad 或 Lap+Edge 组合	150~500
#         模糊分类/失焦监测  Sobel/FFT 频域
# 返回：清晰度得分
def image_clearness(img, method='Laplacian'):
    # 将图像转为灰度图
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # 进行清晰度分析
    if method == 'Laplacian':
        # 使用拉普拉斯方差判断图像清晰度
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return lap.var()
    elif method == 'Sobel':
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad = cv2.magnitude(grad_x, grad_y)
        return np.mean(grad)
    elif method == 'Tenengrad':
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return np.mean(gx ** 2 + gy ** 2)
    elif method == 'FFT':
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        return np.mean(np.log1p(magnitude[gray.shape[0] // 4:-gray.shape[0] // 4, gray.shape[1] // 4:-gray.shape[1] // 4]))
    # 返回
    return 0


# 对比度拉伸
#   low_percent: 低端百分比裁剪
#   high_percent: 高端百分比裁剪
def contrast_stretching(image, low_percent=2, high_percent=98):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    # 计算百分位数
    low_val = np.percentile(gray, low_percent)
    high_val = np.percentile(gray, high_percent)
    # 线性拉伸
    stretched = np.clip((gray - low_val) * (255.0 / (high_val - low_val)), 0, 255)
    stretched = stretched.astype(np.uint8)
    # 返回结果
    return stretched


def image_grayto2value_first(image_gray):
    # 取得图像尺寸
    h, w = image_gray.shape
    # 切分图像
    y0 = 0; y1 = int(h * 0.5); y2 = h
    image_1 = image_gray[y0:y1, 0:w]
    image_2 = image_gray[y1:y2, 0:w]
    # 分别进行二值化处理
    _, image_2value_1 = cv2.threshold(image_1, 25, 255, cv2.THRESH_BINARY_INV)
    _, image_2value_2 = cv2.threshold(image_2, 10, 255, cv2.THRESH_BINARY_INV)
    # 合并图像并返回
    return cv2.vconcat([image_2value_1, image_2value_2])


def image_grayto2value_second(image_gray):
    # 取得图像尺寸
    h, w = image_gray.shape
    # 切分图像
    x0 = 0; x1 = int(w * 0.1); x2 = int(w * 0.9); x3 = w
    y0 = 0; y1 = int(h * 0.25); y2 = int(h * 0.8); y3 = h
    image_1 = image_gray[y0:y1, x0:x1]
    image_2 = image_gray[y0:y1, x1:x2]
    image_3 = image_gray[y0:y1, x2:x3]
    image_4 = image_gray[y1:y2, x0:x3]
    image_5 = image_gray[y2:y3, x0:x3]
    # 分别进行二值化处理
    _, image_2value_1 = cv2.threshold(image_1, 35, 255, cv2.THRESH_BINARY_INV)
    _, image_2value_2 = cv2.threshold(image_2, 40, 255, cv2.THRESH_BINARY_INV)
    _, image_2value_3 = cv2.threshold(image_3, 35, 255, cv2.THRESH_BINARY_INV)
    _, image_2value_4 = cv2.threshold(image_4, 35, 255, cv2.THRESH_BINARY_INV)
    _, image_2value_5 = cv2.threshold(image_5, 20, 255, cv2.THRESH_BINARY_INV)
    # 合并图像
    image_2value_123 = cv2.hconcat([image_2value_1, image_2value_2, image_2value_3])
    image_2value = cv2.vconcat([image_2value_123, image_2value_4, image_2value_5])
    # 返回图像
    return image_2value


# 图像中文字输出
def draw_text(image, point_lt, text, font = cv2.FONT_HERSHEY_PLAIN, font_scale = 1, color = (255, 255, 255), thickness = 1):
     # 绘制文本
    cv2.putText(
        img=image,
        text=text,
        org=point_lt,  # 文本左下角坐标
        fontFace=font,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_4 #cv2.LINE_AA  # 抗锯齿，使文本更平滑
    )

# 在该 box 中居中写字符串 text
def draw_text_centered(image, box, text, font = cv2.FONT_HERSHEY_PLAIN, font_scale=1, color=(255,255,255), thickness = 1):
    x, y, w, h = box
    # 1) 获取文字的尺寸
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    # 2) 计算文字左上角起点 (居中算法)
    text_x = x + (w - text_w) // 2
    text_y = y + (h + text_h) // 2  # text_h 是高度，从基线往上
    # 3) 绘制文本
    cv2.putText(image, text, (text_x, text_y),
                font, font_scale, color, thickness, cv2.LINE_AA)
    return image


# 读取单通道图像并强化边缘
# method: 边缘检测方法 ('sobel', 'laplacian', 'canny')
# strength: 边缘强化强度系数
def enhance_edges(image_gray, method='sobel', strength=1.0):
    # 应用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    # 根据选择的方法进行边缘检测
    if method == 'sobel':
        # Sobel算子 - 检测水平和垂直边缘
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobelx, sobely)
    elif method == 'laplacian':
        # Laplacian算子 - 检测各方向边缘
        edges = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    elif method == 'canny':
        # Canny边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        edges = edges.astype(np.float64)
    # 将边缘检测结果转换为绝对值并归一化
    edges = cv2.convertScaleAbs(edges)
    # 强化边缘：将原始图像与边缘图像结合
    enhanced = cv2.addWeighted(image_gray, 1.0, edges, strength, 0)
    # 返回
    return edges, enhanced


def uniformize_brightness(
    gray: np.ndarray,
    method: str = "clahe",      # "hist" | "clahe" | "bg_sub" | "homomorphic" | "retinex"
    clip_limit: float = 2.0,    # CLAHE参数
    tile_grid: int = 8,         # CLAHE网格
    bg_ksize: int = 101,        # 背景估计高斯核(奇数, 越大越“平”)
    eps: float = 1e-6           # 数值稳定项
):
    """对单通道图像进行亮度均匀化，返回uint8图。"""
    # 统一到8bit便于后续处理
    if gray.dtype != np.uint8:
        g = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        g = gray

    if method == "hist":
        # 全局直方图均衡
        out = cv2.equalizeHist(g)

    elif method == "clahe":
        # 自适应（推荐默认）
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
        out = clahe.apply(g)

    elif method == "bg_sub":
        # 背景估计 + 亮度整形（光照不均匀时很稳）
        # 1) 高斯模糊估计慢变背景
        bg = cv2.GaussianBlur(g, (bg_ksize, bg_ksize), 0)
        # 2) “除背景”得到相对反射分量，再归一化
        norm = (g.astype(np.float32) / (bg.astype(np.float32) + eps))
        norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # 可选：再轻度CLAHE增强
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        out = clahe.apply(norm)

    elif method == "homomorphic":
        # 同态滤波（频域抑制低频光照）
        f = np.log1p(g.astype(np.float32))
        F = np.fft.fft2(f)
        Fshift = np.fft.fftshift(F)

        h, w = g.shape[:2]
        cy, cx = h//2, w//2
        # 低频抑制的高通样式滤波器
        Y, X = np.ogrid[:h, :w]
        D2 = (Y - cy)**2 + (X - cx)**2
        D0 = max(h, w) * 0.05   # 截止频率，可调
        H = 1 - np.exp(-(D2 / (2*(D0**2))))  # 高斯高通

        Fh = Fshift * H
        F_ishift = np.fft.ifftshift(Fh)
        img_back = np.fft.ifft2(F_ishift).real
        out = cv2.normalize(np.expm1(img_back), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    elif method == "retinex":
        # 简化MSR（多尺度Retinex）
        def single_scale(img, sigma):
            blur = cv2.GaussianBlur(img, (0,0), sigma)
            return np.log1p(img.astype(np.float32)) - np.log1p(blur.astype(np.float32) + eps)

        scales = [15, 80, 250]  # 多尺度（像素），可按分辨率调整
        rr = sum(single_scale(g, s) for s in scales) / len(scales)
        out = cv2.normalize(rr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    else:
        raise ValueError("Unknown method")

    return out


# 计算指定区域的最小外接矩形短边中心点，并找出距离指定点最近的点
def nearest_narrow_side_center(contour, target_point):
    # 获取最小外接矩形
    (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
    box = cv2.boxPoints(((cx, cy), (w, h), angle))
    box = np.int16(box)
    # 计算4条边长度
    edges = [np.linalg.norm(box[(i + 1) % 4] - box[i]) for i in range(4)]
    # 找最短的两条边
    min_edge_indices = np.argsort(edges)[:2]
    # 计算窄边中点
    narrow_centers = []
    for i in min_edge_indices:
        p1 = box[i]
        p2 = box[(i + 1) % 4]
        mid = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
        narrow_centers.append(mid)
    # 找距离目标点最近的窄边中点
    target = np.array(target_point)
    distances = [np.linalg.norm(np.array(mid) - target) for mid in narrow_centers]
    nearest_point = narrow_centers[np.argmin(distances)]
    # 返回
    return nearest_point, narrow_centers


# 根据给定的点坐标拟合圆方程
# 输入：
#    points: (N,2) numpy数组，包含N个(x,y)
# 返回：
#    (xc, yc), r： 圆心、半径
def fit_circle(points):
    x = points[:, 0]
    y = points[:, 1]
    # 构造矩阵方程 [x y 1] * [A B C]^T = -(x^2 + y^2)
    A = np.column_stack((x, y, np.ones_like(x)))
    b = -(x ** 2 + y ** 2)
    # 最小二乘求解
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    A_, B_, C_ = params
    # 计算圆心和半径
    xc = int(-A_ / 2)
    yc = int(-B_ / 2)
    r = int(np.sqrt(xc ** 2 + yc ** 2 - C_))
    # 返回
    return xc, yc, r, (A_, B_, C_)


# 自动检测mask区域内的主色，并返回对应HSV区间
# 本函数只用于有限颜色种类的识别
# 输入：
#    image_bgr： 对象BGR图像
#    mask： ROI区域掩码
# 返回：
#    detected_color： 识别出的颜色名称(不可识别时返回None)
#    color_ranges： HSV颜色区间序列(有可能有多个颜色区间的情况)
def detect_color_in_mask(image_bgr, mask):
    # 定义常见颜色的HSV范围
    color_ranges_def = {
        "blue":   (np.array([96, 106, 0]), np.array([125, 255, 255])),
        "green":  (np.array([34, 50, 0]),  np.array([93, 245, 222])),
        "purple": (np.array([105, 54, 0]), np.array([145, 160, 255])),
        "red1":   (np.array([0, 70, 90]),  np.array([15, 255, 255])),
        "red2":   (np.array([170, 60, 5]), np.array([180, 255, 255])),
        "yellow": (np.array([16, 78, 0]),  np.array([57, 255, 241])),
    }
    # 转换为HSV
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    image_masked = cv2.bitwise_and(hsv, hsv, mask=mask)
    # 初始化返回值
    color_ranges = []
    detected_color = None
    # imageShow("mask", mask, 600)
    # imageShow("masked", image_masked, 600)
    # cv2.waitKey(0)
    (b, g, r, _) = cv2.mean(image_bgr, mask=mask)
    if (b < 60 and g < 60 and r < 60):
        detected_color = "black"
    else:
        for color_name, (lower, upper) in color_ranges_def.items():
            image_mask = cv2.inRange(image_masked, lower, upper)
            if cv2.countNonZero(image_mask) > 100:
                if ("red" in color_name):
                    detected_color = "red"
                    color_ranges.append(color_ranges_def.get("red1", None))
                    color_ranges.append(color_ranges_def.get("red2", None))
                else:
                    detected_color = color_name
                    color_ranges.append(color_ranges_def.get(color_name, None))
                break
    # 返回
    return detected_color, color_ranges


# 噪声图像消除处理
# 输入：
#    image_2Value： 二值化图像(必须)
#    area_threshold： 噪音区域面积阈值(可省略）
#    edge_threshold： 噪声区域最大边长阈值(可省略)
#    AreaorEdge： 阈值方式选择
# 返回：
#    image_mask： 消除噪声后的的掩码图像
#    contours_mask： 主要轮廓的多边形坐标序列
def clear_noise(image_2Value, area_threshold = 20000, edge_threshold = 10, AreaorEdge = True):
    # 初始化模板
    image_mask = np.zeros_like(image_2Value)
    contours_mask = []
    # 分析有效区域
    contours, _ = cv2.findContours(image_2Value, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        _, _, h, w = cv2.boundingRect(contour)
        if ((AreaorEdge and cv2.contourArea(contour) >= area_threshold) or
            (not AreaorEdge and (h if h > w else w) > edge_threshold)):
            contours_mask.append(contour)
            cv2.drawContours(image_mask, [contour], 0, 255, -1)
    # 返回掩码图像及轮廓
    return image_mask, contours_mask


# 通过HSV颜色区间取得图像
# 输入：
#    ranges_hsv: hsv_range序列  np.ndarray形式
# 返回：
#    image_mask 满足条件的掩码图像
def get_mask_by_HSVrange(image_BGR, ranges_hsv):
    # 1. 将BGR图像转换为HSV
    image_hsv = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV)
    # 2. 生成掩码模板
    image_mask = np.full((image_hsv.shape[:2]), 0, np.uint8)
    for (lower, upper) in ranges_hsv:
        image_mask_temp = cv2.inRange(image_hsv, lower, upper)
        image_mask = cv2.bitwise_or(image_mask_temp, image_mask)
    # 3. 返回结果
    return image_mask


def draw_cross_grid(
    img,
    angle_deg: float,          # 第一组平行线的角度（度），相对x轴逆时针
    step: float,               # 两条相邻线之间的垂直间距（像素）
    color=(0, 0, 0),           # BGR 颜色
    thickness: int = 1,        # 线宽（像素）
    offset: float = 0.0,       # 第一组线相对原点沿法向的偏移（像素）
    offset_cross: float = 0.0, # 第二组（与第一组垂直）线的偏移
    alpha: float = 1.0         # 与原图混合强度，1 直接画上；<1 半透明
):
    """
    返回绘制后的图像（不修改输入 img）。
    angle_deg=0 表示第一组线与x轴同向（水平），第二组自动取垂直方向（90度）。
    step 是“法向距离”单位（正交距离），与角度无关，统一用像素计。
    """
    if step <= 0:
        raise ValueError("step 必须为正数")
    out = img.copy()
    if out.ndim == 2:  # 灰度转BGR，便于画彩色线
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    H, W = out.shape[:2]
    overlay = out.copy()  # 在 overlay 上画，最后按 alpha 混合

    phi = np.deg2rad(angle_deg)
    # 第一组：方向向量 d1（沿线方向），单位法向 n1（垂直于线）
    d1 = np.array([np.cos(phi), np.sin(phi)], dtype=float)       # 沿线
    n1 = np.array([-np.sin(phi), np.cos(phi)], dtype=float)      # 法向（单位）

    # 第二组（交叉）：与第一组垂直
    d2 = n1.copy()
    n2 = -d1.copy()

    def _draw_family(n, d, step, offset):
        # 计算需要覆盖的 c 范围：ax + by = c，其中 (a,b)=n（单位法向）
        corners = np.array([[0, 0], [W-1, 0], [0, H-1], [W-1, H-1]], dtype=float)
        dots = corners @ n  # 角点在法向上的投影
        cmin, cmax = float(dots.min()), float(dots.max())

        # k 的范围，使 c = k*step + offset 覆盖 [cmin, cmax]
        k_start = int(np.floor((cmin - offset) / step))
        k_end   = int(np.ceil((cmax - offset) / step))

        L = W + H  # 线段绘制长度（足够大即可，OpenCV会自动裁剪）
        for k in range(k_start, k_end + 1):
            c = k * step + offset
            p0 = n * c                  # 直线上一个点（n 为单位法向）
            p1 = (p0 - d * L).astype(int)
            p2 = (p0 + d * L).astype(int)
            cv2.line(overlay, tuple(p1), tuple(p2), color, thickness, cv2.LINE_AA)

    _draw_family(n1, d1, step, offset)
    _draw_family(n2, d2, step, offset_cross)

    if 0 < alpha < 1.0:
        return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    else:
        return overlay


def fill_pattern_in_contours(imageBGR, contours, angle_def = 45, color = (0, 0, 255), step=30, thickness = 1):
    # 准备网格图
    image_pattern = draw_cross_grid(imageBGR.copy(), angle_def, step, color, thickness) # 网格旋转 30°
    # 准备掩码图像并绘制网格图
    image_mask = np.full((imageBGR.shape[:2]), 0, np.uint8)
    cv2.drawContours(image_mask, contours, -1, 255, -1)
    image_pattern = cv2.bitwise_and(image_pattern, image_pattern, mask = image_mask)
    # 准备掩码图像并保留contours以外的图像
    image_mask = np.full((imageBGR.shape[:2]), 255, np.uint8)
    cv2.drawContours(image_mask, contours, -1, 0, -1)
    image_result = cv2.bitwise_and(imageBGR, imageBGR, mask = image_mask)
    # 合并图像并返回
    return cv2.add(image_pattern, image_result)


def detect_edge(image, win_size = 10, edge_threshold = 0.05):
    # 取得图像尺寸
    h, w = image.shape[:2]
    # 创建输出
    image_result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_mask = np.full((h, w), 255, np.uint8)
    # 滑窗扫描
    for y in range(0, h, win_size):
        for x in range(0, w, win_size):
            y_end = min(y + win_size, h)
            x_end = min(x + win_size, w)
            roi = image[y:y_end, x:x_end]

            color_min = np.min(roi)
            color_max = np.max(roi)


            # print(color_min, color_max)


            # 用 Canny 检测边缘
            edges = cv2.Canny(roi, 50, 150)
            # 统计边缘像素比例
            edge_ratio = np.sum(edges > 0) / (roi.size + 1e-5)
            # 如果窗口内边缘明显，则标红
            # if edge_ratio > edge_threshold:
            if color_max - color_min > 20 and color_min < 200:
                image_mask[y:y_end, x:x_end] = roi
                #
                # gscv.imageShow("edges", edges)
                # cv2.waitKey(0)

                # cv2.rectangle(image_mask, (x, y), (x_end, y_end), 255, -1)
                cv2.rectangle(image_result, (x, y), (x_end, y_end), (0, 0, 255), 1)
    # # 使用椭圆结构元素，避免方形窗口造成角块伪影
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # # 先膨胀后腐蚀（闭运算）可连接小断点
    # image_mask = cv2.morphologyEx(image_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # 返回
    return image_result, image_mask


def stretch_gray_around_mean(image, alpha=0, beta=255):
    # 转为float防止溢出
    img = image.astype(np.float32)
    # 计算平均灰度
    mean_val = np.mean(img)
    # 以均值为中心拉伸（线性映射）
    min_val, max_val = np.min(img), np.max(img)
    # # 相对于平均值的偏移范围
    # max_offset = max(mean_val - min_val, max_val - mean_val)
    # # 归一化到 [-1, 1]
    # norm = (img - mean_val) / max_offset
    # # 拉伸到 [-A, B]
    # img_stretch = np.where(norm < 0, norm * alpha + mean_val, norm * beta + mean_val)
    # # 返回结果
    # return img_stretch, mean_val

    # 相对于平均值的偏移范围
    rate_down = (alpha - mean_val) / (min_val - mean_val)
    rate_up = (beta - mean_val) / (max_val - mean_val)
    # 归一化到 [-1, 1]
    norm = img - mean_val
    # 拉伸到 [-A, B]
    img_stretch = np.where(norm < 0, mean_val + norm * rate_down, mean_val + norm * rate_up)
    # 返回结果
    return img_stretch, mean_val


def erase_background(image_gray, alpha=50, beta=200):
    #
    # image_normal = cv2.boxFilter(image_gray, -1, (5, 5), normalize=True, borderType=cv2.BORDER_REPLICATE)
    image_normal = image_gray.copy()
    # 大尺寸模糊模拟背景（暗雾是低频成分）
    image_background = cv2.GaussianBlur(image_normal, (101, 101), 0)
    # 原图 - 背景  → 去除暗雾
    image_removed = cv2.subtract(image_normal, image_background)
    # 归一化提升亮部
    removed_norm = cv2.normalize(image_removed, None, 0, 255, cv2.NORM_MINMAX)
    removed_norm[removed_norm > 200] = 255
    removed_norm[removed_norm < 50] = 0
    #
    return removed_norm


def print_light_range(image_gray):
    min_Value = np.min(image_gray)
    max_Value = np.max(image_gray)
    print(min_Value, max_Value)


def line_angle(line):
    x1, y1, x2, y2 = line
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))


def line_length(line):
    x1, y1, x2, y2 = line
    return np.hypot(x2 - x1, y2 - y1)


# 关键函数：点到线的距离
def point_to_line_distance(px, py, line):
    x1, y1, x2, y2 = line
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return abs(A * px + B * py + C) / np.sqrt(A * A + B * B)


# 线到线的 “平均距离”
def line_to_line_distance(l1, l2):
    x1, y1, x2, y2 = l2
    d1 = point_to_line_distance(x1, y1, l1)
    d2 = point_to_line_distance(x2, y2, l1)
    return (d1 + d2) / 2


def filter_parallel_lines(lines,
                          angle_thresh=5,  # 角度差（°）
                          dist_thresh=5,  # 垂直距离（像素）
                          len_thresh=20):  # 长度差（像素）
    filtered = []
    used = [False] * len(lines)
    for i in range(len(lines)):
        if used[i]:
            continue
        base = lines[i]
        base_angle = line_angle(base)
        base_len = line_length(base)
        group = [base]
        for j in range(i + 1, len(lines)):
            if used[j]:
                continue
            cur = lines[j]
            # 1) 方向相似
            if abs(line_angle(cur) - base_angle) > angle_thresh:
                continue
            # 2) 距离很近（重复或重叠）
            if line_to_line_distance(base, cur) > dist_thresh:
                continue
            # 3) 长度差不大
            if abs(line_length(cur) - base_len) > len_thresh:
                continue
            # 这是一条重复线
            used[j] = True
            group.append(cur)
        # 选择组内最长的一条（最稳定）
        best = max(group, key=line_length)
        filtered.append(best)
    return filtered


def get_mean_value(image_gray, nonzero=True):
    # 全体平均
    if not nonzero:
        return np.mean(image_gray).astype(np.uint8)
    # 取所有非 0 像素
    nonzero_pixels = image_gray[image_gray > 0]
    if len(nonzero_pixels) > 0:
        return np.mean(nonzero_pixels).astype(np.uint8)
    else:
        return 0


def get_min_value(image_gray, nonzero=True):
    if nonzero:
        nonzero_pixels = image_gray[image_gray > 0]
        return np.min(nonzero_pixels).astype(np.uint8)
    else:
        return np.min(image_gray).astype(np.uint8)


"""
    img       : 目标图像
    rect      : (x, y, w, h)
    radius    : 圆角半径
    color     : BGR 颜色
    thickness : 线宽（-1 表示填充）
"""
def draw_rounded_rectangle(img, rect, radius=20, color=(0,255,0), thickness=2):
    x, y, w, h = rect
    r = radius
    # 防止圆角半径过大
    r = min(r, w//2, h//2)
    # 四个角点
    p1 = (x + r, y)
    p2 = (x + w - r, y)
    p3 = (x + w, y + r)
    p4 = (x + w, y + h - r)
    p5 = (x + w - r, y + h)
    p6 = (x + r, y + h)
    p7 = (x, y + h - r)
    p8 = (x, y + r)
    # 如果 thickness = -1，则做填充
    if thickness < 0:
        # 画中心矩形
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
        # 左右矩形
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
    # 绘制直线部分
    cv2.line(img, p1, p2, color, thickness)
    cv2.line(img, p3, p4, color, thickness)
    cv2.line(img, p5, p6, color, thickness)
    cv2.line(img, p7, p8, color, thickness)
    # 四个圆角
    cv2.ellipse(img, (x+r, y+r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x+w-r, y+r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x+w-r, y+h-r), (r, r), 0, 0, 90, color, thickness)
    cv2.ellipse(img, (x+r, y+h-r), (r, r), 90, 0, 90, color, thickness)
    # 返回
    return img


def image_emboss(image_gray):
    # 定义浮雕卷积核（你也可以换成上面那个 3x3）
    kernel = np.array([[-1, -1, 0],
                       [-1, 0, 1],
                       [0, 1, 1]], dtype=np.float32)
    # 卷积
    emboss = cv2.filter2D(image_gray, -1, kernel)
    # 加一个偏移量，把灰度拉回中间（否则会偏暗）
    emboss = cv2.add(emboss, 128)  # 整体提亮到“中灰”附近
    # 限制在 0~255 并转回 uint8
    emboss = np.clip(emboss, 0, 255).astype(np.uint8)

    return emboss


def sort_box_points(pts):
    """
    输入：4x2 的矩形四点坐标（任意顺序）
    输出：按 左上、右上、右下、左下 顺序排列后的坐标
    """

    pts = np.array(pts, dtype=np.float32)

    # 1. 先按 y 排序（top 两点，bottom 两点）
    y_sorted = pts[np.argsort(pts[:, 1])]

    top_two = y_sorted[:2]   # y 最小的 2 个点（上侧）
    bottom_two = y_sorted[2:] # y 最大的 2 个点（下侧）

    # 2. 上侧两个点按 x 排序：左上、右上
    top_left, top_right = top_two[np.argsort(top_two[:, 0])]

    # 3. 下侧两个点按 x 排序：左下、右下
    bottom_left, bottom_right = bottom_two[np.argsort(bottom_two[:, 0])]

    # 输出顺序：左上、右上、右下、左下
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def image_resize(image_src, scale = 1):
    if scale == 1 or scale <= 0:
        return image_src
    h, w = image_src.shape[:2]
    h, w = int(h * scale), int(w * scale)
    return cv2.resize(image_src, (w, h))


def image_apply_CLAHE(image):
    # 判断图像是否为单通道图像
    isGray = (len(image.shape) == 2)
    # CLAHE处理
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if isGray else image.copy()
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    image_clahe = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    # 返回结果
    return cv2.cvtColor(image_clahe, cv2.COLOR_BGR2GRAY) if isGray else image_clahe


def image_illumination(image_gray):
    background = cv2.GaussianBlur(image_gray, (51, 51), 0)
    corrected = image_gray / (background + 1e-6)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    corrected = corrected.astype(np.uint8)

    return corrected


def image_auto_gamma(image_gray):
    mean_val = image_gray.mean()
    # Auto gamma: darker image → smaller gamma；brighter → larger gamma
    gamma = np.clip(1.2 * (128.0 / max(mean_val, 1)), 0.5, 2.5)
    img_float = image_gray / 255.0
    corrected = np.power(img_float, gamma) * 255
    corrected = corrected.astype(np.uint8)

    return corrected


def image_gaussian_illumination_normalization(image_gray):
    blur = cv2.GaussianBlur(image_gray, (51, 51), 0)
    diff = image_gray.astype(np.float32) - blur.astype(np.float32)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    diff = diff.astype(np.uint8)

    return diff
