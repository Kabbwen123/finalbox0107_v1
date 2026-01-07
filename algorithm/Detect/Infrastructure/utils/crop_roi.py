import os
import glob
import time
import cv2

# ================== 配置区 ==================
# 输入图片所在文件夹
INPUT_DIR = r"C:\Users\Kabbw\Desktop\Projects\1203_iriso_anomalib\DATA\1204_trainingDATA\OK\val"

# 输出 ROI 图片保存路径（自动创建）
OUTPUT_DIR = r"C:\Users\Kabbw\Desktop\Projects\1203_iriso_anomalib\DATA\1204_pin\trainingDATA\OK\val"

# ROI 区域（x1, y1, x2, y2）
ROI = (310, 40, 2740, 240)
# ==========================================


def crop_roi(image_bgr, roi=None):
    """
    只做一件事：根据 ROI 从原图中裁剪出对应区域。
    roi: (x1, y1, x2, y2)，如果为 None 则返回原图
    """
    if roi is None:
        return image_bgr

    x1, y1, x2, y2 = roi

    # 简单做个边界保护，防止超出范围
    h, w = image_bgr.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        # ROI 不合法，直接返回原图（也可以选择返回 None）
        return image_bgr

    return image_bgr[y1:y2, x1:x2]


def main():
    # 确保输出文件夹存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 搜索当前文件夹下的 jpg 图片
    pattern = os.path.join(INPUT_DIR, "*.jpg")
    jpg_files = glob.glob(pattern)

    if not jpg_files:
        print("未在输入路径中找到任何 jpg 图片。")
        return

    for index, image_path in enumerate(jpg_files):
        # 取得图像名称（不含扩展名）
        filename = os.path.splitext(os.path.basename(image_path))[0]

        # 加载图像
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[WARN] 无法读取图像: {image_path}")
            continue

        start = time.time()

        # 只做 ROI 截取
        image_result = crop_roi(image_bgr, ROI)

        elapsed = time.time() - start
        print(f"{index + 1:04d} \t {filename} \t Elapsed time: {elapsed:0.5f}s")

        # 输出到文件
        output_image_path = os.path.join(OUTPUT_DIR, f"{filename}.jpg")
        cv2.imwrite(output_image_path, image_result, [cv2.IMWRITE_JPEG_QUALITY, 100])

    print("Completed!")


if __name__ == "__main__":
    main()
