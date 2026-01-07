import os
import time
import cv2
import glob
import shutil
from pathlib import Path
import Infrastructure.align_preprocess.GS_CV_Lib as gscv
import Infrastructure.align_preprocess.GS_Common as gscommon


def file_copy(A_DIR, B_DIR, C_DIR):
    A = Path(A_DIR)
    B = Path(B_DIR)
    C = Path(C_DIR)
    C.mkdir(parents=True, exist_ok=True)
    # 加速搜索：构造 B 文件夹的“文件名 → 路径”字典
    b_dict = {p.name: p for p in B.iterdir() if p.is_file()}
    print(f"[INFO] B 中共有 {len(b_dict)} 张图片")
    IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    total = 0
    # 遍历 A 的所有子文件夹（包括多级）
    for root, dirs, files in os.walk(A):
        root_path = Path(root)
        # 只处理最底层文件夹
        if not all(not p.is_dir() for p in root_path.iterdir()):
            continue
        relative_path = root_path.relative_to(A)  # 用于保持结构
        target_dir = C / relative_path
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[DIR] 最底层目录：{relative_path}")
        for file_name in files:
            file_path = root_path / file_name
            if file_path.suffix.lower() not in IMG_EXT:
                continue
            if file_name in b_dict:
                shutil.copy2(b_dict[file_name], target_dir / file_name)
                total += 1
                print(f"[COPY] {file_name} → {target_dir}")
            else:
                print(f"[MISS] {file_name} 不在 B 中")
    print(f"\n[FINISH] 共复制 {total} 个文件到 C 文件夹")


def imagepro_for_AIdetect(image_BGR, image_size = None, blur=False, contrast=1.2, ROI=None):
    # 转为灰度图
    image_gray = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2GRAY)    # 目前patchcore训练数据所用的预处理方式（2025.12.11）
    # image_gray = gscv.image_apply_CLAHE(image_gray)
    # image_gray1 = gscv.image_auto_gamma(image_gray)

    # gscv.imageShow("AA", image_gray, 1200)
    # gscv.imageShow("BB", image_gray1, 1200)
    # cv2.waitKey(0)
    # 图像尺寸归一化处理
    if image_size is not None:
        image_gray = cv2.resize(image_gray, image_size)

    # 进行ROI截取
    if ROI is not None:
        x1, y1, x2, y2 = ROI
        image_gray = image_gray[y1:y2, x1:x2]
    # 先高斯滤波能更稳
    image_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
    # 通用稳妥：自适应亮度均衡
    image_target = gscv.uniformize_brightness(image_blur, method="bg_sub", clip_limit=2.0, tile_grid=8)
    # 锐化处理
    image_result = gscv.imageadjustsharpen(image_target, 5)
    # 柔化处理（可选）
    if blur:
        image_result = cv2.bilateralFilter(image_result, 9, 100, 100)
    # 对比度增强（可选）
    if contrast != 1.0:
        image_result = gscv.imageadjustContrast(image_result, contrast)
    # 返回
    return image_result

# def imagepro_for_AIdetect(
#     image_BGR,
#     image_size=None,
#     blur=False,
#     contrast=1.2,
#     ROI=None,
#     # ===== 新增：开关参数（默认值保证“与原来一致”）=====
#     enable_gray=True,
#     enable_resize=True,
#     enable_roi=True,
#     enable_gaussian=True,
#     enable_bg_sub=True,
#     enable_sharpen=True,
#     enable_bilateral=None,   # None 表示跟随 blur 参数（保持原行为）
#     enable_contrast=True,
#     # ===== 可选：一些可调参数（默认等于原来写死的值）=====
#     gaussian_ksize=(5, 5),
#     gaussian_sigma=0,
#     bg_method="bg_sub",
#     bg_clip_limit=2.0,
#     bg_tile_grid=8,
#     sharpen_amount=5,
#     bilateral_d=9,
#     bilateral_sigmaColor=100,
#     bilateral_sigmaSpace=100,
# ):
#     """
#     与你原来的 imagepro_for_AIdetect 行为保持一致的版本（默认参数不变时输出一致）：
#       1) BGR->Gray
#       2) resize（若 image_size != None）
#       3) ROI 裁剪（若 ROI != None，ROI 为 x1,y1,x2,y2）
#       4) GaussianBlur(5,5)
#       5) uniformize_brightness(method="bg_sub", clip_limit=2.0, tile_grid=8)
#       6) sharpen(amount=5)
#       7) 若 blur=True，则 bilateralFilter(...)
#       8) 若 contrast != 1.0，则 adjustContrast(contrast)

#     新增的 enable_* 开关默认都为 True（或与 blur 绑定），所以默认行为=原版。
#     """

#     if image_BGR is None:
#         return None

#     # 0) bilateral 开关：默认跟随 blur（保持原逻辑）
#     if enable_bilateral is None:
#         enable_bilateral = bool(blur)

#     # 1) 灰度
#     if enable_gray:
#         image_gray = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2GRAY)
#     else:
#         image_gray = image_BGR.copy()  # 若你想跳过灰度，可保留原图（注意后续步骤可能要求单通道）

#     # 2) resize（保持原先顺序：先 resize 再 ROI）
#     if enable_resize and image_size is not None:
#         image_gray = cv2.resize(image_gray, image_size)

#     # 3) ROI 裁剪（ROI 语义保持原版：x1,y1,x2,y2）
#     if enable_roi and ROI is not None:
#         x1, y1, x2, y2 = ROI
#         image_gray = image_gray[y1:y2, x1:x2]

#     # 4) Gaussian
#     if enable_gaussian:
#         kx, ky = int(gaussian_ksize[0]), int(gaussian_ksize[1])
#         # ksize 必须是奇数
#         if kx % 2 == 0: kx += 1
#         if ky % 2 == 0: ky += 1
#         image_gray = cv2.GaussianBlur(image_gray, (kx, ky), float(gaussian_sigma))

#     # 5) bg_sub 亮度均衡
#     if enable_bg_sub:
#         image_gray = gscv.uniformize_brightness(
#             image_gray,
#             method=bg_method,
#             clip_limit=float(bg_clip_limit),
#             tile_grid=int(bg_tile_grid),
#         )

#     # 6) 锐化
#     if enable_sharpen:
#         image_gray = gscv.imageadjustsharpen(image_gray, int(sharpen_amount))

#     # 7) 柔化（原 blur 行为）
#     if enable_bilateral:
#         image_gray = cv2.bilateralFilter(
#             image_gray,
#             int(bilateral_d),
#             float(bilateral_sigmaColor),
#             float(bilateral_sigmaSpace),
#         )

#     # 8) 对比度
#     if enable_contrast and contrast is not None and float(contrast) != 1.0:
#         image_gray = gscv.imageadjustContrast(image_gray, float(contrast))

#     return image_gray

def main():
    # 取得模板图标准尺寸
    image_tmpl = cv2.imread(r"C:\Users\Kabbw\Desktop\Projects\1203_iriso_anomalib\template.jpg", cv2.IMREAD_COLOR_BGR)
    w, h = image_tmpl.shape[:2]
    print(w,h)
    # 搜索当前文件夹
    pattern = os.path.join(r"C:\Users\Kabbw\Desktop\Projects\1203_iriso_anomalib\DATA\ORG\After", "*.jpg")
    jpg_files = glob.glob(pattern)
    # 逐文件进行分析处理
    for index, image_path in enumerate(jpg_files):
        # 取得图像名称
        filename = gscommon.get_filename_without_extension(image_path)
        # 加载对象文件
        image_BGR = cv2.imread(image_path, cv2.IMREAD_COLOR_BGR)
        # 图像预处理
        start = time.time()
        
        # 无pin
        image_result = imagepro_for_AIdetect(image_BGR, image_size=(h, w), contrast=1.2, ROI=None)
        print(f"{index + 1:04d} \t {filename} \t Elapsed time: {(time.time() - start):0.5f}s")
        # 输出到文件
        output_image_path = rf"C:\Users\Kabbw\Desktop\Projects\1203_iriso_anomalib\DATA\ORG\After\{filename}_11.jpg"
        cv2.imwrite(output_image_path, image_result, [cv2.IMWRITE_JPEG_QUALITY, 100])
        # Debug 用
        # gscv.imageShow("result", image_result, 1600)
        # cv2.waitKey(0)
    print("Completed")


if __name__ == "__main__":
    # DIR_filelist = r"D:\IRISO\1201\OK"           # 原始分类好的大目录
    # DIR_from = r"D:\IRISO\TestData\After\OK"     # 平铺的图片文件
    # DIR_to = r"D:\IRISO\1202\OK"                 # 需要自动生成的目标目录（不存在也没关系）
    #
    # pattern = os.path.join(DIR_filelist, "*.jpg")
    # jpg_files = glob.glob(pattern)
    # for index, image_path in enumerate(jpg_files):
    #     # 取得图像名称
    #     filename = gscommon.get_filename_without_extension(image_path)
    #     path_from = os.path.join(DIR_from, filename + ".jpg")
    #     path_to = os.path.join(DIR_to, filename + ".jpg")
    #     shutil.copy(path_from, path_to)
    #     print(f"{index:04d}  {filename}")
    # print(f"Copy completed! file count: {index}")

    main()