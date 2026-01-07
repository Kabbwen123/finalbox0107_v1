import cv2
import numpy as np
import torch
# from SuperGluePretrainedNetwork_master.models.matching import Matching
# from SuperGluePretrainedNetwork_master.models.utils import frame2tensor



# class SuperGlueMatcher:
#     def __init__(self, superpoint_path, superglue_path, device='cuda'):
#         self.device = device if torch.cuda.is_available() else 'cpu'
#         config = {
#             'superpoint': {
#                 'nms_radius': 3,
#                 'keypoint_threshold': 0.004,
#                 'max_keypoints': 2048
#             },
#             'superglue': {
#                 'weights': 'indoor',
#                 'sinkhorn_iterations': 20,
#                 'match_threshold': 0.2,
#             }
#         }
#         self.matching = Matching(config).eval().to(self.device)
#         self.matching.superpoint.load_state_dict(torch.load(superpoint_path, map_location=self.device))
#         self.matching.superglue.load_state_dict(torch.load(superglue_path, map_location=self.device))


#     def match(self, img_tmpl_gray, img_src_gray):
#         inp0 = frame2tensor(img_tmpl_gray, self.device)
#         inp1 = frame2tensor(img_src_gray, self.device)
#         # 查找匹配点对
#         pred = self.matching({'image0': inp0, 'image1': inp1})
#         kpts0 = pred['keypoints0'][0].detach().cpu().numpy()
#         kpts1 = pred['keypoints1'][0].detach().cpu().numpy()
#         matches = pred['matches0'][0].detach().cpu().numpy()
#         scores0 = pred['matching_scores0'][0].detach().cpu().numpy()
#         # 过滤掉非匹配点
#         valid = (matches > -1) & (scores0 > 0.5)
#         mkpts0 = kpts0[valid]
#         mkpts1 = kpts1[matches[valid]]
#         if len(mkpts0) < 20:
#             print("Not enough matches.")
#             return None, None, None
#         # 计算变换矩阵
#         M, inliers = cv2.estimateAffinePartial2D(
#             mkpts1,  # scene points
#             mkpts0,  # template points
#             method=cv2.RANSAC,
#             ransacReprojThreshold=3,
#             maxIters=5000,
#             confidence=0.99
#         )
#         if M is None:
#             print("Similarity Transform estimation failed.")
#             return None, None, None
#         # 返回变换矩阵和KP序列
#         return M, mkpts0, mkpts1


class TemplateMatcher:
    def __init__(self, method="ORB"):
        if method.upper() == "ORB":
            self.detector = cv2.ORB_create(
                nfeatures=4000,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=31,
                patchSize=31
            )
            self.method = "ORB"
            self.norm_type = cv2.NORM_HAMMING
        elif method.upper() == "SIFT":
            self.detector = cv2.SIFT_create()
            self.method = "SIFT"
            self.norm_type = cv2.NORM_L2
        else:
            raise ValueError("Unsupported method. Choose ORB or SIFT.")
        # BFMatcher 配置
        self.matcher = cv2.BFMatcher(self.norm_type, crossCheck=False)


    def match(self, img_tmpl_gray, img_src_gray, ratio = 0.75):
        # 关键点特征
        kp_t, des_t = self.detector.detectAndCompute(img_tmpl_gray, None)
        kp_s, des_s = self.detector.detectAndCompute(img_src_gray, None)
        if des_t is None or des_s is None:
            print("Feature extraction failed.")
            return None, None, None
        # KNN关键点特征匹配
        knn = self.matcher.knnMatch(des_t, des_s, k=2)
        # Lowe Ratio Test
        good = []
        for m, n in knn:
            if m.distance < ratio * n.distance:
                good.append(m)
        if len(good) < 20:
            print("Not enough matches.")
            return None, None, None
        # 提取匹配后的点
        mkpts_t = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 2)
        mkpts_s = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 2)
        # 计算 similarity transform（仅 平移 + 旋转 + 缩放）
        M, inliers = cv2.estimateAffinePartial2D(
            mkpts_s,  # scene
            mkpts_t,  # template
            method=cv2.RANSAC,
            ransacReprojThreshold=3,
            maxIters=5000,
            confidence=0.99
        )
        if M is None:
            print("Similarity Transform estimation failed.")
            return None, None, None
        # 返回变换矩阵和KP序列
        return M, mkpts_t, mkpts_s