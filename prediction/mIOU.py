import numpy as np
import cv2
import os
from PIL import Image

class MIouCalculator:
    def __init__(self, mask_dir, pred_dir):
        self.mask_dir = mask_dir
        self.pred_dir = pred_dir

    def calculate_miou(self,pred_path, mask_path):
    # 读取预测图像和标签图像并转换为 NumPy 数组
        pred_image = np.array(Image.open(pred_path))
        mask_image = np.array(Image.open(mask_path))

    # 预测图像和标签图像均为黑白图像，因此我们将其转换为二进制图像（0和1）
        pred_image = (pred_image > 0).astype(np.uint8)
        mask_image = (mask_image > 0).astype(np.uint8)

    # 计算交集和并集
        intersection = np.logical_and(pred_image, mask_image).sum()
        union = np.logical_or(pred_image, mask_image).sum()

    # 计算每个类别的 IoU
        iou = intersection / (union + 1e-10)

    # 计算 mIOU
        mIOU = np.mean(iou) * 100.0

        return mIOU

    def compute_miou(self):
        mask_dir = os.listdir(self.mask_dir)
        pred_dir = os.listdir(self.pred_dir)

        all_miou = []
        for filename in os.listdir(self.pred_dir):
            if filename.endswith(".png"):
                pred_path = os.path.join(self.pred_dir, filename)
                mask_path = os.path.join(self.mask_dir, filename)
                # 计算单张图像的 mIOU
                mIOU = self.calculate_miou(pred_path, mask_path)  # 假设只有两类，即黑色和白色
                all_miou.append(mIOU)
        
                print(f"{filename}: mIOU = {mIOU:.2f}")
        
        average_miou = np.mean(all_miou)
        return  average_miou
        

