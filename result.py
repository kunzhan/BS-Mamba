import numpy as np
import os
from PIL import Image

class MIouCalculator:
    def __init__(self, mask_dir, pred_dir):
        self.mask_dir = mask_dir
        self.pred_dir = pred_dir

    def calculate_metrics(self, pred_image, mask_image):
        # 计算 TP, FP, TN, FN
        tp = np.sum((pred_image == 1) & (mask_image == 1))  # True Positive
        tn = np.sum((pred_image == 0) & (mask_image == 0))  # True Negative
        fp = np.sum((pred_image == 1) & (mask_image == 0))  # False Positive
        fn = np.sum((pred_image == 0) & (mask_image == 1))  # False Negative

        # 计算 Accuracy 和 F1 Score
        accuracy = float(tp + tn) / float(tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
        f1_score = float(2 * tp) / float(2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
        
        return accuracy, f1_score

    def calculate_miou(self, pred_image, mask_image):
        # 计算前景和背景的 IoU
        ious = {}
        metrics = {}

        for category, (pred_binary, mask_binary) in {"foreground": (pred_image > 0, mask_image > 0),
                                                     "background": (pred_image == 0, mask_image == 0)}.items():
            pred_binary = pred_binary.astype(np.uint8)
            mask_binary = mask_binary.astype(np.uint8)
            
            intersection = np.logical_and(pred_binary, mask_binary).sum()
            union = np.logical_or(pred_binary, mask_binary).sum()

            # 计算 IoU
            iou = intersection / (union + 1e-10)
            ious[category] = iou

            # 计算 Accuracy 和 F1 Score
            accuracy, f1_score = self.calculate_metrics(pred_binary, mask_binary)
            metrics[category] = {"accuracy": accuracy * 100.0, "f1_score": f1_score * 100.0}

        return ious, metrics

    def compute_miou(self):
        all_ious = {"foreground": [], "background": []}
        all_metrics = {"foreground": {"accuracy": [], "f1_score": []}, "background": {"accuracy": [], "f1_score": []}}

        for filename in os.listdir(self.pred_dir):
            if filename.endswith(".png"):
                pred_path = os.path.join(self.pred_dir, filename)
                mask_path = os.path.join(self.mask_dir, filename)

                # 读取图像
                pred_image = np.array(Image.open(pred_path).convert("L"))
                mask_image = np.array(Image.open(mask_path).convert("L"))

                # 计算前景和背景的 IoU、Accuracy 和 F1 Score
                ious, metrics = self.calculate_miou(pred_image, mask_image)

                for category in ["foreground", "background"]:
                    all_ious[category].append(ious[category] * 100.0)
                    all_metrics[category]["accuracy"].append(metrics[category]["accuracy"])
                    all_metrics[category]["f1_score"].append(metrics[category]["f1_score"])

        # 计算前景和背景的平均值
        average_ious = {category: np.mean(all_ious[category]) for category in all_ious}
        average_metrics = {category: {metric: np.mean(all_metrics[category][metric])
                                      for metric in all_metrics[category]}
                           for category in all_metrics}

        # 计算总体平均值（前景和背景结合）
        overall_average_miou = np.mean([average_ious["foreground"], average_ious["background"]])
        overall_average_accuracy = np.mean([average_metrics["foreground"]["accuracy"], average_metrics["background"]["accuracy"]])
        overall_average_f1_score = np.mean([average_metrics["foreground"]["f1_score"], average_metrics["background"]["f1_score"]])

        # 输出最终结果
        print("Average Metrics:")
        print(f"Foreground: mIoU = {average_ious['foreground']:.2f}, "
              f"Accuracy = {average_metrics['foreground']['accuracy']:.2f}, "
              f"F1 Score = {average_metrics['foreground']['f1_score']:.2f}")
        print(f"Background: mIoU = {average_ious['background']:.2f}, "
              f"Accuracy = {average_metrics['background']['accuracy']:.2f}, "
              f"F1 Score = {average_metrics['background']['f1_score']:.2f}")
        print(f"Overall: mIoU = {overall_average_miou:.2f}, "
              f"Accuracy = {overall_average_accuracy:.2f}, "
              f"F1 Score = {overall_average_f1_score:.2f}")
        
        print(f"Overall Average mIoU: {overall_average_miou:.2f}, "
              f"Accuracy: {overall_average_accuracy:.2f}, "
              f"F1 Score: {overall_average_f1_score:.2f}")

        return overall_average_miou, overall_average_accuracy, overall_average_f1_score

# 使用示例
mask_directory = "..." # 替换为真实掩码文件夹路径
pred_directory = "/home/mac/gdnet_12/testdata/gdnet_1_0.779/" # 替换为真实预测文件夹路径

miou_calculator = MIouCalculator(mask_directory, pred_directory)
average_miou, average_accuracy, average_f1_score = miou_calculator.compute_miou()
