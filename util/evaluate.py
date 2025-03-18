import torch
from torch import nn
import numpy as np
from sklearn.metrics import confusion_matrix
from DiceLoss import DiceLoss,IouLoss

def evaluate_add(model, loader):
    return_dict = {}  # 创建一个空字典，用于存储评估结果
    model.eval()  # 将模型设为评估模式
   # 确保 mode 参数在指定的三种模式中

    # 定义损失函数（交叉熵损失）
    criterion_ce = nn.CrossEntropyLoss()
    criterion_iou= IouLoss('mean')
    total_loss = 0  # 初始化总损失累加器
    mIOU_list = []  # 用于存储每批150张图像的平均 IoU 值
    f1_list = []  # 用于存储每批150张图像的平均 F1 值
    accuracy_list = []  # 用于存储每批150张图像的平均准确率
    sensitivity_list = []  # 用于存储每批150张图像的平均敏感度
    specificity_list = []  # 用于存储每批150张图像的平均特异性

    epoch_tn, epoch_fp, epoch_fn, epoch_tp = 0, 0, 0, 0  # 初始化全局累加器
    num_large_images = 0  # 用于记录大图的数量

    with torch.no_grad():  # 在评估阶段，禁用梯度计算
        # 遍历数据加载器中的批次数据
        for i, (img, mask, ids) in enumerate(loader):
            img = img.cuda()  # 将输入图像移动到 GPU
            res = model(img)  # 通过模型进行前向传播，得到预测结果
            pred = res.argmax(dim=1)  # 获取预测的类别标签（沿类别维度取最大值）
            pred_np = np.array(pred.cpu()).squeeze(axis=0).reshape(-1)  # 将预测结果转换为 numpy 数组
            gt_np = np.array(mask).squeeze(axis=0).reshape(-1)  # 将真实标签转换为 numpy 数组
            # 计算总损失
            loss_ce =  criterion_ce(res.cpu(), mask)
            loss_iou = criterion_iou(res.cpu(), mask)
            loss =  0.5*loss_iou + 0.5*loss_ce
            total_loss += loss.item() 

            confusion = confusion_matrix(gt_np, pred_np, labels=[0, 1])
            TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

            epoch_tn += TN
            epoch_fp += FP
            epoch_fn += FN
            epoch_tp += TP

            # 每150张小图计算一次总的指标
            if (i + 1) % 150 == 0:
                accuracy = float(epoch_tn + epoch_tp) / float(epoch_tn + epoch_fp + epoch_fn + epoch_tp) if float(epoch_tn + epoch_fp + epoch_fn + epoch_tp) != 0 else 0
                sensitivity = float(epoch_tp) / float(epoch_tp + epoch_fn) if float(epoch_tp + epoch_fn) != 0 else 0
                specificity = float(epoch_tn) / float(epoch_tn + epoch_fp) if float(epoch_tn + epoch_fp) != 0 else 0
                f1_or_dsc = float(2 * epoch_tp) / float(2 * epoch_tp + epoch_fp + epoch_fn) if float(2 * epoch_tp + epoch_fp + epoch_fn) != 0 else 0
                miou = float(epoch_tp) / float(epoch_tp + epoch_fp + epoch_fn) if float(epoch_tp + epoch_fp + epoch_fn) != 0 else 0

                mIOU_list.append(miou)  # 存储 mIOU
                f1_list.append(f1_or_dsc)  # 存储 F1 值
                accuracy_list.append(accuracy)  # 存储准确率
                sensitivity_list.append(sensitivity)  # 存储敏感度
                specificity_list.append(specificity)  # 存储特异性

                # 重置累加器
                epoch_tn, epoch_fp, epoch_fn, epoch_tp = 0, 0, 0, 0

        # 计算平均指标
        mean_mIOU = np.mean(mIOU_list)
        mean_f1 = np.mean(f1_list)
        mean_accuracy = np.mean(accuracy_list)
        mean_sensitivity = np.mean(sensitivity_list)
        mean_specificity = np.mean(specificity_list)
        loss = total_loss / len(loader)

    return_dict['iou_class'] = mIOU_list
    return_dict['mean_mIOU'] = mean_mIOU  # 所有批次的平均 mIOU
    return_dict['f1_or_dsc'] = mean_f1
    return_dict['accuracy'] = mean_accuracy
    return_dict['sensitivity'] = mean_sensitivity
    return_dict['specificity'] = mean_specificity
    return_dict['Loss_val'] = loss  # 整体损失值

    return return_dict


