import torch
import torch.nn as nn
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, ignore):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score[ignore != 1] * target[ignore != 1])
        y_sum = torch.sum(target[ignore != 1] * target[ignore != 1])
        z_sum = torch.sum(score[ignore != 1] * score[ignore != 1])
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True, ignore=None):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i], ignore)
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)
    
class IouLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(IouLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, inputs, targets, smooth=1):
        # 该代码是二分类代码
        """
            output : NxCxHxW Variable
            target : NxHxW LongTensor
        """
        # 如果inputs没有归一化可以先归一化
        inputs = torch.softmax(inputs,dim=1)
        # 因为loss计算的是pred和targets的正负样本交集并集，所以pred中的预测值（0~1）需要转为0和1的标签值
        # inputs从NxCxHxW装变为NxHxW，且里面不是预测值而是0和1标签值
        inputs = torch.argmax(inputs, 1).squeeze(0)# 大于0.5概率变0或者1
        # IOU公式计算
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        Iou_loss = 1- (intersection + smooth)/(union + smooth)# smooth防止分母为0
        
        if self.reduction == 'mean':
            return Iou_loss.mean()
        elif self.reduction == 'sum':
            return Iou_loss.sum()
        else:
            return Iou_loss
