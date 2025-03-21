import numpy as np
from scipy import ndimage
from scipy.ndimage import convolve, distance_transform_edt as bwdist
# import math
import torch
import numpy
class cal_fm_(object):
    def __init__(self, num, thds=255):
        # TruePositive + TrueNegative, for accuracy
        self.tp_fp = 0  
        # TruePositive 
        self.tp = 0      
        # Number of '1' predictions, for precision
        self.pred_true = 0  
        # Number of '1's in gt mask, for recall 
        self.gt_true = 0   
        # List to save mean absolute error of each image  
        self.mae_list = [] 
        self.img_size = 384  
        self.fscore, self.cnt, self.number   =  0, 0, 256
        self.mean_pr, self.mean_re, self.threshod = 0, 0, np.linspace(0, 1, self.number, endpoint=False)
        self.num = num
        self.thds = thds
        self.precision = np.zeros((self.num, self.thds))
        self.recall = np.zeros((self.num, self.thds))
        self.meanF = np.zeros((self.num,1))
        self.idx = 0
        # self.num_black = 0
        self.max_F = 0
    def update(self, pred, gt):
        
        # prediction, recall, Fmeasure_temp = self.cal(pred, gt)
        # self.precision[self.idx, :] = prediction
        # self.recall[self.idx, :] = recall
        # self.meanF[self.idx, :] = Fmeasure_temp
        
        self.idx += 1
    def cal(self, pred, gt):
        
        # self.tp += numpy.dot(pred, gt).sum()
        self.tp += (pred*gt).sum()
        self.pred_true += pred.sum()
        self.gt_true += gt.sum()

        # ae = torch.mean(torch.abs(res - gt), dim=(0, 1)).cpu().numpy()
        # mae_list.extend(ae)
        # mae_list.append(ae.item())
        self.cnt += 1
        
    def show(self):
        precision = self.tp / self.pred_true 
        recall = self.tp / self.gt_true
        avgf = (precision*recall*(1.3))/(0.3*precision+recall+1e-12)
        return avgf.max(),avgf,precision,recall   
        
class cal_fm(object):
    # Fmeasure(maxFm,meanFm)---Frequency-tuned salient region detection(CVPR 2009)
    def __init__(self, num, thds=255):
        self.num = num
        self.thds = thds
        self.precision = np.zeros((self.num, self.thds))
        self.recall = np.zeros((self.num, self.thds))
        self.meanF = np.zeros((self.num,1))
        self.idx = 0
        self.num_black = 0
        self.max_F = 0

    # def update(self, pred, gt):
    #     if gt.max() != 0:
    #         prediction, recall, Fmeasure_temp = self.cal(pred, gt)
    #         self.precision[self.idx, :] = prediction
    #         self.recall[self.idx, :] = recall
    #         self.meanF[self.idx, :] = Fmeasure_temp
    #     else:
    #         self.meanF[self.idx, :] = 1 # 让全黑图片F值为1
    #     self.idx += 1
    def update(self, pred, gt):
        if gt.max() != 0:
            self.num_black += 1
        prediction, recall, Fmeasure_temp = self.cal(pred, gt)
        self.precision[self.idx, :] = prediction
        self.recall[self.idx, :] = recall
        self.meanF[self.idx, :] = Fmeasure_temp
        
        self.idx += 1

    def cal(self, pred, gt):
######################## meanF ##############################
        th = 2 * pred.mean()
        if th > 1:
            th = 1
            # 归一化? 1为前景,0为后景
        binary = np.zeros_like(pred)
        binary[pred >= th] = 1
        hard_gt = np.zeros_like(gt)
        hard_gt[gt > 0.5] = 1
        # tp:正确预测数量
        tp = (binary * hard_gt).sum()
        if tp == 0:
            if hard_gt.all() == 0:
                meanF = 1
            else:
                meanF = 0 
        else:
            # 计算precision和recall
            pre = tp / binary.sum()
            rec = tp / hard_gt.sum()
            # beta^2 直接设置
            meanF = (1.3 * pre * rec) / (0.3 * pre + rec + 1e-8)
            if meanF > self.max_F and meanF != 1:
                self.max_F = meanF
######################## maxF ##############################
        # 从[0,1]到[0,255]
        pred = np.uint8(pred * 255)
        # onehot编码?
        target = pred[gt > 0.5]
        nontarget = pred[gt <= 0.5]
        # 这两行代码分别计算了目标和非目标区域中预测值的直方图。
        # np.histogram 函数用于计算数值数据的直方图，这里将预测值划分为 256 个区间，
        # 然后统计每个区间内的值的数量，从而得到直方图
        targetHist, _ = np.histogram(target, bins=range(256))
        nontargetHist, _ = np.histogram(nontarget, bins=range(256))
        # 这两行代码分别对目标和非目标区域中的直方图进行了累积和操作。
        # 首先，np.flip 函数用于翻转直方图的顺序，
        # 然后 np.cumsum 函数计算了翻转后直方图的累积和。
        targetHist = np.cumsum(np.flip(targetHist), axis=0)
        nontargetHist = np.cumsum(np.flip(nontargetHist), axis=0)

        precision = (targetHist) / (targetHist + nontargetHist + 1e-8)
        recall = (targetHist) / (np.sum(gt)+1e-8)
        # F = (1.3 * precision * recall) / (0.3 * precision + recall + 1e-10)
        # if F.max() > self.max_F:
        #     self.max_F = F
        return precision, recall, meanF

    def show(self):
        assert self.num == self.idx
        precision = self.precision.mean(axis=0)
        recall = self.recall.mean(axis=0)
        # if precision == 0 and recall == 0:
        #     fmeasure = 1
        fmeasure = (1.3 * precision * recall) / (0.3 * precision + recall)
        fmeasure_avg = self.meanF.mean(axis=0)
        # return fmeasure.max(),fmeasure_avg[0],precision,recall
        return fmeasure.max(),fmeasure_avg[0],precision,recall
    # def show(self):
    #     assert self.num == self.idx
    #     # precision = -np.partition(-self.precision, 1)[1]
    #     # recall = -np.partition(-self.recall, 1)[1]
    #     precision = self.precision.mean(axis=0)
    #     recall = self.recall.mean(axis=0)
    #     # precision = self.precision
    #     # recall = self.recall
    #     # fmeasure = (1.3 * precision * recall + 1e-10) / (0.3 * precision + recall + 1e-10)
    #     # fmeasure_max = -np.partition(-fmeasure, 1)[1]
    #     fmeasure_avg = self.meanF.mean(axis=0)
    #     # return fmeasure.max(),fmeasure_avg[0],precision,recall
    #     return self.max_F,fmeasure_avg[0],precision,recall


class cal_mae(object):
    # mean absolute error
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, pred, gt):
        return np.mean(np.abs(pred - gt))

    def show(self):
        return np.mean(self.prediction)


class cal_sm(object):
    # Structure-measure: A new way to evaluate foreground maps (ICCV 2017)
    def __init__(self, alpha=0.5):
        self.prediction = []
        self.alpha = alpha

    def update(self, pred, gt):
        gt = gt > 0.5
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def show(self):
        return np.mean(self.prediction)

    def cal(self, pred, gt):
        y = np.mean(gt)
        if y == 0:
            score = 1 - np.mean(pred)
        elif y == 1:
            score = np.mean(pred)
        else:
            # 解决score返回nan问题
            region_value = self.region(pred, gt)
            if np.isnan(region_value):
                region_value = 0
            score = self.alpha * self.object(pred, gt) + (1 - self.alpha) * region_value
        return score

    def object(self, pred, gt):
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)

        u = np.mean(gt)
        return u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, np.logical_not(gt))

    def s_object(self, in1, in2):
        x = np.mean(in1[in2])
        sigma_x = np.std(in1[in2])
        return 2 * x / (pow(x, 2) + 1 + sigma_x + 1e-8)

    def region(self, pred, gt):
        [y, x] = ndimage.center_of_mass(gt)
        y = int(round(y)) + 1
        x = int(round(x)) + 1
        [gt1, gt2, gt3, gt4, w1, w2, w3, w4] = self.divideGT(gt, x, y)
        pred1, pred2, pred3, pred4 = self.dividePred(pred, x, y)

        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)

        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def divideGT(self, gt, x, y):
        h, w = gt.shape
        area = h * w
        LT = gt[0:y, 0:x]
        RT = gt[0:y, x:w]
        LB = gt[y:h, 0:x]
        RB = gt[y:h, x:w]

        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = (h - y) * (w - x) / area

        return LT, RT, LB, RB, w1, w2, w3, w4

    def dividePred(self, pred, x, y):
        h, w = pred.shape
        LT = pred[0:y, 0:x]
        RT = pred[0:y, x:w]
        LB = pred[y:h, 0:x]
        RB = pred[y:h, x:w]

        return LT, RT, LB, RB

    def ssim(self, in1, in2):
        in2 = np.float32(in2)
        h, w = in1.shape
        N = h * w

        x = np.mean(in1)
        y = np.mean(in2)
        sigma_x = np.var(in1)
        sigma_y = np.var(in2)
        sigma_xy = np.sum((in1 - x) * (in2 - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + 1e-8)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0

        return score

class cal_em(object):
    #Enhanced-alignment Measure for Binary Foreground Map Evaluation (IJCAI 2018)
    def __init__(self):
        self.prediction = []

    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)

    def cal(self, pred, gt):
        th = 2 * pred.mean()
        if th > 1:
            th = 1
        FM = np.zeros(gt.shape)
        FM[pred >= th] = 1
        FM = np.array(FM,dtype=bool)
        GT = np.array(gt,dtype=bool)
        dFM = np.double(FM)
        if (sum(sum(np.double(GT)))==0):
            enhanced_matrix = 1.0-dFM
        elif (sum(sum(np.double(~GT)))==0):
            enhanced_matrix = dFM
        else:
            dGT = np.double(GT)
            align_matrix = self.AlignmentTerm(dFM, dGT)
            enhanced_matrix = self.EnhancedAlignmentTerm(align_matrix)
        [w, h] = np.shape(GT)
        score = sum(sum(enhanced_matrix))/ (w * h - 1 + 1e-8)
        return score
    def AlignmentTerm(self,dFM,dGT):
        mu_FM = np.mean(dFM)
        mu_GT = np.mean(dGT)
        align_FM = dFM - mu_FM
        align_GT = dGT - mu_GT
        align_Matrix = 2. * (align_GT * align_FM)/ (align_GT* align_GT + align_FM* align_FM + 1e-8)
        return align_Matrix
    def EnhancedAlignmentTerm(self,align_Matrix):
        enhanced = np.power(align_Matrix + 1,2) / 4
        return enhanced
    def show(self):
        return np.mean(self.prediction)




class cal_wfm(object):
    # How to Evaluate Foreground Maps? W_Fm
    def __init__(self, beta=1):
        self.beta = beta
        self.eps = 1e-6
        self.scores_list = []

    def update(self, pred, gt):
        assert pred.ndim == gt.ndim and pred.shape == gt.shape
        assert pred.max() <= 1 and pred.min() >= 0
        assert gt.max() <= 1 and gt.min() >= 0

        gt = gt > 0.5
        
        if gt.max() == 0:
            score = 1
        else:
            score = self.cal(pred, gt)
        self.scores_list.append(score)

    def matlab_style_gauss2D(self, shape=(7, 7), sigma=5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def cal(self, pred, gt):
        # [Dst,IDXT] = bwdist(dGT);
        '''
        bwdist 函数通常用于计算图像中每个像素到最近的零像素（背景像素）的欧几里得距离。
        在这个函数中，gt 代表了一个二值化图像，其中包含了两个值，
        通常是0（表示背景）和1（表示前景或目标）。
        gt == 0：这部分代码创建一个布尔掩码，
        其中与 gt 中的像素值为0的像素对应的位置为True，
        其他位置为False。这实际上是一个背景像素的二值掩码。

        bwdist(gt == 0, return_indices=True)：这是调用 bwdist 函数的语法。它接受一个布尔掩码作为输入，并计算每个前景像素到最近的背景像素的欧几里得距离。同时，return_indices=True 参数告诉函数返回距离图像和距离图像上每个像素的索引。

        结果解释：

        Dst 是一个与输入图像 gt 具有相同形状的数组，其中每个像素的值表示该像素到最近的背景像素的欧几里得距离。对于背景像素本身，距离为0。

        Idxt 是一个与输入图像 gt 具有相同形状的数组，其中每个像素的值表示该像素在距离图像中对应的最近背景像素的位置索引
        。这个索引可以用于找到距离最近的背景像素的位置。[坐标]
        '''
        Dst, Idxt = bwdist(gt == 0, return_indices=True)
        '''
            Idxt.shape = (2,352,352)
        '''
        # %Pixel dependency
        # E = abs(FG-dGT);
        E = np.abs(pred - gt)
        # Et = E;
        # Et(~GT)=Et(IDXT(~GT)); %To deal correctly with the edges of the foreground region
        Et = np.copy(E)
        '''
        这是关键的一步，用于调整误差。首先，它检查图像中哪些像素属于背景（gt == 0），
        然后使用距离图 Dst 中的信息来调整这些背景像素的误差。

        gt == 0 用于获取背景像素的掩码。

        Idxt[0][gt == 0], Idxt[1][gt == 0] 是根据之前计算的距离图中的索引信息，
        获取背景像素到最近的前景像素的索引。Idxt[0] 包含了纵向索引，Idxt[1] 包含了横向索引。

        Et[gt == 0] = ... 用计算出的索引来更新背景像素的误差，
        将背景像素的误差替换为距离最近前景像素的误差。
        这个操作的目的是考虑像素之间的空间关系，
        使背景像素的误差受到其周围前景像素的影响，以便更好地处理前景区域的边缘。
        '''

        Et[gt == 0] = Et[Idxt[0][gt == 0], Idxt[1][gt == 0]]

        # K = fspecial('gaussian',7,5);
        # EA = imfilter(Et,K);
        # MIN_E_EA(GT & EA<E) = EA(GT & EA<E);

        K = self.matlab_style_gauss2D((7, 7), sigma=5)
        EA = convolve(Et, weights=K, mode='constant', cval=0)
        MIN_E_EA = np.where(gt & (EA < E), EA, E)

        # %Pixel importance
        # B = ones(size(GT));
        # B(~GT) = 2-1*exp(log(1-0.5)/5.*Dst(~GT));
        # Ew = MIN_E_EA.*B;
        '''
        print(np.where(a > 5, 1, -1))
        //array([-1, -1, -1, -1, -1, -1,  1,  1,  1,  1])
        '''
        B = np.where(gt == 0, 2 - np.exp(np.log(0.5) / 5 * Dst), np.ones_like(gt))
        Ew = MIN_E_EA * B

        # TPw = sum(dGT(:)) - sum(sum(Ew(GT)));
        # FPw = sum(sum(Ew(~GT)));
        TPw = np.sum(gt) - np.sum(Ew[gt == 1])
        FPw = np.sum(Ew[gt == 0])

        # R = 1- mean2(Ew(GT)); %Weighed Recall
        # P = TPw./(eps+TPw+FPw); %Weighted Precision
        R = 1 - np.mean(Ew[gt])
        P = TPw / (self.eps + TPw + FPw)

        # % Q = (1+Beta^2)*(R*P)./(eps+R+(Beta.*P));
        Q = ((1 + self.beta) * R * P + 1e-8) / (self.eps + R + self.beta * P + 1e-8)

        return Q

    def show(self):
        return np.mean(self.scores_list)