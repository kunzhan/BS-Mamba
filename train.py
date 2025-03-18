import argparse
import logging
import os
import pprint
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import SGD
from torch.utils.data import DataLoader

import yaml
# import sys
import datetime
from tensorboardX import SummaryWriter

from dataset.data import BSDataset
from baseline.BS_Mamba import BS_Mamba
# from baseline.mamba_unet import MambaUnet
# from baseline.unet import UNet
# from baseline.local_vmamba import UPerNet
from DiceLoss import DiceLoss,IouLoss
from util.evaluate import evaluate_add
from util.utils import count_params, init_log
import random


parser = argparse.ArgumentParser(description='Black_soil_detection_net')
parser.add_argument('--gpu', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--config', default="./configs/BlackSoil.yaml", type=str)
parser.add_argument('--save-path', default="./result/", type=str)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.enabled = True
    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True  


def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    model_name = 'BS_Mamba'# UNet\MambaUnet\...

    results_file = args.save_path  + "results_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0


    rank = 0
    
    if rank == 0:
        logger.info('{}\n'.format(pprint.pformat(cfg)))

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
    init_seeds(0, False)

    model =BS_Mamba()
#   model = UNet()
#   model = MambaUnet()
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(  
        params_to_optimize, 
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.0005
    )
    # optimizer = torch.optim.SGD(param_list, lr=cfg['lr'],
    #                             momentum=args.momentum, weight_decay=args.weight_decay)
    model.cuda()
    criterion_ce = nn.CrossEntropyLoss().cuda()
    criterion_iou = IouLoss(reduction='mean')
    Trainset1 = BSDataset(cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'])
    Trainset2 = BSDataset(cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'])
    Valset = BSDataset(cfg['dataset'], cfg['data_root'], 'val')

    Trainloader1 = DataLoader(Trainset1, batch_size=cfg['batch_size'],
                               pin_memory=False, num_workers=0, drop_last=True, sampler=None)
    Trainloader2 = DataLoader(Trainset2, batch_size=cfg['batch_size'],
                               pin_memory=False, num_workers=0, drop_last=True, sampler=None)
    Valloader = DataLoader(Valset, batch_size=1, pin_memory=True, num_workers=4,
                           drop_last=True, sampler=None)
    
    total_iters = len(Trainloader1) * cfg['epochs']
    previous_best = 0.0
    writer = {'loss_tra' :SummaryWriter('./result/loss_tra'),'loss_val' :SummaryWriter('./result/loss_val')}
    writer_iou = {'val_iou' : SummaryWriter('./result/val_iou')}


    for epoch in range(cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.6f}, Previous best: {:.6f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss = 0.0


        if rank == 0:
            tbar = tqdm(total=len(Trainloader1),desc=f'Epoch {epoch}')
        loader = zip(Trainloader1, Trainloader2)
        for i, ((img,mask,cutmix_box),(img_mix,mask_mix,_)) in enumerate(loader):
            img, mask = img.cuda(),mask.cuda()
            img_mix, mask_mix = img_mix.cuda(),mask_mix.cuda()
            cutmix_box = cutmix_box.cuda()
            img[cutmix_box.unsqueeze(1).expand(img.shape) == 1] = \
                img_mix[cutmix_box.unsqueeze(1).expand(img.shape) == 1]
            mask[cutmix_box == 1] = mask_mix[cutmix_box == 1]
            
            model.train()
            pre = model(img)

            loss_ce =  criterion_ce(pre, mask)

            loss_iou = criterion_iou(pre, mask)

            loss = 0.5*loss_ce + 0.25*loss_iou # lamda_1 and lamda_2

           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            writer['loss_tra'].add_scalar('Loss/Total', total_loss / (i + 1), epoch)
            iters = epoch * len(Trainloader1) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr

            if rank == 0:
                 tbar.update(1)
                 tbar.set_description(' Loss: {:.3f} '
                                     .format(
                    total_loss / (i + 1)
                    ))
            

        if rank == 0:
            tbar.close()

        res_val = evaluate_add(model, Valloader)

        class_IOU = res_val['iou_class']
        mIOU = res_val["mean_mIOU"]
        f1_or_dsc = res_val['f1_or_dsc']
        accuracy = res_val['accuracy']
        sensitivity = res_val['sensitivity']
        specificity = res_val['specificity']
        loss_val = res_val['Loss_val']

        writer['loss_val'].add_scalar('Loss/Total', loss_val, epoch)
        writer_iou['val_iou'].add_scalar('val_iou', mIOU, epoch)

        if rank == 0:
            logger.info('***** Evaluation***** >>>> mIOU: {:.6f} \n'.format(mIOU))

        with open(results_file, "a") as f:
             train_info = f"[epoch: {epoch}]\n" \
                          f"train_loss: {total_loss / (i + 1):.4f}\n" \
                          f"lr: {lr:.6f}\n" \
                          f"val_mIOU: {mIOU} \n" \
                          f"val_class_IOU: {class_IOU}\n" \
                          f"val_mean_mIOU: {mIOU} \n" \
                          f"f1_or_dsc: {f1_or_dsc:.6f}\n" \
                          f"accuracy: {accuracy:.6f}\n" \
                          f"sensitivity: {sensitivity:.6f}\n" \
                          f"specificity: {specificity:.6f}\n" \
                          f"Loss_val: {loss_val:.4f}\n"

             f.write(train_info + "\n\n")


        if mIOU > previous_best and rank == 0:
            if previous_best != 0:
                os.remove(os.path.join(args.save_path , '%s_%.3f.pth' % (model_name, previous_best)))
            previous_best = mIOU
            torch.save(model.state_dict(), os.path.join(args.save_path, '%s_%.3f.pth' % (cfg['backbone'], mIOU)))  


    
if __name__ == '__main__':
    main()
