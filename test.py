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
from prediction.collage import CollageGenerator
from prediction.mIOU import MIouCalculator
from fortest.predict_multi import ImagePredictor
# from fortest.predict_multi import ckpt_path
def main():    
    previous_best=0.770# the value of result
    predictor = ImagePredictor(previous_best)
    image_predictor = predictor.main(previous_best)
    collage = CollageGenerator(image_folder=image_predictor,
    image_list_file="...",# the path to test set 
    output_folder="...",# the path to ouput 
    group_size=150,
    rows=10,
    cols=15)
    pre_collage = collage.create_and_save_collages()
    MIou = MIouCalculator(mask_dir="...", # the label of test
    pred_dir=pre_collage)

    miou= MIou.compute_miou()
    print("mIOU:", miou)

     
if __name__ == '__main__':
    main()