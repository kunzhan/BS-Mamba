import os
import time
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import yaml
import argparse
from baseline.gdnet import gdnet



class ImagePredictor:
    def __init__(self,previous_best):
        self.parser = argparse.ArgumentParser(description='Black_soil_detection_net')
        self.parser.add_argument('--model', default="415", type=str)
        self.parser.add_argument('--config', default="./configs/BlackSoil.yaml", type=str)
        self.previous_best = previous_best


    def main(self,previous_best):
        args = self.parser.parse_args()
        cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
        mIOU = previous_best

        mdl = '%s_%.3f' % ('gdnet', mIOU)# mdl = 'UNet_24' 


        dataset = 'test2_0.770'

        test_data ='/data/grassset2_8/sample2/image2_crop/'
        to_test = {'test':test_data}
        weights_path = "/home/mac/gdnet_1/result/" + mdl +'.pth'

        ckpt_path = './testdata/' + dataset + mdl

        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)

        # roi_mask_path = "./DRIVE/test/mask/01_test_mask.gif"
        assert os.path.exists(weights_path), f"weights {weights_path} not found."
        # assert os.path.exists(img_path), f"image {img_path} not found."
        # assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."
        # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        # mean = (0.709, 0.381, 0.224)  # Ori
        # std = (0.127, 0.079, 0.043)
        mean = (0.485, 0.456, 0.406) # Cor
        std = (0.229, 0.224, 0.225)
        # mean = (0.342, 0.413, 0.359)    # Trian
        # std = (0.085, 0.094, 0.091)

        # get devices
        device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
        print("using {} device.".format(device))

        # create model
        model = gdnet()


        # load weights
        # model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        '''
        # 加载权重参数
        state_dict = torch.load(weights_path, map_location='cpu')

        # 在加载的状态字典中查看权重参数的形状
        weight_shape = state_dict['in_conv.0.weight'].shape
        print("Weight shape:", weight_shape)

        # 使用加载的状态字典来加载模型的权重参数
        model.load_state_dict(state_dict)
        '''
        model.to(device)
        model.eval() 

        # load roi mask
        # roi_img = Image.open(roi_mask_path).convert('L')
        # roi_img = np.array(roi_img)
        '''
        original_img = Image.open(img_path).convert('RGB')

        # from pil image to tensor and normalize
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        '''

         # load image 
        with torch.no_grad():
            for name, root in to_test.items():
                '''
                name = test
                root = /home/ljs/code/cloud/AIR_CD/Test_Images_png
                '''
                # 获取图片名称list
                img_list = [os.path.splitext(f)[0] for f in os.listdir(root) if f.endswith('.png')]
                # 开始计时
                # t_start = self.time_synchronized()
                # 图像处理
                data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)])
                for idx, img_name in enumerate(img_list):
                    print ('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                    original_img = Image.open(os.path.join(root, img_name +'.png')).convert('RGB')
                    img = data_transform(original_img)

                    img = torch.unsqueeze(img, dim=0)
                    # init model？   
                    img_height, img_width = img.shape[-2:] 
                    init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                    model(init_img)

                    output = model(img.to(device))

                    prediction = output.argmax(1).squeeze(0)

                    prediction = prediction.to("cpu").numpy().astype(np.uint8)
                    # # 将前景对应的像素值改成255(白色)
                    prediction[prediction == 1] = 255
                    mask = Image.fromarray(prediction)
                    mask.save(os.path.join(ckpt_path, img_name + '.png'))


        output_value = ckpt_path
        return output_value



