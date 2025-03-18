import os
import cv2
import numpy as np

def load_images(image_folder, image_list):
    images = []
    for image_name in image_list:
        image_path = os.path.join(image_folder, f"{image_name}.png")
        image = cv2.imread(image_path)
        images.append(image)
    return images

def create_collage(images, rows, cols):
    row_images = []
    for i in range(0, len(images), cols):
        row = np.hstack(images[i:i+cols])
        row_images.append(row)
    collage = np.vstack(row_images)
    return collage

class CollageGenerator:
    def __init__(self, image_folder, image_list_file, output_folder, group_size=150, rows=10, cols=15):
        self.image_folder = image_folder
        self.image_list_file = image_list_file
        self.output_folder = output_folder
        self.group_size = group_size
        self.rows = rows
        self.cols = cols

    def create_and_save_collages(self):
        # 读取图像列表文件
        with open(self.image_list_file, 'r') as f:
            image_list = [line.strip() for line in f.readlines()]

        # 加载图像
        images = load_images(self.image_folder, image_list)

        # 将小图按照每 group_size 张一组分组
        grouped_images = [images[i:i+self.group_size] for i in range(0, len(images), self.group_size)]

        os.makedirs(self.output_folder, exist_ok=True)
        for idx, group in enumerate(grouped_images):
            first_image_name = image_list[idx * self.group_size]
            name_parts = first_image_name.split("_")[:2]
            collage_name = "_".join(name_parts)
            
            collage = create_collage(group, self.rows, self.cols)
            cv2.imwrite(os.path.join(self.output_folder, f"{collage_name}.png"), collage)
        return self.output_folder
if __name__ == "__main__":
    
    CollageGenerator()
