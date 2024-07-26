import paddle
from paddle.io import Dataset, DataLoader
from PIL import Image
import numpy as np
from .common_dataset import create_operators
from ppcls.data.preprocess import transform as transform_func
import os

class AngleDataset(Dataset):
    def __init__(self, image_root, cls_label_path, transform_ops=None):
        self.image_root = image_root
        self.label_file = cls_label_path
        self.transform = create_operators(transform_ops)
        self.data = self.load_data()

    def load_data(self):
        with open(self.label_file, 'r') as f:
            lines = f.readlines()
        data = []
        for line in lines:
            parts = line.strip().split(',')
            image_path = os.path.join(self.image_root, parts[0].strip())
            roll = float(parts[1])
            yaw = float(parts[2])
            data.append((image_path, float(roll) / 360.0, float(yaw) / 360.0))  # 归一化角度
        return data

    def __getitem__(self, index):
        img_path, roll, yaw = self.data[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = transform_func(img, self.transform)
        return img, np.array([roll, yaw], dtype=np.float32)

    def __len__(self):
        return len(self.data)
