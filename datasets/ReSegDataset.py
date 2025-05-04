# import os
# from utils.config import Config
# import torch
# from torch.utils.data import Dataset
# from torchvision.transforms import transforms
# from PIL import Image
# import numpy as np
# import yaml
# import random

# def load_config(config_path):
#     with open(config_path, 'r') as file:
#         config = yaml.safe_load(file)
#     return config

# opt = load_config('options/training.yaml')

# class ReSegDataset(Dataset):
#     def __init__(self, mode='train'):
#         super(ReSegDataset, self).__init__()
#         self.mode = mode

#         if mode == 'train':
#             self.data_images = opt['datasets']['train']['data_images']
#             self.data_annotation_mask = opt['datasets']['train']['data_annotation_mask']
#         elif mode == 'val':
#             self.data_images = opt['datasets']['val']['data_images']
#             self.data_annotation_mask = opt['datasets']['val']['data_annotation_mask']
#         else:
#             raise ValueError("Mode should be 'train' or 'val'")

#         # 获取所有图像文件名列表
#         self.image_filenames = sorted([os.path.join(self.data_images, x) for x in os.listdir(self.data_images) if x.lower().endswith(('.png', '.jpg', '.jpeg'))])
#         self.mask_filenames = sorted([os.path.join(self.data_annotation_mask, x) for x in os.listdir(self.data_annotation_mask) if x.lower().endswith(('.png', '.jpg', '.jpeg'))])

#         if len(self.image_filenames) != len(self.mask_filenames):
#             raise ValueError("图像数量和掩码数量不匹配")

#         # 定义数据转换
#         self.rgb_transform = transforms.Compose([
#             transforms.Resize((opt['train']['train_size'], opt['train']['train_size'])),
#             transforms.ToTensor(),
#         ])

#         self.binary_transform = transforms.Compose([
#             transforms.Resize((opt['train']['train_size'], opt['train']['train_size'])),
#             transforms.ToTensor(),
#         ])

#     def __len__(self):
#         return len(self.image_filenames)

#     def __getitem__(self, index):
#         image_path = self.image_filenames[index]
#         mask_path = self.mask_filenames[index]

#         # 加载图像和二值掩码
#         image = Image.open(image_path).convert('RGB')
#         mask = Image.open(mask_path).convert('1')  # 保持二值图像

#         if self.mode == 'train':
#             # 应用随机裁剪
#             h, w = image.size[::-1]
#             patch_size = opt['train']['train_size']
#             if h < patch_size or w < patch_size:
#                 raise ValueError(f"图像尺寸小于裁剪尺寸: {image_path}")
#             top = random.randint(0, h - patch_size)
#             left = random.randint(0, w - patch_size)
#             image = image.crop((left, top, left + patch_size, top + patch_size))
#             mask = mask.crop((left, top, left + patch_size, top + patch_size))

#             # 数据增强
#             if random.random() < 0.5:
#                 image = image.transpose(Image.FLIP_LEFT_RIGHT)
#                 mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

#             if random.random() < 0.5:
#                 image = image.transpose(Image.FLIP_TOP_BOTTOM)
#                 mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

#             if random.random() < 0.5:
#                 rotation_angle = random.choice([90, 180, 270])
#                 image = image.rotate(rotation_angle)
#                 mask = mask.rotate(rotation_angle)

#         # 转换为 PyTorch tensor
#         image = self.rgb_transform(image)
#         mask = self.binary_transform(mask)

#         return image, mask


import os
import yaml
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


opt = load_config("options/training.yaml")


class ReSegDataset(Dataset):
    def __init__(self, data_images, data_annotation_mask, train_size):
        super(ReSegDataset, self).__init__()

        self.data_images = data_images
        self.data_annotation_mask = data_annotation_mask

        self.image_filenames = sorted(
            [
                os.path.join(self.data_images, x)
                for x in os.listdir(self.data_images)
                if x.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        self.mask_filenames = sorted(
            [
                os.path.join(self.data_annotation_mask, x)
                for x in os.listdir(self.data_annotation_mask)
                if x.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )

        if len(self.image_filenames) != len(self.mask_filenames):
            raise ValueError("图像数量和掩码数量不匹配")

        self.rgb_transform = transforms.Compose(
            [transforms.Resize((train_size, train_size)), transforms.ToTensor()]
        )

        self.binary_transform = transforms.Compose(
            [
                transforms.Resize((train_size, train_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_path = self.image_filenames[index]
        mask_path = self.mask_filenames[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 保持二值图像

        h, w = image.size[::-1]
        patch_size = opt["train"]["train_size"]
        if h < patch_size or w < patch_size:
            raise ValueError(f"图像尺寸小于裁剪尺寸: {image_path}")
        top = random.randint(0, h - patch_size)
        left = random.randint(0, w - patch_size)
        image = image.crop((left, top, left + patch_size, top + patch_size))
        mask = mask.crop((left, top, left + patch_size, top + patch_size))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        if random.random() < 0.5:
            rotation_angle = random.choice([90, 180, 270])
            image = image.rotate(rotation_angle)
            mask = mask.rotate(rotation_angle)

        image = self.rgb_transform(image)
        mask = self.binary_transform(mask)

        return image, mask
