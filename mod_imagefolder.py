from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import re


class CustomImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = ImageFolder(root=root, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        path = self.dataset.imgs[idx][0]  # 获取图片文件路径
        class_number = re.search(r'class_(\d+)', path).group(1)  # 类别名称
        image_number = re.search(r'/(\d+).jpg', path).group(1)  # 在类别里的序号
        pos = class_number + '-' + image_number
        pos2 = (class_number, image_number)
        # pos3 = 1
        return img, label, pos, pos2

class CustomImageFolder2(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = ImageFolder(root=root, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        path = self.dataset.imgs[idx][0]  # 获取图片文件路径
        parts = path.split('_')
        class_number = re.search(r'class_(\d+)', path).group(1)  # 类别名称
        # image_number = re.search(r'\\(\d+)\.jpg', path).group(1)  # 在类别里的序号
        image_number = parts[1].split('/')[-1].split('.')[0]
        pos = class_number + '-' + image_number
        pos2 = (class_number, image_number)
        # 获得是a数据集还是b数据集
        is_a = parts[0].split('/')[2]
        return img, label, pos, pos2, is_a


# class CustomImageFolder2(Dataset):
#     def __init__(self, root, transform=None):
#         self.dataset = ImageFolder(root=root, transform=transform)
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         img, label = self.dataset[idx]
#         path = self.dataset.imgs[idx][0]  # 获取图片文件路径
#         parts = path.split('_')
#         class_number = re.search(r'class_(\d+)', path).group(1)  # 类别名称
#         # image_number = re.search(r'\\(\d+)\.jpg', path).group(1)  # 在类别里的序号
#         image_number = parts[-2].split('\\')[-1]
#         pos = class_number + '-' + image_number
#         pos2 = (class_number, image_number)
#         # 获得是a数据集还是b数据集
#         is_a = parts[-1].split('.')[0]
#         return img, label, pos, pos2, is_a
