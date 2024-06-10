import os
import numpy as np
from paddle.io import Dataset
from paddle.vision import transforms
from PIL import Image
# 训练集图像预处理：缩放裁剪、图像增强、转 Tensor、归一化
train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                     ])

# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(224),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])

# 自定义数据集
class MyDataset(Dataset):
    """
    步骤一：继承 paddle.io.Dataset 类
    """

    def __init__(self, data_dir, label_path, transform=None):
        """
        步骤二：实现 __init__ 函数，初始化数据集，将样本和标签映射到列表中
        """
        super(MyDataset, self).__init__()
        self.data_list = []
        with open(label_path) as f:
            for line in f.readlines():
                image_path, label = line.strip().split('\t')
                im_1,im2 = image_path.split('\\')
                image_path = data_dir+"/"+im_1+"/"+im2
                # print(image_path)
                self.data_list.append([image_path, label])
                # print(image_path)
        # 传入定义好的数据处理方法，作为自定义数据集类的一个属性
        # 属性唯一化
        self.transform = transform

    def __getitem__(self, index):
        """
        步骤三：实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        # 根据索引，从列表中取出一个图像
        image_path, label = self.data_list[index]
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # 飞桨训练时内部数据格式默认为float32，将图像数据格式转换为 float32
        image = self.transform(image)
        image = image.astype('float32')
        # CrossEntropyLoss要求label格式为int，将Label格式转换为 int
        label = np.array([label], dtype="int64")        # 返回图像和对应标签
        return image, label

    def __len__(self):
        """
        步骤四：实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)


