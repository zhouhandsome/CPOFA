import os
import paddle
model_names = ['VGGNet', 'VisionTransformer', 'AlexNet', 'ResNet']

MODEL = "AlexNet"

image_size = [224,224]

# Cifar10,self_create
DATASET_CHOICE = "Cifar10"

modle_save_path = r"D:\VsCodeProjects\pythonProjects\Smart_Algorithm\model_params"
modle_save_path = os.path.join(modle_save_path,MODEL)

EPOCH = 4

BATCH_SIZE = 64
# 数据集文件夹路径
# 类别数目

classes = ["shuibei","bi","smartphone","Tshirt","watermaller"]
classes2label = {name:i for i,name in enumerate(classes) }
label2classes = {i:name for i,name in enumerate(classes)}
if DATASET_CHOICE == "self_create":
    CLASS_DIM = len(classes)
else:
    classes = [ 'airplane' , 'automobile' ,'bird' ,'cat','deer' ,'dog' ,'rog' , 'horse' , 'ship' , 'truck' ]
    CLASS_DIM = 10
dataset_dir = r'D:\VsCodeProjects\pythonProjects\Smart_Algorithm\dataset'
train_path = os.path.join(dataset_dir, 'train')
train_label_path = os.path.join(dataset_dir,'train','label.txt')
test_path = os.path.join(dataset_dir, 'test')
test_label_path = os.path.join(dataset_dir,'test','label.txt')



# import os
# # VisionTransformer
# MODEL = "VGGNet"

# image_size = [224,224]

# dataset = "self_create"

# modle_save_path = r"/home/aistudio/model_params/"+MODEL

# EPOCH = 10

# BATCH_SIZE = 16
# # 数据集文件夹路径
# # 类别数目
# classes = ["shuibei","bi","smartphone","Tshirt","watermaller"]
# classes2label = {name:i for i,name in enumerate(classes) }
# label2classes = {i:name for i,name in enumerate(classes)}
# CLASS_DIM = len(classes)
# dataset_dir = r'/home/aistudio/data/data202960/temp'
# train_path = os.path.join(dataset_dir, 'train')
# train_label_path = os.path.join(dataset_dir,'train','label.txt')
# test_path = os.path.join(dataset_dir, 'test')
# test_label_path = os.path.join(dataset_dir,'test','label.txt')