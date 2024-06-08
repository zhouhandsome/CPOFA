import os

MODEL = "VGG"

image_size = [224,224]

dataset = "self_create"

modle_save_path = r"D:\VsCodeProjects\pythonProjects\Smart_Algorithm\model_params"

EPOCH = 15

BATCH_SIZE = 4
# 数据集文件夹路径
# 类别数目
classes = ["shuibei","bi","smartphone","Tshirt","watermaller"]
classes2label = {name:i for i,name in enumerate(classes) }
label2classes = {i:name for i,name in enumerate(classes)}
CLASS_DIM = len(classes)
dataset_dir = r'D:\VsCodeProjects\pythonProjects\Smart_Algorithm\dataset'
train_path = os.path.join(dataset_dir, 'train')
train_label_path = os.path.join(dataset_dir,'train','label.txt')
test_path = os.path.join(dataset_dir, 'test')
test_label_path = os.path.join(dataset_dir,'test','label.txt')