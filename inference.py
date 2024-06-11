import numpy as np
import paddle 
import os
from PIL import Image
import config
from datasetloader import test_transform,MyDataset
from model.vgg import VGGNet
from model.vit import VisionTransformer
from model.AlexNet import AlexNet
from model.ResNet import *

import paddle.nn.functional as F
import matplotlib.pyplot as plt
test_dataset = MyDataset(config.test_path,config.test_label_path, test_transform)
test_loader = paddle.io.DataLoader(dataset=test_dataset,batch_size=config.BATCH_SIZE)
model_state_dict = paddle.load(os.path.join(config.modle_save_path,"save_dir_final.pdparams"))
if config.MODEL == 'VGGNet':
    model_eval = VGGNet(config.CLASS_DIM)
if config.MODEL == 'VisionTransformer':
    model_eval = VisionTransformer(
            img_size=224,
            patch_size=16,
            class_dim=config.CLASS_DIM,
            embed_dim=768,
            depth=10,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            epsilon=1e-6)
if config.MODEL =="AlexNet":
    model_eval = AlexNet(config.CLASS_DIM)
if config.MODEL == "ResNet":
    model_eval = ResNet50( config.CLASS_DIM)
model_eval.set_state_dict(model_state_dict) 
model_eval.eval()
all_preds = []
all_labels = []
accs = []
# 在测试集上进行预测
# 开始评估
for _, data in enumerate(test_loader()):
    x_data = data[0]
    y_data = data[1]
    predicts = model_eval(x_data)
    preds = F.softmax(predicts, axis=1).argmax(axis=1).numpy()
    acc = paddle.metric.accuracy(predicts, y_data)
    accs.append(acc.numpy())
    all_preds.extend(preds)
    all_labels.extend(y_data.numpy())
print(f'{config.MODEL}模型在验证集上的准确率为：', np.mean(accs))

# print(all_labels,all_preds)

# 测试 20张图片
# 加载训练过程保存的最后一个模型
def inference_one_image(image_path):
    origin_image = Image.open(image_path)
    if origin_image.mode != 'RGB':
        origin_image = origin_image.convert('RGB')
    image = test_transform(origin_image)
    image = image.astype('float32')
    image = image[np.newaxis,:, : ,:]  #reshape(-1,3,224,224)    
    result = model_eval(image)
    lab = np.argmax(result.numpy())
    plt.title("predict:{}".format(config.label2classes[lab]))
    plt.imshow(origin_image)
    plt.show()
# inference_one_image(r"D:\VsCodeProjects\pythonProjects\Smart_Algorithm\dataset\test\smartphone\67.jpg")

# 绘制混淆矩阵
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, classes):

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{config.MODEL} Confusion Matrix")
    plt.show()
# 绘制混淆矩阵
import config
# classes = ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5"]  # 根据你的具体分类标签进行调整
plot_confusion_matrix(all_labels, all_preds, config.classes)