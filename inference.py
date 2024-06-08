import numpy as np
import paddle 
from PIL import Image
from datasetloader import test_transform,MyDataset
from config import test_path,test_label_path,CLASS_DIM,label2classes
from model.vit import VisionTransformer
from model.vgg import VGGNet
import matplotlib.pyplot as plt
test_dataset = MyDataset(test_path,test_label_path, test_transform)
test_loader = paddle.io.DataLoader(dataset=test_dataset,batch_size=4)
model_state_dict = paddle.load(r'D:\VsCodeProjects\pythonProjects\Smart_Algorithm\model_params\save_dir_200.pdparams')
# model_eval = VGGNet(CLASS_DIM)
# 实例化模型
model_eval = VisionTransformer(
        img_size=224,
        patch_size=16,
        class_dim=5,
        embed_dim=768,
        depth=10,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6)
model_eval.set_state_dict(model_state_dict) 
model_eval.eval()
accs = []
# 开始评估
for _, data in enumerate(test_loader()):
    x_data = data[0]
    y_data = data[1]
    predicts = model_eval(x_data)
    acc = paddle.metric.accuracy(predicts, y_data)
    accs.append(acc.numpy())
print('模型在验证集上的准确率为：',np.mean(accs))

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
    plt.title("predict:{}".format(label2classes[lab]))
    plt.imshow(origin_image)
    plt.show()
inference_one_image(r"D:\VsCodeProjects\pythonProjects\Smart_Algorithm\dataset\test\cup\94.jpg")

