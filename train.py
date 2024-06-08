import paddle
import matplotlib.pyplot as plt
from model.vgg import VGGNet
from model.vit import VisionTransformer
from datasetloader import *
from config import *
# 折线图，用于观察训练过程中loss和acc的走势
def draw_process(title,color,iters,data,label):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel(label, fontsize=20)
    plt.plot(iters, data,color=color,label=label)
    plt.legend()
    plt.grid()
    plt.show()
train_dataset = MyDataset(train_path,train_label_path, train_transform)

train_loader = paddle.io.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)

model = VGGNet(CLASS_DIM)
# 实例化模型
# model = VisionTransformer(
#         img_size=224,
#         patch_size=16,
#         class_dim=5,
#         embed_dim=768,
#         depth=10,
#         num_heads=12,
#         mlp_ratio=4,
#         qkv_bias=True,
#         epsilon=1e-6)
model.train()
# print(paddle.summary(model,(1,3,224,224)))
# 配置loss函数
cross_entropy = paddle.nn.CrossEntropyLoss()
optimizer = paddle.optimizer.Adam(learning_rate=0.0001,
                                  parameters=model.parameters())
steps = 0
Iters, total_loss, total_acc = [], [], []
for epo in range(EPOCH):
    for _, data in enumerate(train_loader()):
        steps += 1
        x_data = data[0]
        y_data = data[1]
        predicts, acc = model(x_data, y_data)
        loss = cross_entropy(predicts, y_data)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        if steps % 10 == 0:
            Iters.append(steps)
            total_loss.append(loss.numpy())
            total_acc.append(acc.numpy())
            #打印中间过程
            print('epo: {}, step: {}, loss is: {}, acc is: {}'\
                  .format(epo, steps, loss.numpy(), acc.numpy()))
        #保存模型参数
        if steps % 100 == 0:
            save_path =os.path.join(modle_save_path,"save_dir_" + str(steps) + '.pdparams')
            print('save model to: ' + save_path)
            paddle.save(model.state_dict(),save_path)
paddle.save(model.state_dict(),os.path.join(modle_save_path,"save_dir_final.pdparams"))
draw_process("trainning loss","red",Iters,total_loss,"trainning loss")
draw_process("trainning acc","green",Iters,total_acc,"trainning acc")