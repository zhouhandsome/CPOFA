import os
import paddle
import matplotlib.pyplot as plt
from model.vgg import VGGNet
from model.vit import VisionTransformer
from model.AlexNet import AlexNet
from model.ResNet import *
from datasetloader import *
from config import *
import pickle


# 数据加载器
def get_data_loaders(dataset_choice,train_path, train_label_path, test_path, test_label_path, train_transform, test_transform, batch_size):
    if dataset_choice == "self_create":
        train_dataset = MyDataset(train_path, train_label_path, train_transform)
        test_dataset = MyDataset(test_path, test_label_path, test_transform)
        train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = paddle.io.DataLoader(test_dataset, batch_size=batch_size)
    if dataset_choice == "Cifar10":
        train_dataset = paddle.vision.datasets.Cifar10(transform=train_transform)
        test_dataset = paddle.vision.datasets.Cifar10(mode="test",transform=test_transform)
        train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = paddle.io.DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

# 初始化模型
def initialize_model(model_name, class_dim):
    if model_name == 'VGGNet':
        return VGGNet(class_dim)
    elif model_name == 'VisionTransformer':
        return VisionTransformer(
            img_size=224,
            patch_size=16,
            class_dim=5,
            embed_dim=768,
            depth=10,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            epsilon=1e-6)
    elif model_name=="AlexNet":
        return  AlexNet(class_dim)
    elif model_name == "ResNet":
        return ResNet18( num_classes=class_dim)
    elif model_name == "ResNet50":
        return ResNet50( num_classes=class_dim)
    elif model_name == "ResNet101":
        return ResNet101( num_classes=class_dim)
    elif model_name == "ResNet152":
        return ResNet152( num_classes=class_dim)
    else:
        raise ValueError("Unknown model name")
    

# 训练和评估模型
def train_and_evaluate(model, train_loader, test_loader, epochs, model_save_path, learning_rate=0.0001):
    model.train()
    cross_entropy = paddle.nn.CrossEntropyLoss()
    optimizer = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters())

    steps = 0
    Iters, train_loss, total_acc, test_acc = [], [], [], []
    test_iters = []

    for epo in range(epochs):
        for _, data in enumerate(train_loader()):
            steps += 1
            x_data, y_data = data[0], data[1]
            predicts, acc = model(x_data, y_data.reshape((-1,1)))
            loss = cross_entropy(predicts, y_data)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()


            model.train()
            if steps % 10 == 0:
                Iters.append(steps)
                train_loss.append(loss.numpy())
                total_acc.append(acc.numpy())
                model.eval()
                accs = []
                for _, data in enumerate(test_loader()):
                    x_data, y_data = data[0], data[1]
                    predicts = model(x_data)
                    acc = paddle.metric.accuracy(predicts, y_data.reshape((-1,1)))
                    accs.append(acc.numpy())
                test_iters.append(steps)
                test_acc.append(np.mean(accs))
                print(f'{MODEL}模型在验证集上的准确率为：', np.mean(accs))
                print(f'epo: {epo}, step: {steps}, loss is: {loss.numpy()}, acc is: {acc.numpy()}')

            if steps % 100 == 0:
                save_path = os.path.join(model_save_path, f"save_dir_{steps}.pdparams")
                print(f'save model to: {save_path}')
      #          paddle.save(model.state_dict(), save_path)

    
    model.eval()
    accs = []
    for _, data in enumerate(test_loader()):
        x_data, y_data = data[0], data[1]
        predicts = model(x_data)
        acc = paddle.metric.accuracy(predicts, y_data)
        accs.append(acc.numpy())
    test_iters.append(steps)
    test_acc.append(np.mean(accs))
    print('模型在验证集上的准确率为：', np.mean(accs))

    paddle.save(model.state_dict(), os.path.join(model_save_path, "save_dir_final.pdparams"))
    return Iters, train_loss, total_acc, test_iters, test_acc


def save_log(model_name,batch_size,dir_save_path ,Iters, train_loss, test_iters, test_acc):
    log_dit = {
        "model_name":model_name,
        "batchsize":batch_size,
        "train_iters":Iters,
        "train_loss":train_loss,
        "test_iters":test_iters,
        "test_acc":test_acc,
    }
    with open(os.path.join(dir_save_path, "results.pkl"), "wb") as f:
        pickle.dump(log_dit, f)




if __name__ == '__main__':
    # train_loader, test_loader = get_data_loaders(DATASET_CHOICE,train_path, train_label_path, test_path, test_label_path, train_transform, test_transform, BATCH_SIZE)
    # model =  initialize_model(MODEL, CLASS_DIM)
    # Iters, train_loss, total_acc, test_iters, test_acc = train_and_evaluate(model, train_loader, test_loader, EPOCH, modle_save_path)
    # # 保存训练过程
    # save_log(MODEL,BATCH_SIZE,modle_save_path,Iters, train_loss, test_iters, test_acc)
    # ['VGGNet','VisionTransformer','AlexNet','ResNet']
    model_names = ['VGGNet','VisionTransformer','AlexNet','ResNet152','ResNet101','ResNet50','ResNet']
    base_path = r"D:\VsCodeProjects\pythonProjects\Smart_Algorithm\model_params"

    for model_name in model_names:
        MODEL=model_name
        model_save_path = os.path.join(base_path, model_name)
        print(model_save_path)
        train_loader, test_loader = get_data_loaders(DATASET_CHOICE,train_path, train_label_path, test_path, test_label_path, train_transform, test_transform, BATCH_SIZE)
        model = initialize_model(MODEL, CLASS_DIM)
        Iters, train_loss, total_acc, test_iters, test_acc = train_and_evaluate(model, train_loader, test_loader, EPOCH, modle_save_path)
        # 保存训练过程
        save_log(MODEL,BATCH_SIZE,modle_save_path,Iters, train_loss, test_iters, test_acc)



