import os
import pprint
import shutil
import random
import pandas as pd

Test_frac = 0.2  # 测试集比例
dataset_path = r'D:\VsCodeProjects\pythonProjects\Smart_Algorithm\picture_classify_datast'
save_dataset_path = r"D:\VsCodeProjects\pythonProjects\Smart_Algorithm\dataset"
classes = os.listdir(dataset_path)
print(classes)
class2English ={
    "手机":"smartphone",
    "铅笔":"pencle",
    "西瓜":"watermaller",
    "水杯":"cup",
    "衬衫":"Tshirt",
    "地图":'map'
}
names = ['train', 'test']
random.seed(123)  # 随机数种子
class_to_idx = {}

# 创建训练文件夹
def create_test_train_datasets():
    """
        文件名存在就不创建，不存在就创建
        :return:
        """
    for name in names:
        if name in os.listdir(save_dataset_path):
            print('训练集已存在，将训练集数据保存在 {}/{}中'.format(save_dataset_path, name))
        else:
            os.mkdir(os.path.join(save_dataset_path, name))
            print('创建 {}/{}文件夹'.format(save_dataset_path, name))
            for picture_class in classes:
                os.mkdir(os.path.join(save_dataset_path, name, class2English[picture_class]))


# 图片重命名
def pictrue_rename():
    for pictrue_class in classes:
        i = 1
        old_dir = os.path.join(dataset_path, pictrue_class)
        img_filname = os.listdir(old_dir)
        for image in img_filname:
            try:
                old_filename = os.path.join(dataset_path, pictrue_class, image)  # 获取原文件路径
                new_filename = os.path.join(dataset_path, class2English[pictrue_class], i + '.jpg')
                os.rename(old_filename, new_filename)
            except:
                pass
            i += 1


# 划分数据集
def Divide_the_dataset(test_frac):
    """
        划分数据集
        :param test_frac: 划分到测试集的图像个数
        :return:
        """
    df = pd.DataFrame()  # 用于保存分割信息
    for pictrue_class in classes:
        old_dir = os.path.join(dataset_path, pictrue_class)
        img_filname = os.listdir(old_dir)
        random.shuffle(img_filname)

        testset_number = int(len(img_filname) * test_frac)  # 测试集个数
        testset_images = img_filname[:testset_number]  # 测试集的文件名称
        trainset_images = img_filname[testset_number:]  # 训练集的文件名称

        for image in testset_images:
            old_img_path = os.path.join(dataset_path, pictrue_class, image)  # 获取原文件路径
            new_test_path = os.path.join(save_dataset_path, 'test',  class2English[pictrue_class], image)  # 获取test目录的新文件路径
            shutil.copy(old_img_path, new_test_path, )  # 移动文件

        # 图像移动到train目录
        for image in trainset_images:
            old_img_path = os.path.join(dataset_path, pictrue_class, image)  # 获取原文件路径
            new_train_path = os.path.join(save_dataset_path, 'train',  class2English[pictrue_class], image)  # 获取train目录的新文件路径
            shutil.copy(old_img_path, new_train_path)  # 移动文件

        # 删除旧的文件夹
        # assert len(os.listdir(old_dir))==0
        # shutil.rmtree(old_dir)
        # 标准打印
        print('{:^18} {:^18} {:^18}'.format(pictrue_class, len(trainset_images), len(testset_images)))
        # 保存到表格中
        df = df._append({'class': pictrue_class, 'trainset': len(trainset_images), 'testset': len(testset_images)},
                       ignore_index=True)
        df.to_csv('dataset_spilt_info.csv', index=False)


# 得到标签文件
def get_label_document(test_dir):
    # 得到测试集的标签文件
    test_lable_list = []
    labels = os.listdir(test_dir)
    for i,label in enumerate(labels):
        j = i
        class_to_idx[label] = j
        picture_names = os.listdir(os.path.join(test_dir, label))
        for picture_name in picture_names:
            picture_path = os.path.join(label, picture_name)
            # print(picture_path+'    ' +label)
            test_lable_list.append(picture_path + '\t' + str(j))
    with open(test_dir + '\label.txt', 'w', encoding='utf-8') as f:
        for test_label in test_lable_list:
            f.write(test_label + '\n')


if __name__ == '__main__':
    pictrue_rename()   # 图片重命名
    create_test_train_datasets() # 创建训练集和测试集文件夹
    Divide_the_dataset(Test_frac) # 划分数据集
    # 得到训练集和测试集的label标签文件，和标签映射文件
    try:
        test_dir = save_dataset_path+ '/test'
        get_label_document(test_dir)
        train_dir = save_dataset_path+'./train'
        get_label_document(train_dir)
    except:
        pass
    print(class_to_idx)