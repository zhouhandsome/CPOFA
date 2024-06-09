import itertools
import paddle.vision.transforms as T
from paddle.vision.datasets import Cifar100
from paddle.io import DataLoader
class_dim = 100
cifar100 = Cifar100()
print(len(cifar100))

for i in range(5):  # only show first 5 images
    img, label = cifar100[i]
    # do something with img and label
    print(type(img), img.size, label)
    # <class 'PIL.Image.Image'> (32, 32) 19


transform = T.Compose(
    [
        T.Resize(64),
        T.ToTensor(),
        T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            to_rgb=True,
        ),
    ]
)
cifar100_train = Cifar100(
    mode="train",
    transform=transform,  # apply transform to every image
    backend="cv2",  # use OpenCV as image transform backend
)
cifar100_test = Cifar100(
    mode="test",
    transform=transform,  # apply transform to every image
    backend="cv2",  # use OpenCV as image transform backend
)
train_loader = DataLoader(cifar100_train,batch_size=4,shuffle=True)

print(len(cifar100_test))

for img, label in itertools.islice(iter(cifar100_test), 5):  # only show first 5 images
    # do something with img and label
    print(type(img), img.shape, label)
    # <class 'paddle.Tensor'> [3, 64, 64] 49
# 定义卷积池化网络