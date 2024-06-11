import paddle
from paddle import nn
from paddle.io import DataLoader
import paddle.vision.transforms as T
from paddle.vision.datasets import Cifar10

cifar10 = Cifar10()
transform = T.Compose(
    [
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            to_rgb=True,
           ),
    ]
)
train_dataset = DataLoader(cifar10,batch_size=16)

if __name__ == "__main__":


