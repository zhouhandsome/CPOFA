import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class AlexNet(nn.Layer):
    def __init__(self, num_classes=5):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2),
            nn.Conv2D(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2),
            nn.Conv2D(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2D(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2D(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2D((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x,label=None):
        x = self.features(x)
        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        x = self.classifier(x)
        if label is not None:
            acc = paddle.metric.accuracy(input=x, label=label)
            return x, acc
        else:
            return x
if __name__ == "__main__": 
    tmp_model = AlexNet()
    paddle.summary(tmp_model, (1, 3, 224,224))