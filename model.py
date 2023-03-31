
import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5),     # input [1, 28, 28] output [6, 24, 24]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     # input [6, 24, 24] output [6, 12, 12]
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),    # input [6, 12, 12] output [16, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(2, 2)      # input [16, 8, 8] output [16, 4, 4]
        )
        self.features = nn.Sequential(
            self.conv1,
            self.conv2,
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, images):
        features = self.features(images)
        outputs = self.fc(features.view(features.shape[0], -1))
        return outputs


# # model 测试
# inputs = torch.rand([64, 1, 32, 32])    # 定义 shape
# model = LeNet()                         # 实例化
# print(model)
# outputs = model(inputs)                 # 输入网络中
# print(outputs.shape)
