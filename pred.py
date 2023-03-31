from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F

# 数据增广方法
transform = transforms.Compose([
    transforms.Pad(4),                                                  # +4 填充至 36 x 36
    transforms.RandomHorizontalFlip(),                                  # 随机水平翻转
    transforms.RandomCrop(32),                                          # 随机裁剪至 32 x 32
    transforms.ToTensor(),                                              # 转换至 Tensor
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))     # 归一化
])

# 50000 张训练图片
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

class Predcit:
    def pred(self, set):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load('./models/model-cifar10.pth')  # 加载模型
        model = model.to(device)
        model.eval()  # 把模型转为 test 模式

        # 读取要预测的图片
        sample_idx = torch.randint(len(train_set), size=(1,)).item()    # 从数据集中随机采样
        img, label = train_set[sample_idx]                              # 取得数据集的图和标签
        img_trans = (img.numpy().transpose((1, 2, 0)) + 1) / 2
        # img = Image.open(img).convert("RGB")
        # img.show()
        plt.imshow(img_trans)                                                 # 显示图片
        plt.axis('off')                                                 # 不显示坐标轴
        plt.show()

        # 导入图片，图片扩展后为[1，1，32，32]
        # trans = transforms.Compose(
        #     [
        #         # 将图片尺寸 resize 到 32x32
        #         transforms.Resize((32, 32)),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #     ])
        # img = trans(img)
        img = (img + 1) / 2
        img = img.to(device)
        img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]

        # 预测
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        output = model(img)
        prob = F.softmax(output, dim=1)  # prob 是 10 个分类的概率
        print("概率：", prob)
        value, predicted = torch.max(output.data, 1)
        predict = output.argmax(dim=1)
        print(predict)
        pred_class = classes[predicted.item()]
        print("预测类别：", pred_class)

predict = Predcit()
predict.pred(train_set)