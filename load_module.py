
import os
import sys
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class Load:
    def load_data(self, path, train_size, valid_size):

        num_workers = 4
        if sys.platform.startswith('win'):
            num_workers = 0

        # 数据增广方法
        data_transforms = {"train": transforms.Compose([transforms.ToTensor(),    # 将图片转化为 Tensor 格式
                                                        transforms.Normalize(0.5, 0.5)]),
                           "valid": transforms.Compose([transforms.ToTensor(),    # 将图片转化为 Tensor 格式
                                                        transforms.Normalize(0.5, 0.5)])}

        flag = True
        if os.path.exists(path):
            flag = False

        # 50000 张训练图片
        train_set = torchvision.datasets.MNIST(root=path, train=True, download=flag, transform=data_transforms["train"])
        train_loader = data.DataLoader(train_set, batch_size=train_size, shuffle=True, num_workers=num_workers)

        # 10000 张验证图片
        val_set = torchvision.datasets.MNIST(root=path, train=False, download=flag, transform=data_transforms["valid"])
        val_loader = data.DataLoader(val_set, batch_size=valid_size, shuffle=True, num_workers=num_workers)

        # MNIST数据集中的十种标签
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        load = Load()
        load.show_sets(train_set, classes)

        return train_loader, val_loader

    def show_sets(self, train_sets, classes):
        # 查看数据
        figure = plt.figure(figsize=(7, 5))
        cols, rows = 6, 3
        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(len(train_sets), size=(1,)).item()   # 从数据集中随机采样
            img, label = train_sets[sample_idx]                             # 取得数据集的图和标签
            figure.add_subplot(rows, cols, i)                               # 画子图，也可以 plt.subplot(rows, cols, i)
            img = img.numpy().transpose((1, 2, 0))
            plt.title(classes[label])
            plt.axis("off")
            plt.imshow((img + 1) / 2)                                       # (img + 1) / 2 是为了还原被归一化的数据
        plt.show()


# # 测试
# load = Load()
# train_loader, val_loader = load.load_data('./datasets', 64, 100)
