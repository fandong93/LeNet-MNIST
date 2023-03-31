
import os
import torch
import torch.nn as nn
from model import LeNet     # 导入训练模型
import torch.optim as optim
import matplotlib.pyplot as plt
from load_module import Load
import train_module
import valid_module
import time


def main():
    # device : GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))
    # 加载数据
    load = Load()
    train_loader, val_loader = load.load_data("./datasets", 64, 100)
    # 创建模型，部署 gpu
    model = LeNet().to(device)
    # 交叉熵损失
    loss_function = nn.CrossEntropyLoss().to(device)
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 调用
    epochs = 100
    best_acc = 0.0
    max_loss = 0.0
    min_loss = 1.0
    max_acc = 0.0
    min_acc = 1.0

    Loss = []
    Accuracy = []

    train = train_module.Train()
    valid = valid_module.Valid()

    save_path = './model/LeNet-CIFAR10.pth'
    if not os.path.exists("./model"):
        os.mkdir("./model")

    img_path = './img/LeNet-CIFAR10.jpg'
    if not os.path.exists("./img"):
        os.mkdir("./img")

    print("start_time", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    for epoch in range(0, epochs + 1):
        loss, train_acc = train.train_method(model, device, loss_function, train_loader, optimizer, epoch)

        if loss > max_loss:
            max_loss = loss
        if loss < min_loss:
            min_loss = loss

        valid_acc = valid.valid_method(model, device, val_loader, epoch)

        if valid_acc > max_acc:
            max_acc = valid_acc

        if valid_acc < min_acc:
            min_acc = valid_acc

        Loss.append(loss)
        Accuracy.append(valid_acc)

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), save_path)

    print("end_time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    print('Finished Training')

    plt.subplot(2, 1, 1)
    plt.plot(Loss)
    plt.title('Loss')
    x_ticks = torch.arange(0, epochs + 1, 10)
    plt.xticks(x_ticks)
    y_ticks = torch.arange(min_loss, max_loss, 0.1)
    plt.yticks(y_ticks)

    plt.subplot(2, 1, 2)
    plt.plot(Accuracy)
    plt.title('Accuracy')
    x_ticks = torch.arange(0, epochs + 1, 10)
    plt.xticks(x_ticks)
    y_ticks = torch.arange(min_acc, max_acc, 0.01)
    plt.yticks(y_ticks)

    plt.subplots_adjust(hspace=0.3)  # 调整子图间距
    plt.savefig(img_path)
    plt.show()


if __name__ == "__main__":
    main()
