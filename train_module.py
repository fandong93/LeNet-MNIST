
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class Train(nn.Module):
    def train_method(self, model, device, loss_function, train_loader, optimizer, epoch):
        # 训练模型, 启用 BatchNormalization 和 Dropout, 将 BatchNormalization 和 Dropout 置为 True
        model.train()
        model = model.to(device)

        num = 0
        total = 0
        correct = 0.0
        sum_loss = 0.0

        train_bar = tqdm(train_loader, file=sys.stdout)
        # enumerate 迭代已加载的数据集，同时获取数据和数据下标
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)   # 把模型部署到 device 上
            optimizer.zero_grad()       # 梯度清零
            outputs = model(images)     # 保存训练结果
            # 计算损失和
            # 多分类情况通常使用 cross_entropy（交叉熵损失函数），而对于二分类问题，通常使用 sigmod
            loss = loss_function(outputs, labels)
            sum_loss += loss.item()
            # 获取最大概率的预测结果
            predicts = outputs.argmax(dim=1)     # dim=1 表示返回每一行的最大值对应的列下标
            total += labels.size(0)             # 计算标签总数
            correct += torch.eq(predicts, labels).sum().item()   # 计算预测正确的标签总数
            num = step
            loss.backward()     # 反向传播
            optimizer.step()    # 更新参数
            # loss.item() 表示当前 loss 的数值
            train_bar.desc = 'Train Epoch {:d}, Loss {:.4f}, Acc {:.3f}%'.format(epoch, loss.item(), 100 * (correct / total))

        return sum_loss / num, correct / total
