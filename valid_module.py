
import sys
import torch
import torch.nn as nn
from tqdm import tqdm


class Valid(nn.Module):
    def valid_method(self, model, device, valid_loader, epoch):
        # 模型验证, 必须要写, 否则只要有输入数据, 即使不训练, 它也会改变权值
        # 因为调用 eval() 将不启用 BatchNormalization 和 Dropout, BatchNormalization和Dropout置为False
        model.eval()
        model = model.to(device)
        # 统计模型正确率, 设置初始值
        correct = 0.0
        total = 0
        # torch.no_grad 将不会计算梯度, 也不会进行反向传播
        with torch.no_grad():
            val_bar = tqdm(valid_loader, file=sys.stdout)
            for step, data in enumerate(val_bar):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predicts = outputs.argmax(dim=1)
                # 计算正确数量
                total += labels.size(0)
                correct += torch.eq(predicts, labels).sum().item()
                val_bar.desc = 'Valid Epoch {}, Acc {:.3f}%'.format(epoch, 100 * (correct / total))

            return correct / total
