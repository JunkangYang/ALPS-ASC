import torch
import numpy as np
import torch.nn.functional as F


def labels_to_onehot(labels, num_classes=10):
    """
    将整型标签转换为 one-hot 编码（PyTorch 实现）

    Args:
        labels (torch.Tensor): 输入标签，shape=(batch_size,), dtype=torch.int64
        num_classes (int): 类别数（默认为 10）

    Returns:
        torch.Tensor: one-hot 编码，shape=(batch_size, num_classes)
    """
    return F.one_hot(labels.long(), num_classes=num_classes)  # .float() 可选，取决于是否需要浮点数


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''

    y = F.one_hot(y.long(), num_classes=10).float()

    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
