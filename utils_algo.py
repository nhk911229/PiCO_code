
""" 1. calculate training loss and accuracy """
""" 2. print training progress """
""" 3. adjust learning rate """
""" 4. calculate accuracy """
""" 5. build Partial Label """


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle



class AverageMeter(object): 
    """ 紀錄並計算每個epoch的平均損失值與準確值 """

    def __init__(self, name, fmt=':f'):
        self.name = name #數據名稱(損失值or準確值)
        self.fmt = fmt #輸出格式
        self.reset()

    def reset(self): 
        self.val = 0    #初始化當前值
        self.avg = 0    #初始化平均值 
        self.sum = 0    #初始化總和
        self.count = 0  #初始化樣本數

    def update(self, val, n=1):
        self.val = val                    #更新當前值
        self.sum += val * n               #更新總和
        self.count += n                   #更新樣本數
        self.avg = self.sum / self.count  #更新平均值

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__) #訓練過程的資訊



class ProgressMeter(object):
    """ 輸出訓練進度 """

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches) #第幾個batch
        self.meters = meters #指標(損失值or準確值)
        self.prefix = prefix #第幾個epoch

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries)) #輸出目前的訓練進度

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']' #出整齊對齊的訓練進度



def adjust_learning_rate(args, optimizer, epoch):
    """ 動態調整學習率 """

    lr = args.lr
    if args.cosine: #每個epoch都慢慢降低學習率
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else: #只有特定epoch才降低學習率
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0: lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups: param_group['lr'] = lr #更新學習率



def accuracy(output, target, topk=(1,)):
    """ 計算 Top-k 準確值 """

    with torch.no_grad():
        batch_size = target.size(0) #取得批次量
        _, pred = output.topk(max(topk), 1, True, True) #取預測機率前K高的索引編號
        pred = pred.t() #轉置，將預測結果從 (batch_size, k) 轉為 (k, batch_size)
        correct = pred.eq(target.view(1, -1).expand_as(pred)) #判斷預測標籤是否和真實標籤一致
        res = []
        for k in topk: #計算 Top-k 準確值
            correct_k = correct[:k].reshape((-1,)).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size)) 
        return res



def accuracy_check(loader, model, device):
    """ 計算測試資料集的準確值 """

    with torch.no_grad():
        total, num_samples = 0, 0
        for images, labels in loader:
            labels, images = labels.to(device), images.to(device)
            outputs, _ = model(images) #模型預測，會輸出預測機率與特徵向量
            _, predicted = torch.max(outputs.data, 1) #取的預測機率最高的索引編號
            total += (predicted == labels).sum().item() #計算預測正確的樣本數量
            num_samples += labels.size(0) #計算總樣本數量
    return total / num_samples #平均準確值


def sigmoid_rampup(current, rampup_length, exp_coe=5.0):
    """ 產生指數型增長曲線 (Exponential rampup from https://arxiv.org/abs/1610.02242) """
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-exp_coe * phase * phase))


def linear_rampup(current, rampup_length):
    """ 產生線性增長曲線 (Linear rampup) """
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length: return 1.0
    else: return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """ Cosine rampdown from https://arxiv.org/abs/1608.03983 """
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1):
    """
    建立候選標籤 
    第一個參數train_labels，是訓練資料集的真實標籤，形狀為 (50000, )，每個元素是0-9的其中一個數字
    第二個參數partial_rate，是論文中提及的q值，用來控制錯誤標籤被加進候選標籤的機率
    """

    if torch.min(train_labels) > 1: raise RuntimeError('testError') #確認標籤是從 0 開始
    elif torch.min(train_labels) == 1: train_labels = train_labels - 1 #確認標籤是從 0 開始

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1) #類別數量10
    n = train_labels.shape[0] #樣本數量50000

    partialY = torch.zeros(n, K) #建立候選標籤
    partialY[torch.arange(n), train_labels] = 1.0 #將真實標籤加入候選標籤集合
    transition_matrix = np.eye(K) #建立單位矩陣

    transition_matrix[np.where(~np.eye(transition_matrix.shape[0], dtype=bool))] = partial_rate #機率矩陣，用來表示正確標籤被加入候選標籤的機率為 100%
    random_n = np.random.uniform(0, 1, size=(n, K)) #建立隨機矩陣，用來表示加上錯誤標籤被加入候選標籤的機率為 ??%
    for j in range(n):  #在候選標籤中加上錯誤標籤
        partialY[j, :] = torch.from_numpy((random_n[j, :] < transition_matrix[train_labels[j], :]) * 1)

    print("Finish Generating Candidate Label Sets!\n")
    return partialY

    # tensor([[1., 1., 0.,  ..., 0., 0., 0.],
    #         [0., 1., 0.,  ..., 1., 0., 0.],
    #         [1., 1., 0.,  ..., 0., 0., 1.],
    #         ...,
    #         [1., 1., 1.,  ..., 1., 1., 1.],
    #         [1., 1., 1.,  ..., 0., 0., 0.],
    #         [0., 0., 0.,  ..., 0., 0., 0.]])


def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res