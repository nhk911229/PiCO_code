import torch
import torch.nn.functional as F
import torch.nn as nn


class partial_loss(nn.Module):
    """
    負責 label disambiguation，以計算 classification loss L_cls
    """
    def __init__(self, confidence, conf_ema_m=0.99):
        super().__init__()
        self.confidence = confidence #每個候選標籤的預測機率值
        self.init_conf = confidence.detach()
        self.conf_ema_m = conf_ema_m #滑動平均機制的參數𝜙，控制偽標籤的更新速度

    def set_conf_ema_m(self, epoch, args): #動態調整滑動平均機制的參數𝜙
        start = args.conf_ema_range[0]
        end = args.conf_ema_range[1]
        self.conf_ema_m = 1. * epoch / args.epochs * (end - start) + start #更新參數𝜙

    def forward(self, outputs, index): #計算L_cls
        logsm_outputs = F.log_softmax(outputs, dim=1) #計算每個類別的 log probability
        final_outputs = logsm_outputs * self.confidence[index, :] # log probability 與候選標籤的預測機率值相乘，得到加權的 log probability
        average_loss = - ((final_outputs).sum(dim=1)).mean() #計算損失值，希望在候選集內的機率總和為 1，且不在候選集內的標籤機率為 0
        return average_loss

    def confidence_update(self, temp_un_conf, batch_index, batchY): #更新
        """
        其中temp_un_conf是指某個樣本和各類別代表向量 (prototype) 的相似度機率值
        其中batch_index是指當前樣本在整個訓練集中的索引位置
        其中batchY是指候選標籤
        """
        with torch.no_grad():
            _, prot_pred = (temp_un_conf * batchY).max(dim=1) #候選標籤與對應的代表向量 (prototype) 計算相似度，並找最像的作為偽標籤
            pseudo_label = F.one_hot(prot_pred, batchY.shape[1]).float().cuda().detach() #對偽標籤進行one-hot編碼
            self.confidence[batch_index, :] = self.conf_ema_m * self.confidence[batch_index, :] + (1 - self.conf_ema_m) * pseudo_label #使用滑動平均機制更新每個候選標籤的預測機率值
        return None


class SupConLoss(nn.Module):
    """
    Following Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    負責 representation learning，以計算 contrastive loss L_cont
    """

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature #溫度參數τ，控制對比學習的嚴格程度，溫度愈小模型會愈嚴格區分正負樣本
        self.base_temperature = base_temperature

    def forward(self, features, mask=None, batch_size=-1):
        """
        其中features是指 embedding pool A，包含了當前批次的 query embeddings、key embeddings、與歷史累積的 key embeddings
        """
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if mask is not None: # Partial Label Mode for PiCO

            mask = mask.float().detach().to(device) #遮罩，用來區分哪些是正(相同偽標籤的)樣本、哪些是負(不同偽標籤的)樣本
            anchor_dot_contrast = torch.div(torch.matmul(features[:batch_size], features.T), self.temperature) #計算目前樣本與其餘正負樣本的相似度

            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0) #避免目前樣本去計算自己與自己的相似度
            mask = mask * logits_mask #避免正負樣本去計算自己與自己的相似度

            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12) #計算 log probability
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) #僅針對正樣本計算 mean log probability

            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean() #計算損失值，希望在候選集內的樣本相似度總和為 1，且不在候選集內的相似度為 0
        
        
        else: # unsupervised Mode for PiCO+

            q = features[:batch_size] # 當前批次的 query embeddings
            k = features[batch_size:batch_size * 2] # 當前批次的 key embeddings
            queue = features[batch_size * 2:] # 歷史累積的 key embeddings

            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) # 計算正樣本的相似度
            l_neg = torch.einsum('nc,kc->nk', [q, queue]) # 計算負樣本的相似度
            logits = torch.cat([l_pos, l_neg], dim=1) # 合併正負樣本的相似度數值
            logits /= self.temperature

            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = F.cross_entropy(logits, labels)

        return loss
