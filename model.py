
""" build PiCO model """


import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F

class PiCO(nn.Module):
    """
    對比學習 (Contrastive Representation Learning) 希望相同類別的特徵相近、不同類別的特徵遠離
    標籤消歧異 Prototype-based Label Disambiguation 希望從候選標籤中選出最有可能的真實標籤作為偽標籤
    """

    def __init__(self, args, base_encoder):
        super().__init__()
        
        pretrained = args.dataset == 'cub200'

        self.encoder_q = base_encoder( # encoder
            num_class=args.num_class, #類別數量 10
            feat_dim=args.low_dim, # 將query view 轉換成 128 維的 query embedding
            name=args.arch, #使用 resnet18 模型架構進行特徵提取
            pretrained=pretrained
        )
        self.encoder_k = base_encoder( # momentum encoder
            num_class=args.num_class, #類別數量 10
            feat_dim=args.low_dim, # 將key view 轉換成 128 維的 key embedding
            name=args.arch, #使用 resnet18 模型架構進行特徵提取
            pretrained=pretrained
        )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  #初始化 momentum encoder 的參數
            param_k.requires_grad = False  # momentum encoder 的參數不計算梯度，而是透過動量慢慢趨近 encoder 的參數

        # yjuny:创建各条队列 /负样本/伪标签/ptr(队列指针)/原型向量
        self.register_buffer("queue", torch.randn(args.moco_queue, args.low_dim)) # contrastive learning embedding pool,用來記錄 key embedding 的特徵
        self.register_buffer("queue_pseudo", torch.randn(args.moco_queue)) # contrastive learning embedding pool,用來記錄 key embedding 的偽標籤，以便後續分出正樣本與負樣本
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  #紀錄 pool 中每個 key embedding 的下一個人的指標
        self.register_buffer("prototypes", torch.zeros(args.num_class,args.low_dim)) #紀錄每個類別的代表向量
        self.queue = F.normalize(self.queue, dim=0) #L2正規化，以統一所有向量


    @torch.no_grad()
    def _momentum_update_key_encoder(self, args):
        """
        更新 momentum encoder 的參數，不是透過梯度更新參數，而是利用公式更新參數
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m) #慢慢靠近 encoder 的參數



    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, args):
        """
        以先進先出的順序更新 contrastive learning embedding pool 中紀錄的 key embedding
        """
        keys = concat_all_gather(keys) #新的 key embedding
        labels = concat_all_gather(labels) #新的偽標籤
        batch_size = keys.shape[0] #批次量，也就是有多少個 key embedding 和多少個偽標籤

        ptr = int(self.queue_ptr) #取得 dequeue 的結尾指標位址，表示下一批次的向量從這個位置開始存放
        assert args.moco_queue % batch_size == 0  # for simplicity

        self.queue[ptr:ptr + batch_size, :] = keys #更新 pool 中的 key embedding
        self.queue_pseudo[ptr:ptr + batch_size] = labels #更新 pool 中的 偽標籤

        ptr = (ptr + batch_size) % args.moco_queue  #指標循環，若已經到最後一個位址了就從頭接續
        self.queue_ptr[0] = ptr #更新 dequeue 的結尾指標位址



    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index 随机打乱样本索引
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus 广播至所有的GPU
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring 恢复索引
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu 当前GPU的乱序索引
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    # yjuny:恢复样本索引
    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)

        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    # yjuny:模型前向传播
    def forward(self, img_q, im_k=None, partial_Y=None, args=None, eval_only=False):
        """
        論文模型 PICO 的前向傳播
        第一個參數 img_q，是 query view、 weak augmentation image，作為 encoder 的輸入
        第二個參數 im_k，是 key view、 strong augmentation image，作為 momentum encoder 的輸入
        第三個參數 partial_Y，是候選標籤矩
        第五個參數 eval_only，用來表示模型處於訓練模式還是測試模式
        """

        #--------------------------------------------------------------
        #  Step1: 利用 encoder 產生 classifier output 與 query embedding
        #---------------------------------------------------------------
        output, q = self.encoder_q(img_q) #如果是訓練模式，還需要取得 query embedding q
        if eval_only: return output #如果是測試模式，只需要即可分類器對每個類別的預測機率值

        #--------------------------------------------------------------
        #  Step2: 從 classifier output 中選出偽標籤
        #---------------------------------------------------------------
        predicted_scores = torch.softmax(output, dim=1) * partial_Y #過濾掉候選標籤以外的預測機率
        max_scores, pseudo_labels_b = torch.max(predicted_scores, dim=1) #從候選標籤中選出偽標籤
        print('pseudo_labels_b:', pseudo_labels_b)

        #--------------------------------------------------------------
        #  Step3: 計算 query embedding 和 prototypy 的相似度
        #---------------------------------------------------------------
        prototypes = self.prototypes.clone().detach() #取得每個類別的 prototypy
        logits_prot = torch.mm(q, prototypes.t()) #計算 query embedding 和 每個 prototypy 的相似度
        score_prot = torch.softmax(logits_prot, dim=1) #將相似度轉換成機率

        #---------------------------------------------------------------
        #  Step4: 更新 prototypes
        #---------------------------------------------------------------
        for feat, label in zip(concat_all_gather(q), concat_all_gather(pseudo_labels_b)): #更新 prototypy
            self.prototypes[label] = self.prototypes[label]*args.proto_m + (1-args.proto_m)*feat
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1) #L2正規化，以統一所有向量
        
        #---------------------------------------------------------------
        #  Step5: 利用 momentum encoder 產生 key embedding k
        #---------------------------------------------------------------
        with torch.no_grad(): # momentum encoder 的參數不是透過梯度更新的
            self._momentum_update_key_encoder(args)  #而是利用公式慢慢趨近 encoder 的參數
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            _, k = self.encoder_k(im_k) #取得 key embedding k
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        #---------------------------------------------------------------
        #  Step6: 更新 queue、contrastive embedding pool A
        #---------------------------------------------------------------
        features = torch.cat((q, k, self.queue.clone().detach()), dim=0) # contrastive embedding pool A，包含當前批次的 q、k、以及歷史累積的 k
        pseudo_labels = torch.cat((pseudo_labels_b, pseudo_labels_b, self.queue_pseudo.clone().detach()), dim=0) #偽標籤
        self._dequeue_and_enqueue(k, pseudo_labels_b, args) #更新 contrastive embedding pool A

        #---------------------------------------------------------------
        #  Step7: 回傳所有 loss 需要的資訊
        #---------------------------------------------------------------
        return output, features, pseudo_labels, score_prot, pseudo_labels_b


class PiCO_PLUS(PiCO):
    '''yjuny:PiCO+是PiCO的强大扩展，能够缓解嘈杂的部分标签学习问题'''
    def __init__(self, args, base_encoder):
        super().__init__(args, base_encoder)
        # yjuny:相关性队列
        self.register_buffer("queue_rel", torch.zeros(args.moco_queue, dtype=torch.bool))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, is_rel, args):
        super()._dequeue_and_enqueue(keys, labels, args)
        # change:yjuny:聚类的时候选择比较靠谱的
        is_rel = concat_all_gather(is_rel)
        batch_size = is_rel.shape[0]
        ptr = int(self.queue_ptr)
        self.queue_rel[ptr:ptr + batch_size] = is_rel
        # update queue_rel

    def forward(self, img_q, im_k=None, Y_ori=None, Y_cor=None, is_rel=None, args=None, eval_only=False, ):

        output, q = self.encoder_q(img_q)
        if eval_only:
            return output
        # for testing

        batch_weight = is_rel.float()
        with torch.no_grad():  # no gradient 

            predicetd_scores = torch.softmax(output, dim=1)
            # yjuny:一个batch里面的所有类别
            _, within_max_cls = torch.max(predicetd_scores * Y_ori, dim=1)
            _, all_max_cls = torch.max(predicetd_scores, dim=1)
            pseudo_labels_b = batch_weight * within_max_cls + (1 - batch_weight) * all_max_cls
            pseudo_labels_b = within_max_cls
            pseudo_labels_b = pseudo_labels_b.long()
            # for clean data, using partial labels to filter out negative labels
            # for noisy data, we enable a full set pseudo-label selection

            # compute protoypical logits
            prototypes = self.prototypes.clone().detach()
            logits_prot = torch.mm(q, prototypes.t())
            score_prot = torch.softmax(logits_prot, dim=1)
            # prototypes follows the same

            # change:yjuny:在这里，我们使用到分类器预测的原始集合原型内的距离来检测候选标签集是否有噪声。
            # change:yjuny:如果实例远离分类器预测的原型，它可能违反对比学习的聚类趋势，因此我们将其视为有噪声
            # change:yjuny:is_rel正是我们用来判断哪些是噪音样本的标记向量(可靠程度)
            _, within_max_cls_ori = torch.max(predicetd_scores * Y_ori, dim=1)
            distance_prot = - (q * prototypes[within_max_cls_ori]).sum(dim=1)
            # Here we use the distances to those within the original set prototype of classifier prediction
            #       to detect whether a candidate label set is noisy
            # if the instance is far away from the classifier-predicted prototype,
            #       it may violate the clustering tendency of the contrastive learning
            #       and hence we regard it as noisy

            # update momentum prototypes with pseudo labels
            # yjuny:动量更新原型标签
            for feat, label in zip(concat_all_gather(q[is_rel]), concat_all_gather(pseudo_labels_b[is_rel])):
            # for feat, label in zip(concat_all_gather(q), concat_all_gather(pseudo_labels_b)):
                self.prototypes[label] = self.prototypes[label]*args.proto_m + (1-args.proto_m)*feat
            # normalize prototypes
            self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
            # print:
            # print('prototypes:', self.prototypes, self.prototypes.shape)
            # print('queue_pseudo:', self.queue_pseudo, self.queue_pseudo.shape)
            
            # compute key features 
            self._momentum_update_key_encoder(args)  # update the momentum encoder
            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            _, k = self.encoder_k(im_k)
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        # print:
        # print('self.queue:', self.queue, self.queue.shape)
        # print('pseudo_labels_before:', pseudo_labels_b, pseudo_labels_b.shape)
        pseudo_labels = torch.cat((pseudo_labels_b, pseudo_labels_b, self.queue_pseudo.clone().detach()), dim=0)
        # print:
        # print('pseudo_labels:', pseudo_labels, pseudo_labels.shape)
        is_rel_queue = torch.cat((is_rel, is_rel, self.queue_rel.clone().detach()), dim=0)
        # to calculate SupCon Loss using pseudo_labels and partial target
        
        # dequeue and enqueue
        self._dequeue_and_enqueue(k, pseudo_labels_b, is_rel, args)

        return output, features, pseudo_labels, score_prot, distance_prot, is_rel_queue, within_max_cls

# utils
# 聚合函数 PiCO使用多GPU进行分布式训练，这使得一个batch的样本被拆分至多个GPU上，因此在对字典编码器更新时需要汇聚所有GPU上的样本。
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
