
""" build CIFAR-10 Partial Label Dataset """


import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from .randaugment import RandomAugment
from .utils_algo import generate_uniform_cv_candidate_labels


def load_cifar10(partial_rate, batch_size): 
    """
    載入CIFAR-10資料集，並將資料集轉換為 PiCO 所需的 Partial Label Learning 格式
    第一個參數partial_rate，是論文中提及的q值，用來控制錯誤標籤被加進候選標籤的機率
    第二個參數batch_size，是批次量，用來控制每次處理的資料筆數
    """


    #---------------------------------------------------------------------------
    # 下載測試資料集，並對測試資料集進行預處理
    #---------------------------------------------------------------------------
    test_transform  = transforms.Compose([ #定義測試資料集的轉換格式
        transforms.ToTensor(), #把 numpy image 轉換成 PyTorch tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)) #正規化，先減去平均值在儲上標準差
    ])

    test_dataset = dsets.CIFAR10( #下載測試資料集
        root='./data', 
        train=False, #測試資料集中有 10000 張圖片，每張圖片標有0-9的其中一個物件
        download=True, #下載
        transform=test_transform
    )

    test_loader = torch.utils.data.DataLoader( #建立測試資料的加載器
        dataset=test_dataset, 
        batch_size=batch_size*4, #批次量
        shuffle=False, #不打亂資料
        num_workers=4, #使用4個子進程來加載資料(加快資料加載速度)
        sampler=torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False) #並行處理
    )


    #---------------------------------------------------------------------------
    # 下載測試資料集，並對測試資料集進行預處理
    #---------------------------------------------------------------------------
    temp_train = dsets.CIFAR10( #下載訓練資料集
        root='./data', 
        train=True, #訓練資料集中有 50000 張圖片，每張圖片標有0-9的其中一個數字
        download=True, 
        transform=transforms.ToTensor() #前面先不做正規化，後續資料擴增時會補做正規化
    )

    data = temp_train.data #訓練資料集的特徵形狀 (50000, 32, 32, 3)
    labels = torch.Tensor(temp_train.targets).long() #訓練資料集的真實標籤形狀 (50000, )
    
    partialY = generate_uniform_cv_candidate_labels( #訓練資料集的候選標籤形狀 (50000, 10)
        labels,
        partial_rate #加上錯誤標籤，可能從 [1,0,0,0,0,0,0,0,0,0] 變成 [1,0,0,1,0,1,0,0,1,0]
    )

    temp = torch.zeros(partialY.shape); temp[torch.arange(partialY.shape[0]), labels] = 1 #one-hot編碼，標籤形狀從 (50000, )變成 (50000, 10)
    if torch.sum(partialY * temp) == partialY.shape[0]: print('partialY correctly loaded') #候選標籤有包含真實標籤
    else: print('inconsistent permutation') #候選標籤沒有包含真實標籤
    print('Average candidate num: ', partialY.sum(1).mean()) #輸出候選標籤的平均數量，確認候選標籤都有包含真實標籤
    
    partial_matrix_dataset = CIFAR10_Augmentention(data, partialY.float(), labels.float()) #訓練資料集的擴增圖像

    train_sampler = torch.utils.data.distributed.DistributedSampler(partial_matrix_dataset) #並行處理
    partial_matrix_train_loader = torch.utils.data.DataLoader( #建立訓練資料的加載器
        dataset=partial_matrix_dataset, 
        batch_size=batch_size, #批次量
        shuffle=(train_sampler is None), #打亂資料
        num_workers=4, #使用4個子進程來加載資料(加快資料加載速度)
        pin_memory=True, 
        sampler=train_sampler, #並行處理
        drop_last=True
    )

    return partial_matrix_train_loader,partialY,train_sampler,test_loader


class CIFAR10_Augmentention(Dataset): #資料擴增
    def __init__(self, images, given_label_matrix, true_labels):
        self.images = images                          #原始圖像
        self.given_label_matrix = given_label_matrix  #候選標籤
        self.true_labels = true_labels                #真實標籤
        self.weak_transform = transforms.Compose([    #利用弱資料擴增建立 query image，以便後續 contrastive learning 
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),                         #裁切
            transforms.RandomHorizontalFlip(),                                              #左右翻轉
            transforms.RandomApply([ transforms.ColorJitter(0.4, 0.4, 0.4, 0.1) ], p=0.8),  #變色
            transforms.RandomGrayscale(p=0.2),                                              #灰階化
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        self.strong_transform = transforms.Compose([   #利用強資料擴增 key image，以便後續 contrastive learning 
            transforms.ToPILImage(), #把 numpy image 轉換成 PIL image
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, 5), #隨機做出3張資料增強圖像
            transforms.ToTensor(), #把 PIL image 轉換成 PyTorch tensor
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        each_image_w = self.weak_transform(self.images[index])    #產生弱資料擴增的圖片
        each_image_s = self.strong_transform(self.images[index])  #產生強資料擴增的圖片
        each_label = self.given_label_matrix[index]               #取得候選標籤
        each_true_label = self.true_labels[index]                 #取得真實標籤
        return each_image_w, each_image_s, each_label, each_true_label, index

