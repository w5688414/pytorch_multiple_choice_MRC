import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import random
import os
import time
from sklearn.model_selection import *
from transformers import *
import numpy as np
import matplotlib.pyplot as plt


train_df=pd.read_json('./mrc_data/c3/d-train.json')

train_df.rename(columns={0:'context',1:'qa',2:'id'},inplace=True)

train_df['content']=train_df['context'].apply(lambda x:''.join(x[0]))
train_df['question']=train_df['qa'].apply(lambda x:x[0]['question'])
train_df['choice']=train_df['qa'].apply(lambda x:x[0]['choice'])
train_df['answer']=train_df['qa'].apply(lambda x:x[0]['answer'])


def convert_answer_to_id(x):
#     print(x)
    answer=x['answer']
    choice=x['choice']
    return choice.index(answer)

train_df['label'] = train_df.apply(lambda x:convert_answer_to_id(x),axis=1) 

CFG = { #训练的参数配置
    'fold_num': 5, #五折交叉验证
    'seed': 42,
    'model': './check_points/prev_trained_model/chinese_roberta_wwm_large_ext', #预训练模型
    'max_len': 256, #文本截断的最大长度
    'epochs': 8,
    'train_bs': 16, #batch_size，可根据自己的显存调整
    'valid_bs': 16,
    'lr': 2e-5, #学习率
    'num_workers': 16,
    'accum_iter': 2, #梯度累积，相当于将batch_size*2
    'weight_decay': 1e-4, #权重衰减，防止过拟合
    'device': 1,
}

tokenizer = BertTokenizer.from_pretrained(CFG['model']) #加载bert的分词器

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['seed']) #固定随机种子

torch.cuda.set_device(CFG['device'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx): #将一条数据从(文章,问题,4个选项)转成(文章,问题,选项1)、(文章,问题,选项2)...
        label = self.df.label.values[idx]
        question = self.df.question.values[idx]
        content = self.df.content.values[idx]
        choice = self.df.choice.values[idx]
        if len(choice) < 4: #如果选项不满四个，就补“不知道”
            for i in range(4-len(choice)):
                choice.append('D．不知道')
        
        content = [content for i in range(len(choice))]
        pair = [question + ' ' + i[2:] for i in choice]
        
        return content, pair, label


def collate_fn(data): #将文章问题选项拼在一起后，得到分词后的数字id，输出的size是(batch, n_choices, max_len)
    input_ids, attention_mask, token_type_ids = [], [], []
    for x in data:
        text = tokenizer(x[1], text_pair=x[0], padding='max_length', truncation=True, max_length=CFG['max_len'], return_tensors='pt')
        input_ids.append(text['input_ids'].tolist())
        attention_mask.append(text['attention_mask'].tolist())
        token_type_ids.append(text['token_type_ids'].tolist())
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    label = torch.tensor([x[-1] for x in data])
    return input_ids, attention_mask, token_type_ids, label
    

