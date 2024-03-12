# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from gensim.models.word2vec import Word2Vec
from torch.utils.data import Dataset, DataLoader
import sys

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MyDataset(Dataset):
    def __init__(self, data, sequence_length=10):
        # data 是一个形状为 (num_samples, 11) 的 numpy 数组
        # 其中第一列是 y，后 10 列是 x
        self.y = data[:, 0]
        self.x = data[:, 1:]
        self.sequence_length = sequence_length


    def __getitem__(self, index):
        # 根据索引返回 x 的序列和 y 的值
        # x_sequence 对应 index 到 index + sequence_length 的数据
        # y_value 对应 index到index + sequence_length 的数据，因为我们是做当前状态估计，因此只需要用同一时间的输入数据和输出数据进行训练即可
        x_sequence = self.x[index:index + self.sequence_length]
        y_value = self.y[index:index + self.sequence_length]
        return torch.tensor(x_sequence, dtype=torch.float32), torch.tensor(y_value, dtype=torch.float32)
    def __len__(self):
        # Dataset 的长度是原始数据的行数减去序列长度,即index最大能有多大
        return len(self.x) - self.sequence_length + 1

# # 生成示例数据
# # 假设我们有100个样本，每个样本的第一列是y，后10列是x
# data = np.random.rand(1001, 11)
#
# # 创建数据集实例
# dataset = MyDataset(data)
#
# # 创建 DataLoader
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
# a=0
# # 使用 DataLoader
# for x_sequence, y_value in dataloader:
#     if a==0:
#         print("X sequence: ", x_sequence.size(), x_sequence)
#         print("Y value: ", y_value.size(), y_value)
#     a=a+1
#     # 这里可以添加模型的训练代码
#     #break # 只显示一个批次的数据
# print(a)