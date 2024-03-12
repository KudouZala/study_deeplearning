# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from gensim.models.word2vec import Word2Vec
from torch.utils.data import Dataset, DataLoader
import sys

class MyDataset(Dataset):# 定义一个名为MyDataset的类，继承自Dataset基类，用于数据处理。
    # 加载所有的数据、存储和初始化一些变量
    """
    dataset：这是一个数据集对象，需要是torch.utils.data.Dataset类的一个实例，它定义了数据的加载方式和数据集的大小。DataLoader将从这个数据集中加载数据。
    batch_size：这是一个整数，指定了每个批次加载的数据项数目。批处理是深度学习中常用的一种技术，可以一次性处理多个数据项，提高计算效率，利用现代计算库（如GPU）的并行计算能力。
    shuffle：这是一个布尔值，指示是否在每个epoch开始时打乱数据。打乱数据有助于模型泛化能力的提高，防止模型过度适应训练数据的顺序，导致训练效果不佳。在这段代码中，shuffle被设置为False，意味着数据将按照在数据集中的顺序加载，不会进行打乱。
    DataLoader的作用和意义：
    批处理（Batching）：DataLoader能自动将数据集分割成多个批次，每个批次包含指定数量的数据项。这对于使用批梯度下降法等优化算法在训练模型时非常重要。
    数据打乱（Shuffling）：通过打乱数据顺序，DataLoader有助于模型学习到更加泛化的特征，避免对数据顺序的依赖，提高模型的泛化能力。
    并行加载（Parallelism）：DataLoader支持多进程加载数据，可以显著提高数据准备的速度，减少模型训练时的等待时间。
    自动化和灵活性：DataLoader提供了一种标准化的方式来加载数据，无论是简单还是复杂的数据集，都可以通过实现Dataset类并使用DataLoader来高效地加载。它还支持自定义的数据处理和增强方法，使得数据加载过程更加灵活。
    总的来说，DataLoader在PyTorch中的作用和意义在于提供了一个高效、灵活且易于使用的数据加载工具，它通过批处理、数据打乱和并行加载等功能，帮助研究者和开发者在训练深度学习模型时节省时间，提高效率，并促进模型性能的提升。
    """
    def __init__(self, w1, word_2_index, all_data):# 构造函数，初始化MyDataset对象。
        self.w1 = w1# w1是预先训练好的词向量或某种映射，用于将单词转换成向量。5319行128列的向量，128是因为我设置的词向量大小是128，即1个单词用1*128的向量来表示。5319是因为总共用5319个不同的字符。
        self.word_2_index = word_2_index# word_2_index是一个字典，用于将单词映射到其索引值。一个字典，共5319个键值对，键是字符，值是从0-5319的数字，5319是因为训练的数据中总共用5319个不同的字符。
        self.all_data = all_data# all_data包含了所有的训练数据，是一个list，包含train_num=6000个元素，每个元素是一首古诗
        #print(1)
        # all_data是一个list，包含train_num个元素，每个元素是一个str，这个str是一首诗
    def __getitem__(self, index):# 特殊方法，使得MyDataset实例可以使用索引操作符（如dataset[index]）来获取数据。
        #getitem用于把文字变成词向量
        a_poetry = self.all_data[index]# 根据索引获取单条数据，这里是一首诗。当index=0时，是：‘仓储十万发关中，伟绩今时富郑公。有米成珠资缓急，此心如秤慎初终。’
        a_poetry_index = [self.word_2_index[i] for i in a_poetry]# 将诗中的每个单词转换为其索引值。
        xs = a_poetry_index[:-1] # 创建输入序列xs，包含除最后一个单词外的所有单词的索引。一个列表，包含31个元素，因为排除了最后一个单词。
        ys = a_poetry_index[1:]# 创建目标序列ys，包含从第二个单词开始的所有单词的索引。一个列表，包含31个元素，因为排除了第一个单词。
        xs_embedding = self.w1[xs]# 使用w1将输入序列xs中的索引转换为词向量。一个31行128列的向量，一个索引对应一个词向量，一个词向量用一行来表示。相当于返回了这首诗的词向量。
        #print("")
        return xs_embedding, np.array(ys).astype(np.int64)# 返回一对数据：输入序列的词向量和目标序列的索引，后者转换为int64类型。

    def __len__(self):# 特殊方法，返回数据集中数据项的总数。
        #a=len(self.all_data)
        return len(self.all_data) # 返回all_data中数据的总数。即train_num=6000，训练的古诗的数量