# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from gensim.models.word2vec import Word2Vec
from torch.utils.data import Dataset, DataLoader
import sys
from model_def import MyModel
from dataset_def import MyDataset

# 出自：https://www.bilibili.com/video/BV1G54y177iw?p=8&spm_id_from=pageDriver&vd_source=2098929c3c4c88c0fe4fb524026d2a33

if __name__ == "__main__":#主程序
    params = {
        "batch_size": 5,  # batch大小
        "epochs": 100,  # epoch大小
        "lr": 0.003,  # 学习率
        "hidden_num": 64,  # 隐层大小
        "embedding_num": 128,  # 词向量维度
        "train_num": 2000,  # 训练的故事数量, 七言古诗:0~6290, 五言古诗:0~2929
        "optimizer": torch.optim.AdamW,  # 优化器 , 注意不要加括号
        "batch_num_test": 100,  # 多少个batch 打印一首古诗进行效果测试
    }
    """
    这些参数是用于配置神经网络训练过程中的关键设置。每个参数都有其特定的作用，下面是对这些参数的解释以及如何调整它们来减少训练时间的建议：

    batch_size：批大小。这是每次训练迭代中网络同时处理的数据样本数量。较大的批大小可以增加计算效率（特别是在GPU上），但也会增加内存需求。调整建议：增加batch_size可以减少训练迭代的次数，从而可能减少总的训练时间，但要注意不要超出硬件的内存限制。

    epochs：训练周期。一个周期表示整个训练数据集被遍历一次。调整建议：减少epochs的数量会直接减少训练时间，但可能会影响模型的最终性能。

    lr（学习率）：学习率。它控制着权重调整的幅度。较高的学习率可能会导致训练过程快速收敛，但也可能导致过度拟合或不稳定。调整建议：调整学习率本身不直接减少训练时间，但合适的学习率可以帮助模型更快收敛。

    hidden_num：隐藏层大小。这个参数设置了LSTM层中隐藏状态的维度。调整建议：减少hidden_num会减少模型的参数数量，可能会稍微加快训练速度，但也可能影响模型性能。

    embedding_num：词向量维度。这决定了词嵌入层输出向量的大小。调整建议：减少embedding_num同样会减少模型的参数数量，可能加快训练，但可能会降低模型捕捉词义的能力。

    train_num：训练的数据量。这个参数限制了用于训练的数据样本数量。调整建议：减少train_num会直接减少训练所需处理的数据量，从而减少训练时间，但也会减少模型学习的信息量。

    optimizer：优化器。这个参数选择了用于训练网络的优化算法。调整建议：选择不同的优化器不会直接影响训练时间，但某些优化器可能会帮助模型更快收敛。

    batch_num_test：多少个batch打印一首古诗进行效果测试。这是一个用于监控训练过程的参数，它决定了多频繁地在训练过程中输出模型的生成结果。调整建议：这个参数不影响训练速度，但减少打印操作可以稍微减少一些计算开销。

总结：要减少训练时间，最直接的方法是减少epochs（可能会牺牲一些模型性能），增加batch_size（但要注意硬件限制），或减少train_num（使用更少的数据）。同时，适当调整hidden_num和embedding_num可以减少模型复杂度，从而可能加快训练速度，但要权衡模型性能的损失
    """
    model = MyModel(params)  # 模型定义
    model = model.to_train()  # 模型训练
    model.generate_poetry_based_on_start_word("啊")
    a="我爱你啊"
    print(a[1])
    model.generate_poetry_acrostic2("爱陈品初")  # 测试藏头诗
