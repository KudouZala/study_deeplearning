import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from dataset_def import MyDataset
from model_def import MyModel

# 假设您已经有了 MyModel, MyDataset, 和 DataLoader 的定义
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def read_csv_to_numpy_array(file_path):
    # 使用 pandas 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 确保 DataFrame 的形状符合预期，即 (num_samples, 11)
    if df.shape[1] != 11:
        raise ValueError("CSV文件的列数不符合预期。确保文件有11列数据。")

    # 将 DataFrame 转换为 numpy 数组
    data_array = df.to_numpy()

    return data_array


# 示例使用


if __name__ == "__main__":
    file_path = 'simulated_data.csv'  # 替换为您的 CSV 文件路径
    data = read_csv_to_numpy_array(file_path)#1000行11列，第一列是

    params = {
        "batch_size": 5,  # batch大小
        "epochs": 50,  # epoch大小
        "lr":  0.00001,  # 学习率
        "hidden_num": 60,  # 隐层大小
        "embedding_num": 128,  # 词向量维度
        "train_num": 2000,  # 训练的故事数量, 七言古诗:0~6290, 五言古诗:0~2929
        "optimizer": torch.optim.AdamW,  # 优化器 , 注意不要加括号
        "batch_num_test": 100,  # 多少个batch 打印一首古诗进行效果测试
        "input_size":10,#每秒多少个特征
    }

    model = MyModel(params,data)  # 模型定义
    model,loss_tongji = model.to_train()  # 模型训练

    plt.figure(figsize=(10, 5))
    plt.plot(loss_tongji, label='Training Loss')
    plt.title('Loss During Training')
    plt.xlabel('每100个batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    # 示例输入，假定 batch_size=1，seq_length=10
    x_dummy = torch.rand(1, 10, params["input_size"])

    # 获取模型预测结果
    output = model(x_dummy)
    print(output)  # 输出的预测结果

