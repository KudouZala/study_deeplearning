import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from torch.utils.data import Dataset, DataLoader
from dataset_def import MyDataset
import matplotlib.pyplot as plt

class MyModel(nn.Module):
    # 根据模型的需要，我们要构建一个LSTM层，从而将5*10*10（5是batch_size,10是输入的是10s内的数据，10是一秒的数据对应10个数字）转化为hidden层，（5*31*64），这里可以设定hidden_num = 64
    # hidden层（5*10*20）经过flatten或叫roll之后，变成（50*20）的向量，随后将hidden_flatten通过linear层（20*1），得到一个50*1的向量（50个字数字），即最终预测结果
    # 通过比较预测结果和真实值的差，我们可以计算损失，从而反向传播修改模型
    """
    模型流程解析:
    LSTM层转换：您正确地指出，一个LSTM层可以将输入张量从形状[5, 10, 10]转换为形状[5, 10, 20]。这里，5是批量大小（batch_size），10是序列长度（10s的数据），10和20分别是输入和隐藏层的特征维度。设置hidden_num = 20确实意味着隐藏状态的每个元素（或每个时间步）将被编码为一个20维的向量。
    Flatten操作：然后，您提到的flatten（或roll）操作是将LSTM层的输出从形状[5, 10, 10]转换为形状[50, 10]。这里，50是批量大小和序列长度乘积（即5*10），表示现在有50个独立的序列元素，每个都有一个20维的表示。
    Linear层转换：通过线性层（linear）将每个20维的表示映射到一个更大的维度因此，线性层的输出形状为[50, 1]，每行表示数据

    关于PyTorch张量展平的解释:
    当我们说PyTorch将输入张量的第0维和第1维合并展平时，我们是指它将这两个维度中的元素组合成一个单一的维度，而不改变这些元素的总数。在您的例子中，第0维（批量大小，5）和第1维（序列长度，31）被合并，导致一个新的维度，其大小是这两个维度大小的乘积（即155）。这个操作不是将整个张量变为一维（这将是完全展平），而是将特定的维度合并。这对于准备数据以供例如全连接层处理非常有用，因为全连接层期望其输入是二维的（其中一个维度是特征维度）。
    在实践中，这意味着每个序列（或批处理中的每个项目）的所有时间步现在被视为独立的数据点，但每个数据点仍然保留其原始的特征维度（在这个案例中是64）。这使得模型可以在每个时间步独立地做出预测，而不是必须一次性处理整个序列。
    """
    def __init__(self, params, data):
        super(MyModel, self).__init__()
        self.all_data = data
        self.epochs = params["epochs"]
        self.lr = params["lr"]
        self.optimizer = params["optimizer"]
        self.input_size= 10#每s有10个特征
        self.hidden_num = params["hidden_num"]#LSTM输出后的特征数量
        self.batch_size = params["batch_size"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_num, num_layers=2, batch_first=True,bidirectional=False)
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten(0,1)
        self.linear = nn.Linear(self.hidden_num, 1)
        self.criterion= torch.nn.MSELoss()

    def forward(self, x, h_0=None, c_0=None):
        #x是输入的向量：5*10*10
        # LSTM 层的输出形状: (batch_size, seq_length, hidden_size)
        if h_0 == None or c_0 == None:
            # 如果未提供初始的隐藏状态h_0或细胞状态c_0，则创建全零的张量作为初始状态。
            h_0 = torch.tensor(np.zeros((2, x.shape[0], self.hidden_num), dtype=np.float32))
            # 创建一个形状为[2, batch_size, hidden_num]，数据类型为float32的全零张量作为初始隐藏状态h_0。
            c_0 = torch.tensor(np.zeros((2, x.shape[0], self.hidden_num), dtype=np.float32))
        h_0 = h_0.to(self.device)  # 将初始隐藏状态h_0移动到模型指定的设备上（例如CPU或GPU）。# tensor(2,5,64)
        c_0 = c_0.to(self.device)  # 将初始细胞状态c_0移动到模型指定的设备上。# tensor(2,5,64)
        x = x.to(self.device)  # 将输入的词嵌入xs_embedding移动到模型指定的设备上。#tensor(5,31,128)

        hidden, (h_0, c_0) = self.lstm(x,(h_0,c_0))

        hidden_drop = self.dropout(hidden)
        hidden_flatten = self.flatten(hidden_drop)
        pre = self.linear(hidden_flatten)
        return pre, (h_0, c_0)


    def to_train(self):
        model_result_file = "dianliu_lstm_model.pkl"  # 定义一个字符串变量，指定保存模型的文件名。
        if os.path.exists(model_result_file):  # 检查指定的模型文件是否已经存在。
            return pickle.load(open(model_result_file, "rb"))  # 如果模型文件已存在，则直接加载并返回这个预训练模型，不再进行下面的训练过程。
        dataset = MyDataset(self.all_data)
        dataloader = DataLoader(dataset, self.batch_size)
        optimizer = self.optimizer(self.parameters(), self.lr)  # 创建一个优化器实例，用于更新模型的参数。优化器的类型和学习率从模型的初始化参数中获取。
        self = self.to(self.device)  # 将模型移动到指定的设备上（CPU或GPU），以利用GPU加速（如果可用）。
        tongji=[]
        for e in range(self.epochs):
            for batch_index, (batch_x, batch_y) in enumerate(dataloader):# 在每个epoch内部，遍历数据加载器返回的每个批次的数据。
                self.train()
                optimizer.zero_grad()  # 梯度清零
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                pre,_ = self(batch_x)
                batch_y= batch_y.reshape(-1)
                loss =self.criterion(pre,batch_y)
                loss.backward()  # 梯度反传 , 梯度累加, 但梯度并不更新, 梯度是由优化器更新的
                optimizer.step()  # 使用优化器更新梯度

                if batch_index % 100 == 0:  # 每处理100个批次，打印一次当前的损失值。
                    print(f"loss:{loss:.3f}")
                    a=loss.item()
                    tongji.append(a)

        pickle.dump(self, open(model_result_file, "wb"))  # 训练完成后，将训练好的模型保存到文件中，以便将来可以重新加载和使用。
        return self,tongji  # 返回训练完成的模型实例。

