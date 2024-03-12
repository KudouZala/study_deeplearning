import torch
from torch.utils.data import DataLoader,TensorDataset  # 用于加载数据集
from torchvision import transforms  # 用于数据预处理
from torchvision.datasets import MNIST  # 导入MNIST数据集
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # 用于绘图
import pandas as pd
#出自pytorch手写数字识别：https://www.bilibili.com/video/BV1GC4y15736/?spm_id_from=333.337.top_right_bar_window_default_collection.content.click&vd_source=2098929c3c4c88c0fe4fb524026d2a33

class Net(torch.nn.Module):
    """
    Net类定义了一个简单的全连接神经网络，包含四个全连接层（也称为线性层）。这个网络设计用于接收展平的MNIST图像数据作为输入（每个图像转换为一个大小为28x28=784的一维向量），并输出一个大小为10的向量。每个输出向量的元素对应于模型对该图像分别属于10个可能数字类别（0到9）的预测概率。
    第一个全连接层（fc1）：接收一个784维的向量（MNIST图像展平后的结果）作为输入，输出一个64维的向量。
    第二和第三全连接层（fc2和fc3）：进一步处理数据，每层都有64个输入节点和64个输出节点，增加模型的表达能力。
    第四个全连接层（fc4）：将网络的最终输出缩减到一个10维的向量，每个维度对应一个数字类别的预测概率。
    """
    def __init__(self):
        super().__init__()
        """
        super().__init__()这行代码的作用是调用父类（在这个例子中是torch.nn.Module）的构造函数。
        在Python中，super()函数返回了一个代表父类的临时对象，允许你调用该父类的方法。此处的__init__()方法是类的构造函数，负责进行类的初始化。
        在继承自torch.nn.Module的自定义模块中调用super().__init__()是非常重要的，因为它执行了父类torch.nn.Module的初始化代码，这包括设置一些在模块背后自动运行的机制，使得诸如参数管理、模型到设备的迁移（例如，GPU）、序列化等功能能够正常工作。
        简而言之，这个Net类定义了一个用于MNIST数字分类的简单全连接神经网络，super().__init__()确保了这个类能够正确继承和使用torch.nn.Module类的所有功能和特性。
        """
        # 定义第一个全连接层，输入为28*28个特征（MNIST图像的尺寸展平后的大小），输出为64个节点
        self.fc1 = torch.nn.Linear(28 * 28, 64)
        # 定义第二个全连接层，输入为64个节点，输出也为64个节点
        self.fc2 = torch.nn.Linear(64, 64)
        # 定义第三个全连接层，输入输出同上
        self.fc3 = torch.nn.Linear(64, 64)
        # 定义第四个全连接层，输入为64个节点，输出为10个节点（对应MNIST的10个类别）
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        # 通过第一个全连接层后应用ReLU激活函数
        x = torch.nn.functional.relu(self.fc1(x))
        # 通过第二个全连接层后再次应用ReLU激活函数
        x = torch.nn.functional.relu(self.fc2(x))
        # 通过第三个全连接层后再次应用ReLU激活函数
        x = torch.nn.functional.relu(self.fc3(x))
        # 最后通过第四个全连接层，并应用log_softmax激活函数，用于多分类的概率输出
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x

def get_data_loader(is_train):
    # 定义数据转换操作：将数据转换为PyTorch张量
    to_tensor = transforms.Compose([transforms.ToTensor()])
    # 加载MNIST数据集，is_train标志用于指定是加载训练集还是测试集
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    # 返回一个数据加载器，设置batch_size为15，打乱数据
    return DataLoader(data_set, batch_size=15, shuffle=True)

def get_data_loader2(is_train, batch_size=15):
    # 加载CSV数据
    df = pd.read_csv('mnist-demo.csv', skiprows=1)  # 跳过第一行
    # 将数据分为特征和标签
    X = df.iloc[:, 1:].values  # 除去第一列标签的数据
    y = df.iloc[:, 0].values  # 第一列是标签

    # 将数据转换为torch.FloatTensor
    X = torch.FloatTensor(X) / 255.  # 通常对数据进行归一化处理
    y = torch.LongTensor(y)

    # 使用train_test_split从数据中分出训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if is_train:
        dataset = TensorDataset(X_train, y_train)
    else:
        dataset = TensorDataset(X_test, y_test)

    # 数据转换操作可以按需求添加，这里我们直接使用原始数据
    # 创建DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def evaluate(test_data, net):#评估训练好的模型正确率的函数
    # 初始化正确预测的数量和总数量
    n_correct = 0
    n_total = 0
    with torch.no_grad():  # 不计算梯度，减少计算和内存消耗
        for (x, y) in test_data:
            # 将输入数据展平并通过网络
            outputs = net.forward(x.view(-1, 28 * 28))
            for i, output in enumerate(outputs):
                # 对于每个输出，如果预测类别（最大概率的类别）等于真实类别，则正确计数加一
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    # 返回准确率
    return n_correct / n_total

def main():
    # 获取训练和测试数据加载器
    train_data = get_data_loader2(is_train=True,batch_size=10)
    test_data = get_data_loader2(is_train=False)

    # 实例化网络模型
    net = Net()

    # 在训练开始前评估模型的初始准确率
    print("initial accuracy:", evaluate(test_data, net))

    # 使用Adam优化器，设置学习率为0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(10):  # 训练10个epoch,注意：每个epoch所有的训练数据都会遍历，
        for (x, y) in train_data:
            net.zero_grad()  # 清除之前的梯度
            """
            由于梯度是累积的，如果不在每次迭代开始前清零，那么从上一次的反向传播中累积的梯度将会与本次迭代的梯度相加。这将导致每次迭代结束时，梯度不仅反映当前批次的数据，还包含了之前所有批次的梯度信息，从而使得梯度更新方向和大小变得不正确，影响训练过程的稳定性和最终模型的性能。
            调用.zero_grad()清除梯度是为了确保每次反向传播计算得到的梯度只反映当前批次的数据，使得每次参数更新都是基于最新的数据进行的。这对于模型正确学习和收敛至关重要。
            总之，.zero_grad()的调用是为了保证模型训练的正确性和高效性，确保每一步梯度下降都是准确的，从而有助于模型以正确的方向更新参数，最终达到更好的训练效果。
            """
            output = net.forward(x.view(-1, 28 * 28))  # 获取模型输出
            loss = torch.nn.functional.nll_loss(output, y)  # 计算损失
            #上面这行代码计算了预测输出output和真实标签y之间的损失。这里使用了负对数似然损失函数（nll_loss），这是分类问题中常用的损失函数之一，尤其是当模型的输出通过了log_softmax激活函数时。y是相应的真实标签，这些标签也是从train_data加载器中获得的。
            loss.backward()  # 反向传播计算梯度
            #上行代码通过调用损失张量的.backward()方法，PyTorch自动计算所有可训练参数（模型权重）的梯度，并将这些梯度存储在参数的.grad属性中。这是利用自动微分技术实现的，是训练神经网络的关键步骤之一。
            optimizer.step()  # 更新模型参数
            #上面这行代码利用之前计算得到的梯度来更新模型的参数。optimizer是一个优化器对象，其负责按照特定的优化算法（如SGD、Adam等）调整模型参数，以减小损失函数的值。在调用.step()方法之前，通常需要调用.zero_grad()清除上一次迭代所遗留的旧的梯度，因为梯度是累积计算的。

        # 每个epoch结束后评估模型准确率
        #在每个训练周期（epoch）结束后，这行代码通过调用evaluate函数来评估当前模型在测试数据集test_data上的性能。这里的evaluate函数没有在代码片段中定义，但通常它会计算模型的准确率或其他性能指标。这有助于监控训练过程，了解模型是否在学习并改进其预测能力。
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))

    # 绘制一些测试图像及其预测标签
    for (n, (x, _)) in enumerate(test_data):
        if n > 3:  # 只绘制前4个样本
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28 * 28)))  # 获取模型预测结果
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))  # 显示图像
        plt.title("prediction: " + str(int(predict)))  # 显示预测的类别
    plt.show()  # 显示图像

if __name__ == "__main__":
    main()  # 运行主函数
