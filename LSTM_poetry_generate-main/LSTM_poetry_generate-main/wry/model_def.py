# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from gensim.models.word2vec import Word2Vec
from torch.utils.data import Dataset, DataLoader
import sys
from dataset_def import MyDataset

# 在自然语言文字处理中使用LSTM，假设我一首诗是4个7字话，那么算上逗号和句号，一共30个东西。那么LSTM具有记忆功能，通过前一个字预测下一个字的时候，记住了该字，并影响到了下一次预测。
def split_text(file="poetry_7.txt", train_num=6000):  # 处理txt文件，
    all_data = open(file, "r", encoding="utf-8").read()  #str 将txt文件读取为字符串str格式,这部分代码打开名为file的文件，以只读模式（"r"）并指定编码为"utf-8"。"utf-8"编码支持大多数语言的字符，包括一些特殊字符。
    #.read()：这个方法从打开的文件中读取所有内容，并返回一个字符串。这意味着，all_data变量将包含文件的完整内容作为一个大字符串。
    with open("split_7.txt", "w", encoding="utf-8") as f:#split_7.txt是我们要保存的文件夹
        """
        这行代码使用with语句打开（或创建）一个名为"split_7.txt"的文件，用于写入（"w"），并指定编码为"utf-8"。with语句确保文件在操作完成后正确关闭。变量f是打开文件的文件对象，用于文件操作。
        该语句的作用是打开（或创建）一个新文件，准备写入处理后的数据。
        """
        split_data = " ".join(all_data)
        """
        这个表达式使用一个空格" "作为分隔符，将all_data字符串中的每个字符连接成一个新的字符串。因为all_data是一个字符串，所以这个操作实质上是在字符串中的每个字符之间插入一个空格。
        split_data变量存储了这个加入空格后的新字符串。str
        """
        f.write(split_data)#这行代码将split_data（处理后的字符串）写入之前以"w"模式打开的文件中。这实际上是将加工后的文本内容保存到"split_7.txt"文件中。
    return split_data[:train_num * 64]#这行代码返回处理后的字符串split_data的前train_num * 64个字符。train_num是一个参数，表示你想要多少个训练样本，而64可能是每个样本的固定长度。因此，这部分代码意在从处理后的全文中提取一部分作为训练数据。
    #解释：为什么是*64，因为假设我们想训练10首诗，每首诗的长度是4*（7+1）=32，4是4句话，7是汉字数量，1是符号，但是由于添加了空格，因此是32*2=64.


# 定义一个训练词向量的函数，可以设置词向量的大小，默认是128，还可以指定分割文件和原始文件的名称，以及训练的数据量
def train_vec(vector_size=128, split_file="split_7.txt", org_file="poetry_7.txt", train_num=6000):#train_num=6000表示用6000首古诗来做词向量训练
    #vector_size表示词向量的大小，一般是一个汉字用一个1*128的向量表示，
    param_file = "My_word_vec_wry.pkl"  # 设置保存词向量模型参数的文件名。
    # 从原始文件中读取数据，将其分割成行，并只取前train_num行作为训练数据。
    org_data = open(org_file, "r", encoding="utf-8").read().split("\n")[:train_num]#这个变量包含的是从原始文件org_file（如诗歌文本）中读取的数据。这里直接读取文件内容，按行分割，然后取前train_num行。这些数据可能是未经过任何预处理的原始文本数据。
    #org_data是一个list，包含train_num个元素，每个元素是一个str，这个str是一首诗
    # 如果分割文件存在，则从该文件读取数据，同样只取前train_num行作为训练数据。
    #if os.path.exists(split_file):
        #all_data_split = open(split_file, "r", encoding="utf-8").read().split("\n")[:train_num]
        #all_data_split返回的是一个list，包含train_num个元素，每个元素是一个str，这个str是一首诗，但是相比org_data，每个字符之间用空格隔开
    #else:
        # 如果分割文件不存在，则调用split_text函数生成分割后的数据，并只取前train_num行。
    all_data_split = split_text().split("\n")[:train_num]
    """
    org_data：这个变量包含的是从原始文件org_file（如诗歌文本）中读取的数据。这里直接读取文件内容，按行分割，然后取前train_num行。这些数据可能是未经过任何预处理的原始文本数据。
    all_data_split：这个变量则根据情况可以包含两种数据。如果存在一个已经预处理（例如分词、去除停用词等操作）的文件split_file，它会从这个文件中读取数据；如果这个文件不存在，它会调用split_text函数生成这样的预处理数据。与org_data相同，它也只取前train_num行。总的来说，all_data_split包含的数据是经过某种预处理的，更适合用来训练Word2Vec模型。
    """
    # 如果模型参数文件已经存在，则直接返回原始数据和加载的模型参数。
    if os.path.exists(param_file):
        return org_data, pickle.load(open(param_file, "rb"))
    #使用all_data_split作为训练数据，创建Word2Vec模型。
    models = Word2Vec(all_data_split, vector_size=vector_size, workers=7, min_count=1)
    """
    Word2Vec：这是一个用于训练词向量的模型，它可以将词转换为向量形式，从而能够捕捉到词之间的语义关系。在这段代码中，Word2Vec使用all_data_split作为输入数据，即每个字符用空格隔开的，根据这些数据训练出词向量模型。
    注意：Word2Vec这个函数有许多的参数可以调节，调节这些参数可以优化
    """
    # 将训练好的模型参数保存到文件中。
    pickle.dump([models.syn1neg, models.wv.key_to_index, models.wv.index_to_key], open(param_file, "wb"))
    """
    Word2Vec：这是一个用于训练词向量的模型，它可以将词转换为向量形式，从而能够捕捉到词之间的语义关系。在这段代码中，Word2Vec使用all_data_split作为输入数据，根据这些数据训练出词向量模型。
    pickle.dump 和 pickle.load：这两个函数用于将Python对象序列化（保存）到文件和从文件反序列化（加载）Python对象。在这段代码中，使用pickle.dump将训练好的Word2Vec模型参数保存到文件，以便将来可以通过pickle.load快速加载这些参数，而不需要重新训练模型。
    通过这种方式，代码实现了一个功能，即根据原始文本数据训练出词向量模型，并将模型参数保存起来，使得在未来需要使用这些词向量时，可以直接加载已保存的参数，而无需重新训练，从而节省时间和资源。
    """
    #print(1)
    # 返回原始数据和模型的三个主要参数：负采样的词向量、词到索引的映射、索引到词的映射。
    return org_data, (models.syn1neg, models.wv.key_to_index, models.wv.index_to_key)
    """
    负采样的词向量 (models.syn1neg): 5319*128的向量，Word2Vec模型通过训练过程学习到的词向量存储在模型的内部结构中。当使用负采样（Negative Sampling）优化算法时，syn1neg是存储负采样权重的矩阵。负采样是一种训练优化方法，旨在减少计算复杂度，通过随机选择“负”样本来更新权重，而不是使用整个词汇表。这种方法特别适用于大词汇表的情况，可以显著加快训练速度。syn1neg中的每一行对应于词汇表中每个词的词向量，这些词向量捕捉了词与词之间的语义关系。
    词到索引的映射 (models.wv.key_to_index): dict字典，5319个键值对，这是一个从词到其在词向量矩阵中索引的映射。在Word2Vec模型中，每个唯一的词都被分配了一个索引，用于在词向量矩阵中快速查找该词的向量。key_to_index是一个字典，其中的键是词汇，值是这些词汇对应的索引。通过这个映射，可以方便地根据词找到其对应的词向量。
    索引到词的映射 (models.wv.index_to_key): list列表，5319个元素，这与key_to_index相反，是从索引到词的映射。这是一个列表，列表中的每个位置对应一个索引，而该位置存储的值是对应的词。这个映射允许我们根据索引找到相应的词。在某些操作中，如寻找与给定词最相似的词时，可能会首先基于向量相似度找到最接近的词向量的索引，然后使用这个映射来确定这些索引对应的词。
    这三个组件共同支持Word2Vec模型的主要功能，如词向量的查找、相似词的检索等。通过这些组件，可以有效地利用训练好的模型来探索和分析词汇之间的语义关系。
    """
class MyModel(nn.Module):
    #根据模型的需要，我们要构建一个LSTM层，从而将5*31*128（5是batch_size,31是输入的是31个字符（一首古诗除了最后一个字符），128是一个字符对应一个1*128的词向量）转化为hidden层，（5*31*64），这里可以设定hidden_num = 64
    #hidden层（5*31*64）经过flatten或叫roll之后，变成（155*64）的向量，随后将hidden_flatten通过linear层（64*3542），得到一个155*5319的向量（155个字符，下一个字符的概率值），该向量可以转换为一个155*1的向量，即最终预测结果
    #通过比较预测结果和真实值的差，我们可以计算损失，从而反向传播修改模型
    """
    模型流程解析:
    LSTM层转换：您正确地指出，一个LSTM层可以将输入张量从形状[5, 31, 128]转换为形状[5, 31, 64]。这里，5是批量大小（batch_size），31是序列长度（一首古诗的字符数），128和64分别是输入和隐藏层的特征维度。设置hidden_num = 64确实意味着隐藏状态的每个元素（或每个时间步）将被编码为一个64维的向量。
    Flatten操作：然后，您提到的flatten（或roll）操作是将LSTM层的输出从形状[5, 31, 64]转换为形状[155, 64]。这里，155是批量大小和序列长度乘积（即5*31），表示现在有155个独立的序列元素，每个都有一个64维的表示。
    Linear层转换：通过线性层（linear）将每个64维的表示映射到一个更大的维度（在这个例子中是3542，可能是一个误解，因为您后面提到了5319），这代表词汇表的大小或可能的下一个字符的数量。因此，线性层的输出形状为[155, 5319]（或[155, 3542]，根据实际的词汇表大小），每行表示一个字符的下一个字符的概率分布。
    预测结果转换：最后，您提到将得到的向量转换为[155, 1]的向量，即最终的预测结果。实际上，这个步骤通常涉及选择每行（对每个时间步）概率最高的索引，这代表模型预测的下一个字符的索引。这通常通过argmax操作完成，而不是直接转换。
    关于PyTorch张量展平的解释:
    当我们说PyTorch将输入张量的第0维和第1维合并展平时，我们是指它将这两个维度中的元素组合成一个单一的维度，而不改变这些元素的总数。在您的例子中，第0维（批量大小，5）和第1维（序列长度，31）被合并，导致一个新的维度，其大小是这两个维度大小的乘积（即155）。这个操作不是将整个张量变为一维（这将是完全展平），而是将特定的维度合并。这对于准备数据以供例如全连接层处理非常有用，因为全连接层期望其输入是二维的（其中一个维度是特征维度）。
    在实践中，这意味着每个序列（或批处理中的每个项目）的所有时间步现在被视为独立的数据点，但每个数据点仍然保留其原始的特征维度（在这个案例中是64）。这使得模型可以在每个时间步独立地做出预测，而不是必须一次性处理整个序列。
    """
    def __init__(self, params):
        #这里我们用一个params来表示要传入的参数：
        super().__init__()
        self.all_data, (self.w1, self.word_2_index, self.index_2_word) = train_vec(vector_size=params["embedding_num"],
                                                                                   train_num=params["train_num"])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hidden_num = params["hidden_num"]#设置hidden_size,即隐藏层的大小
        self.batch_size = params["batch_size"]
        """
        在您的代码中，确实只有在初始化Poetry_Model_lstm类时通过params["batch_size"]设置了self.batch_size，看起来似乎之后并没有直接使用self.batch_size来显式地调整xs_embedding的形状。这里涉及到PyTorch如何处理批量数据和模型的自动推断机制。
        批量数据处理
        当使用DataLoader来加载数据时，它会自动根据每个批次的大小（batch_size）来组织数据。这意味着DataLoader会自动将数据集中的样本分组成批次，每个批次包含batch_size个样本。这是在调用DataLoader(dataset, self.batch_size)时发生的，其中self.batch_size就是您在模型初始化时设置的批次大小。
        自动形状推断
        在您的模型中，即使没有显式使用self.batch_size来调整xs_embedding或其他张量的形状，PyTorch模型在前向传播时也能够自动处理不同的批次大小。这是因为模型的层（例如nn.LSTM和nn.Linear）设计为能够接受任意批次大小的输入——只要输入的其余维度（如序列长度和特征维度）与模型期望的一致。
        例如，当您的DataLoader生成一个形状为[batch_size, sequence_length, embedding_dim]的批次数据并传递给模型时，nn.LSTM层自动识别批次中有batch_size个序列，每个序列长度为sequence_length，每个时间步的特征维度为embedding_dim。因此，即使在代码中没有直接引用self.batch_size来修改数据形状，模型在执行前向传播时也能正确处理批次中的数据。
        结论
        您的代码通过DataLoader和模型的设计自然而然地处理了批次数据，而无需手动指定或调整批次大小。这是深度学习框架的一大优点，它允许开发者专注于模型结构的设计，而不是数据批次的具体管理。您提到的xs_embedding的形状变化实际上是由于DataLoader自动将数据组织成批次，以及模型层自动处理任意批次大小的能力共同作用的结果。
        """
        self.epochs = params["epochs"]
        self.lr = params["lr"]
        self.optimizer = params["optimizer"]
        self.word_size, self.embedding_num = self.w1.shape#w1是5319*128的向量大小,因此word_size是int5319，embedding_num是int128

        #定义LSTM层：
        self.lstm = nn.LSTM(input_size=self.embedding_num, hidden_size=self.hidden_num, batch_first=True, num_layers=2,
                            bidirectional=False)
        #注意这个LSTM层的定义，有很多参数可以调节，从而调整最终的效果

        #dropout层，随即让一些很神经元失活，让神经网络不要过拟合，
        self.dropout = nn.Dropout(0.3)  # dropout的参数表示随机失活率，有了随机失活，生成的古诗不会唯一了
        """
        至于为什么加入dropout后，生成的古诗不会唯一，这主要是因为：
        随机性：由于dropout在每次训练迭代中随机"关闭"一部分神经元，这引入了随机性。因此，即使是同一个输入，在不同的训练迭代中，由于受影响的神经元不同，模型的内部表示和输出也可能不同。
        泛化能力：dropout通过减少神经元的依赖关系，提高了模型的泛化能力。这意味着模型在面对相同的输入时，能够从多个角度理解和解释输入数据，从而有可能生成不同的输出（即使是在确定性的推断阶段，如果保留dropout）。
        创造性：在生成任务，如生成藏头诗这类文本生成任务中，模型的一些随机性实际上可以增加输出的多样性，从而使生成的文本更具创造性和新颖性。dropout作为一种引入随机性的手段，可以帮助模型避免每次都生成相同或高度相似的输出。
        通常，在模型推断（inference）阶段生成文本时，会禁用dropout以获得模型的确定性输出。但是，如果在推断阶段保留dropout（或以较低的dropout率应用），可以增加输出的多样性，使得同一输入在不同时间的推断结果略有不同，这种做法在某些创造性任务中可能是有益的。
        总的来说，dropout通过引入随机性，不仅帮助模型防止过拟合，提高泛化能力，还可能在特定应用中增加输出的多样性和创造性。
        """

        """
        如果模型已经训练完成，是否对于同一个输入会有不同的输出，主要取决于模型在推断（inference）阶段的行为设置：
        如果在推断阶段禁用了dropout：在深度学习模型，尤其是LSTM等循环神经网络中，通常在训练阶段使用dropout来防止过拟合，而在推断阶段禁用dropout。这是因为在推断阶段，我们通常希望模型能给出稳定且一致的输出。如果dropout被禁用，给定相同的输入，模型将产生相同的输出，因为模型的权重是固定的，且没有引入任何形式的随机性。
        如果在推断阶段保留了dropout：虽然不常见，但如果你选择在推断阶段保留dropout，或者使用其他引入随机性的技术（如在生成文本时采用随机采样策略而不是贪心策略），那么即使对于同一个输入，模型也可能给出不同的输出。这种做法可以增加输出的多样性，但可能会牺牲一致性和预测的确定性。
        使用随机采样或温度采样：在文本生成任务中，即使在推断阶段禁用了dropout，模型也可能产生不同的输出，这取决于输出层采用的策略。例如，使用随机采样（从模型的输出分布中随机选择下一个词）或温度采样（调整模型输出分布的“温度”，使其更倾向于高概率词或更加平均）时，即使是相同的输入，也可能导致不同的输出。
        总结来说，如果你的模型已经训练好了，对于同一个输入是否会有不同的输出主要取决于推断阶段的设置。在大多数情况下，为了保证输出的一致性和可预测性，会在推断阶段禁用dropout和其他随机性技术，从而确保对于相同的输入总是产生相同的输出。但是，特定的应用场景可能会故意引入随机性，以提高输出的多样性或创造性。
        """
        #定义flatten层
        self.flatten = nn.Flatten(0, 1)
        """
        作用：nn.Flatten是一个层，用于将多维的输入张量（tensor）展平成更低维度的张量。在神经网络中，经常需要在卷积层（处理多维数据）之后和全连接层（期望二维数据，即[批大小, 特征数]）之前进行展平操作。
        参数：nn.Flatten(0, 1)中的参数0和1表示展平操作的起始维度和结束维度。在这个例子中，它告诉PyTorch将输入张量的第0维和第1维（通常是批处理维度和序列长度维度）合并展平。这对于处理例如序列数据（如文本或时间序列）时非常有用，当你想要将序列的每个元素都视为独立的特征，但同时保留批处理的概念时。
        关于PyTorch张量展平的解释:
        当我们说PyTorch将输入张量的第0维和第1维合并展平时，我们是指它将这两个维度中的元素组合成一个单一的维度，而不改变这些元素的总数。在您的例子中，第0维（批量大小，5）和第1维（序列长度，31）被合并，导致一个新的维度，其大小是这两个维度大小的乘积（即155）。这个操作不是将整个张量变为一维（这将是完全展平），而是将特定的维度合并。这对于准备数据以供例如全连接层处理非常有用，因为全连接层期望其输入是二维的（其中一个维度是特征维度）。
        在实践中，这意味着每个序列（或批处理中的每个项目）的所有时间步现在被视为独立的数据点，但每个数据点仍然保留其原始的特征维度（在这个案例中是64）。这使得模型可以在每个时间步独立地做出预测，而不是必须一次性处理整个序列。
        """

        #定义全连接层（线性层）
        self.linear = nn.Linear(self.hidden_num, self.word_size)
        """
        作用：nn.Linear是一个全连接（线性）层，它对输入数据进行线性变换（即y = xA^T + b），其中A是层的权重，b是偏置项。nn.Linear层在神经网络中广泛用于从一组特征映射到另一组特征，是构建深度学习模型的基本组件之一。
        参数：
        self.hidden_num：这是nn.Linear层的输入特征数量，即每个输入向量的大小。在上下文中，它可能表示LSTM或其他类型的循环神经网络层的隐藏状态的维度。
        self.word_size：这是nn.Linear层的输出特征数量，即该层输出向量的大小。在生成文本或类似任务中，这通常对应于词汇表的大小，因为全连接层的输出经常用来预测下一个词的概率分布。
        """

        #定义交叉熵损失函数。这是多分类任务中常用的损失函数，适用于分类问题中的损失计算。
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, xs_embedding, h_0=None, c_0=None):
        # 定义模型的前向传播函数。xs_embedding是输入的词嵌入（5*31*128的向量），h_0和c_0是LSTM层的初始隐藏状态和细胞状态。
        #但要注意：在getitem（MyDataset(Dataset)）中xs_embedding是31*128的向量，改变是发生在当我们创建dataloader时候用的是31*128，从dataloader提取，是5*31*128的向量，以方便我们进行forward前向传播
        """
        在PyTorch中，模型的forward函数不是直接被调用的。相反，当你执行模型实例与输入数据的调用时（即model(input_data)），PyTorch自动调用forward方法。这是通过nn.Module基类中定义的__call__方法实现的，该方法在内部调用forward方法，并进行一些额外的步骤，比如处理钩子。
        在您提供的代码中，forward函数被间接调用在两个地方：
        训练循环中：在to_train方法内部的训练循环中，每当处理一个批次的数据时，通过这行代码调用forward函数：
        pre, _ = self(batch_x_embedding)
        这里的self(batch_x_embedding)实质上是调用了Mymodel实例的forward方法，传递batch_x_embedding作为输入。这个过程在每个训练批次中重复执行，以进行前向传播并计算损失。
        生成诗歌时：在generate_poetry_auto和generate_poetry_acrostic方法中，也通过类似的方式调用了forward函数。例如，在generate_poetry_auto方法中：
        pre, (h_0, c_0) = self(word_embedding, h_0, c_0)
        这行代码同样通过self调用模型实例，传入当前的单词嵌入和先前的隐藏状态和细胞状态，来生成下一个词的预测。
        """
        if h_0 == None or c_0 == None:
            # 如果未提供初始的隐藏状态h_0或细胞状态c_0，则创建全零的张量作为初始状态。
            h_0 = torch.tensor(np.zeros((2, xs_embedding.shape[0], self.hidden_num), dtype=np.float32))
            # 创建一个形状为[2, batch_size, hidden_num]，数据类型为float32的全零张量作为初始隐藏状态h_0。
            c_0 = torch.tensor(np.zeros((2, xs_embedding.shape[0], self.hidden_num), dtype=np.float32))
            # 创建一个形状为[2, batch_size, hidden_num]，数据类型为float32的全零张量作为初始细胞状态c_0。
        """
        对于h_0和c_0的形状为[2, batch_size, hidden_num]，这里的2实际上可能是一个误解。对于标准的LSTM层，初始状态的形状通常是[num_layers * num_directions, batch_size, hidden_size]。其中：num_layers指的是LSTM层的层数。
        num_directions取值为1或2，对应于LSTM是否为双向的（双向LSTM会处理数据的两个方向，因此num_directions为2）。这里由于我们之前设置了是单向，因此是LSTM有2个单向层
        """
        h_0 = h_0.to(self.device) # 将初始隐藏状态h_0移动到模型指定的设备上（例如CPU或GPU）。# tensor(2,5,64)
        c_0 = c_0.to(self.device) # 将初始细胞状态c_0移动到模型指定的设备上。# tensor(2,5,64)
        xs_embedding = xs_embedding.to(self.device) # 将输入的词嵌入xs_embedding移动到模型指定的设备上。#tensor(5,31,128)

        hidden, (h_0, c_0) = self.lstm(xs_embedding, (h_0, c_0))# 通过LSTM层处理输入的词嵌入xs_embedding和初始状态(h_0, c_0)，得到输出hidden和最新的隐藏状态和细胞状态(h_0, c_0)。# tensor(5,31,64)
        #注意上面的h_0和c_0并不意味着初始状态，它也时不断更新的
        hidden_drop = self.dropout(hidden)# 对LSTM层的输出hidden应用dropout，以减少过拟合，提高模型的泛化能力。# tensor(5,31,64)
        hidden_flatten = self.flatten(hidden_drop)# 将dropout后的LSTM输出hidden_drop展平，为全连接层的输入准备。#tensor(155,64)
        pre = self.linear(hidden_flatten) # 将展平后的张量hidden_flatten通过全连接层linear，得到最终的预测结果pre。#tensor(155,5319)

        return pre, (h_0, c_0)# 返回最终的预测结果pre和LSTM的最新隐藏状态和细胞状态(h_0, c_0)。

    def to_train(self):
        # 定义一个名为to_train的方法，用于训练当前的模型。
        model_result_file = "My_Model_lstm_model.pkl"  # 定义一个字符串变量，指定保存模型的文件名。
        if os.path.exists(model_result_file): # 检查指定的模型文件是否已经存在。
            return pickle.load(open(model_result_file, "rb"))# 如果模型文件已存在，则直接加载并返回这个预训练模型，不再进行下面的训练过程。
        dataset = MyDataset(self.w1, self.word_2_index, self.all_data) # 创建一个数据集实例，传入初始化模型时获得的词向量、单词到索引的映射和全部数据。
        dataloader = DataLoader(dataset, self.batch_size)# 使用DataLoader来加载数据集，设置批次大小为self.batch_size。
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
        optimizer = self.optimizer(self.parameters(), self.lr)# 创建一个优化器实例，用于更新模型的参数。优化器的类型和学习率从模型的初始化参数中获取。
        self = self.to(self.device) # 将模型移动到指定的设备上（CPU或GPU），以利用GPU加速（如果可用）。
        """
        为什么要多次使用.to(self.device)？
        在模型的前向传播过程中，确保所有输入数据、模型参数、中间变量等都在同一设备上是很重要的。这样可以避免在CPU和GPU之间频繁移动数据，因为这种移动是非常耗时的操作，会大大降低模型的运行效率。
        当您从外部接收数据（例如，从DataLoader）时，需要将这些数据移动到模型正在运行的设备上。
        如果您在模型内部创建了新的张量（如初始化的隐藏状态和细胞状态），也需要确保这些张量在正确的设备上。
        """
        for e in range(self.epochs): # 开始训练过程，循环遍历每一个epoch。
            for batch_index, (batch_x_embedding, batch_y_index) in enumerate(dataloader):# 在每个epoch内部，遍历数据加载器返回的每个批次的数据。
                self.train() # 将模型设置为训练模式。
                batch_x_embedding = batch_x_embedding.to(self.device)
                batch_y_index = batch_y_index.to(self.device)
                # 将当前批次的数据移动到模型所在的设备上，确保数据和模型在同一个设备上进行计算。
                pre, _ = self(batch_x_embedding)# 对当前批次的输入数据进行前向传播，获取模型的预测结果。
                """
                既然：pre, _ = self(batch_x_embedding)就是执行forward，那我可不可以写成：pre, _ = self.forward(batch_x_embedding)
                是的，您完全可以直接调用self.forward(batch_x_embedding)来执行模型的前向传播过程。在PyTorch中，当您调用模型实例并传递输入数据时，
                比如self(batch_x_embedding)，它实际上是在内部自动调用self.forward(batch_x_embedding)方法。这种调用方式是等效的，但直接使用self(batch_x_embedding)是更常见和推荐的做法，因为它遵循了PyTorch框架设计的意图和习惯。
                为什么推荐使用self(input)而不是直接调用self.forward(input)？
                抽象封装：使用self(input)调用方式，更符合面向对象编程中的封装原则。它隐藏了内部实现的细节（即不需要直接调用forward方法），让对象的使用者可以更专注于使用对象的功能，而不是其实现细节。
                框架兼容性：PyTorch的模块化设计让self(input)调用方式在执行forward方法之前和之后，可以自动进行一些框架层面的处理，比如钩子函数的调用。直接调用forward可能会绕过这些处理，虽然在大多数情况下可能没有显著影响，但在某些特定情况下可能会导致不预期的行为。
                代码一致性：在PyTorch社区中，使用self(input)的方式更加普遍，遵循这种做法可以增加代码的可读性和一致性，特别是在与其他PyTorch开发者协作时。
                总的来说，虽然直接调用self.forward(input)在技术上是可行的，但遵循PyTorch的约定使用self(input)更加推荐。这样做不仅能保持代码的简洁和一致性，还能充分利用PyTorch框架提供的功能和优化。
                """
                loss = self.cross_entropy(pre, batch_y_index.reshape(-1))
                loss.backward()  # 梯度反传 , 梯度累加, 但梯度并不更新, 梯度是由优化器更新的
                optimizer.step()  # 使用优化器更新梯度
                optimizer.zero_grad()  # 梯度清零

                if batch_index % 100 == 0:# 每处理100个批次，打印一次当前的损失值。
                    print(f"loss:{loss:.3f}")
                    self.generate_poetry_auto()# 打印当前模型生成的一首自动诗歌，作为训练过程中的一个样本输出。
        pickle.dump(self, open(model_result_file, "wb"))# 训练完成后，将训练好的模型保存到文件中，以便将来可以重新加载和使用。
        return self# 返回训练完成的模型实例。

    def generate_poetry_auto(self):#该函数用于，输入第一个字，后面出来一首诗
        # self.eval()
        result = ""# 初始化一个空字符串，用于存储生成的诗句。
        word_index = np.random.randint(0, self.word_size, 1)[0]# 随机选择一个词的索引。self.word_size表示词汇表的大小，np.random.randint用于生成一个位于[0, self.word_size)区间的随机整数。

        #上面的函数用于随便产生一个汉字的index
        result += self.index_2_word[word_index]# 将随机选择的第一个字（通过索引转换成字）添加到结果字符串中。
        h_0 = torch.tensor(np.zeros((2, 1, self.hidden_num), dtype=np.float32))
        c_0 = torch.tensor(np.zeros((2, 1, self.hidden_num), dtype=np.float32))
        # 初始化LSTM的隐藏状态h_0和细胞状态c_0。这两个状态的形状为[2, 1, self.hidden_num]，
        # 其中2可能代表LSTM层的数量或者是双向LSTM的两个方向，1代表批次大小（这里每次处理一个字），self.hidden_num是隐藏层的维度。
        for i in range(31):# 对于诗的每个字（假设总共有32个字，已经生成了第一个字），重复以下步骤生成剩余的字。
            word_embedding = torch.tensor(self.w1[word_index][None][None])  # 根据当前字的索引，获取该字的词嵌入。self.w1是一个词嵌入矩阵。[None][None]的操作是为了增加两个维度，使其形状与模型输入匹配。
            pre, (h_0, c_0) = self(word_embedding, h_0, c_0) # 使用当前字的词嵌入和上一个状态的隐藏状态和细胞状态作为输入，通过模型预测下一个字的概率分布。同时更新h_0和c_0为新的状态。
            #上面这个函数实际上也是self.forward()
            word_index = int(torch.argmax(pre))# 从预测的概率分布中选取概率最高的字的索引。
            result += self.index_2_word[word_index]# 将预测的字（通过索引转换成字）添加到结果字符串中。
        print(result)

    def generate_poetry_based_on_start_word(self, start_word):
        # 定义一个方法，根据输入的起始汉字自动生成一首诗。
        result = start_word
        # 将输入的起始字作为结果字符串的开始。
        if start_word in self.word_2_index:
            # 检查起始字是否在词汇表中。
            word_index = self.word_2_index[start_word]
            # 获取起始字的索引。
        else:
            print("起始字不在词汇表中，请尝试其他字。")
            return
            # 如果起始字不在词汇表中，打印提示信息并结束函数。
        h_0 = torch.tensor(np.zeros((2, 1, self.hidden_num), dtype=np.float32))
        c_0 = torch.tensor(np.zeros((2, 1, self.hidden_num), dtype=np.float32))
        # 初始化LSTM的隐藏状态和细胞状态，与之前相同。
        for i in range(31):
            # 假设一首诗有32个字，已经有了第一个字，接下来生成剩余的字。
            word_embedding = torch.tensor(self.w1[word_index][None][None])
            # 获取当前字的词嵌入。
            pre, (h_0, c_0) = self(word_embedding, h_0, c_0)
            # 使用当前字的词嵌入和上一个状态的隐藏状态和细胞状态作为输入，通过模型预测下一个字的概率分布，并更新状态。
            word_index = int(torch.argmax(pre))
            # 从预测的概率分布中选取概率最高的字的索引。
            result += self.index_2_word[word_index]
            # 将预测的字添加到结果字符串中。
        print(result)
        # 打印生成的诗句。

    def generate_poetry_acrostic(self,input_text):
        # 定义一个方法，用于生成藏头诗。
        while True:
            #print("请输入四个汉字：", end="")
            #input_text = sys.stdin.readline().strip()[:4]
            #input_text = input("请输入四个汉字：")[:4]# 从用户那里接收输入，限制输入的汉字数量为四个。
            if input_text == "":
                self.generate_poetry_auto() # 如果用户没有输入任何内容，则自动生成一首诗。
            else:
                result = ""# 初始化一个空字符串，用于存储生成的诗句。
                punctuation_list = ["，", "。", "，", "。"] # 定义一个标点列表，用于在诗句中插入标点，形成标准的古诗格式。
                for i in range(4):
                    # 对于用户输入的每个汉字（总共四个），执行以下操作生成藏头诗的每一行。
                    h_0 = torch.tensor(np.zeros((2, 1, self.hidden_num), dtype=np.float32))
                    c_0 = torch.tensor(np.zeros((2, 1, self.hidden_num), dtype=np.float32))
                    # 初始化LSTM的隐藏状态和细胞状态，准备生成新的一行诗。
                    word = input_text[i] # 获取当前的藏头诗的头一个字。
                    try:# 尝试获取输入字的索引。如果输入的字不在词汇表中，随机选择一个字作为替代。
                        word_index = self.word_2_index[word]
                    except:
                        word_index = np.random.randint(0, self.word_size, 1)[0]
                        word = self.index_2_word[word_index]
                    result += word# 将当前的藏头字添加到结果字符串中。

                    for j in range(6):# 为当前行生成剩余的字，假设每行总共有7个字。
                        word_index = self.word_2_index[word]# 获取当前字的索引。
                        word_embedding = torch.tensor(self.w1[word_index][None][None]) # 根据当前字的索引，获取其词嵌入。
                        pre, (h_0, c_0) = self(word_embedding, h_0, c_0)# 使用当前字的词嵌入和上一个状态的隐藏状态和细胞状态作为输入，通过模型预测下一个字的概率分布。同时更新h_0和c_0为新的状态。
                        word = self.index_2_word[int(torch.argmax(pre))] # 从预测的概率分布中选取概率最高的字。
                        result += word# 将预测的字添加到结果字符串中。

                    result += punctuation_list[i] # 在每行的末尾添加适当的标点符号。
                print(result) # 打印生成的藏头诗。
    def generate_poetry_acrostic2(self,input_text):
        # 定义一个方法，用于生成藏头诗。

            #print("请输入四个汉字：", end="")
            #input_text = sys.stdin.readline().strip()[:4]
            #input_text = input("请输入四个汉字：")[:4]# 从用户那里接收输入，限制输入的汉字数量为四个。
        if input_text == "":
            self.generate_poetry_auto() # 如果用户没有输入任何内容，则自动生成一首诗。
        else:
            result = ""# 初始化一个空字符串，用于存储生成的诗句。
            punctuation_list = ["，", "。", "，", "。"] # 定义一个标点列表，用于在诗句中插入标点，形成标准的古诗格式。
            for i in range(4):
                # 对于用户输入的每个汉字（总共四个），执行以下操作生成藏头诗的每一行。
                h_0 = torch.tensor(np.zeros((2, 1, self.hidden_num), dtype=np.float32))
                c_0 = torch.tensor(np.zeros((2, 1, self.hidden_num), dtype=np.float32))
                # 初始化LSTM的隐藏状态和细胞状态，准备生成新的一行诗。
                word = input_text[i] # 获取当前的藏头诗的头一个字。
                try:# 尝试获取输入字的索引。如果输入的字不在词汇表中，随机选择一个字作为替代。
                    word_index = self.word_2_index[word]
                except:
                    word_index = np.random.randint(0, self.word_size, 1)[0]
                    word = self.index_2_word[word_index]
                result += word# 将当前的藏头字添加到结果字符串中。

                for j in range(6):# 为当前行生成剩余的字，假设每行总共有7个字。
                    word_index = self.word_2_index[word]# 获取当前字的索引。
                    word_embedding = torch.tensor(self.w1[word_index][None][None]) # 根据当前字的索引，获取其词嵌入。
                    pre, (h_0, c_0) = self(word_embedding, h_0, c_0)# 使用当前字的词嵌入和上一个状态的隐藏状态和细胞状态作为输入，通过模型预测下一个字的概率分布。同时更新h_0和c_0为新的状态。
                    word = self.index_2_word[int(torch.argmax(pre))] # 从预测的概率分布中选取概率最高的字。
                    result += word# 将预测的字添加到结果字符串中。

                result += punctuation_list[i] # 在每行的末尾添加适当的标点符号。
            print(result) # 打印生成的藏头诗。