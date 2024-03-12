import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from gensim.models.word2vec import Word2Vec
from torch.utils.data import Dataset, DataLoader
from dataset_def_embedding import MyDataset

def split_text(file="poetry_7.txt", train_num=6000):  # 处理txt文件，
    all_data = open(file, "r", encoding="utf-8").read()
    with open("split_7.txt", "w", encoding="utf-8") as f:
        split_data = " ".join(all_data)
        f.write(split_data)
    return split_data[:train_num * 64]
def train_vec(vector_size=128, split_file="split_7.txt", org_file="poetry_7.txt", train_num=6000):

    param_file = "My_word_vec_wry_embedding3.pkl"
    org_data = open(org_file, "r", encoding="utf-8").read().split("\n")[:train_num]
    all_data_split = split_text().split("\n")[:train_num]

    if os.path.exists(param_file):
        return org_data, pickle.load(open(param_file, "rb"))

    models = Word2Vec(all_data_split, vector_size=vector_size, workers=7, min_count=1)

    pickle.dump([models.syn1neg, models.wv.key_to_index, models.wv.index_to_key], open(param_file, "wb"))

    return org_data, (models.syn1neg, models.wv.key_to_index, models.wv.index_to_key)


class MyModel(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.all_data, (self.w1, self.word_2_index, self.index_2_word) = train_vec(vector_size=params["embedding_num"],
                                                                                   train_num=params["train_num"])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hidden_num = params["hidden_num"]
        self.batch_size = params["batch_size"]

        self.epochs = params["epochs"]
        self.lr = params["lr"]
        self.optimizer = params["optimizer"]
        self.word_size, self.embedding_num = self.w1.shape

        self.embedding = nn.Embedding(num_embeddings=self.word_size, embedding_dim=self.embedding_num)


        self.lstm = nn.LSTM(input_size=self.embedding_num, hidden_size=self.hidden_num, batch_first=True, num_layers=2,
                            bidirectional=False)

        self.dropout = nn.Dropout(0.3)

        self.flatten = nn.Flatten(0, 1)

        self.linear = nn.Linear(self.hidden_num, self.word_size)

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, input_ids, h_0=None, c_0=None):
        if h_0 == None or c_0 == None:
            h_0 = torch.tensor(np.zeros((2, input_ids.shape[0], self.hidden_num), dtype=np.float32))
            c_0 = torch.tensor(np.zeros((2, input_ids.shape[0], self.hidden_num), dtype=np.float32))
        h_0 = h_0.to(self.device)
        c_0 = c_0.to(self.device)
        input_ids = input_ids.to(self.device)
        input_ids = input_ids.long()

        #word_embedding = self.embedding(input_ids).unsqueeze(0)  # 添加batch维度
        #print("Word embedding shape before LSTM:", input_ids.shape)


        xs_embedding = self.embedding(input_ids)
        hidden, (h_0, c_0) = self.lstm(xs_embedding, (h_0, c_0))
        hidden_drop = self.dropout(hidden)
        hidden_flatten = self.flatten(hidden_drop)
        pre = self.linear(hidden_flatten)
        a3=3

        return pre, (h_0, c_0)

    def to_train(self):
        model_result_file = "My_Model_lstm_model_embedding3.pkl"
        if os.path.exists(model_result_file):
            return pickle.load(open(model_result_file, "rb"))
        dataset = MyDataset(self.word_2_index, self.all_data)
        dataloader = DataLoader(dataset, self.batch_size)

        optimizer = self.optimizer(self.parameters(), self.lr)
        self = self.to(self.device)

        for e in range(self.epochs):
            for batch_index, (batch_x_embedding, batch_y_index) in enumerate(dataloader):

                #print(112)
                a5=len(dataloader)
                #print("batch_x_embedding:", batch_x_embedding.shape)
                #print("batch_y_index:", batch_y_index.shape)
                try:
                    batch_x_embedding = batch_x_embedding.to(self.device)
                    batch_y_index = batch_y_index.to(self.device)
                    # 尝试执行模型前向传播

                    pre, _ = self.forward(batch_x_embedding)
                    a5=5
                    # 确保损失计算在 try 块内部
                except Exception as e:
                    # 打印异常信息
                    #print(f"An error occurred: {e}")
                    continue  # Skip this iteration
                loss = self.cross_entropy(pre, batch_y_index.reshape(-1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if batch_index % 100 == 0:
                    print(f"loss:{loss:.3f}")
                    try:
                        #print(134)
                        self.generate_poetry_auto()
                    except Exception as e:
                        # 打印异常信息
                        #print(f"An error occurred: {e}")
                        continue  # Skip this iteration
        pickle.dump(self, open(model_result_file, "wb"))
        return self


    def generate_poetry_auto(self):
        result = ""
        word_index = np.random.randint(0, self.word_size, 1)[0]
        h_0 = torch.zeros((2, 1, self.hidden_num), dtype=torch.float32)
        c_0 = torch.zeros((2, 1, self.hidden_num), dtype=torch.float32)
        a6=3
        for i in range(31):
            try:
                word_tensor = torch.tensor([[word_index]], dtype=torch.long)  # 这里要改，这将创建一个形状为[1, 1,]的张量
                pre, (h_0, c_0) = self.forward(word_tensor, h_0, c_0)
            except Exception as e:
            # 打印异常信息
                print(f"An error occurred: {e}")
                continue  # Skip this iteration
            word_index = int(torch.argmax(pre, dim=-1))#这里要改，这里注意dim=-1
            result += self.index_2_word[word_index]
        print(result)

    def generate_poetry_based_on_start_word(self, start_word):
        result = start_word
        if start_word in self.word_2_index:
            word_index = self.word_2_index[start_word]
        else:
            print("起始字不在词汇表中，请尝试其他字。")
            return
        h_0 = torch.tensor(np.zeros((2, 1, self.hidden_num), dtype=np.float32))
        c_0 = torch.tensor(np.zeros((2, 1, self.hidden_num), dtype=np.float32))
        for i in range(31):
            word_embedding = torch.tensor(self.w1[word_index][None][None])
            pre, (h_0, c_0) = self(word_embedding, h_0, c_0)
            word_index = int(torch.argmax(pre))
            result += self.index_2_word[word_index]
        print(result)



    def generate_poetry_acrostic(self,input_text):
        if input_text == "":
            self.generate_poetry_auto()
        else:
            result = ""
            punctuation_list = ["，", "。", "，", "。"]
            for i in range(4):
                h_0 = torch.tensor(np.zeros((2, 1, self.hidden_num), dtype=np.float32))
                c_0 = torch.tensor(np.zeros((2, 1, self.hidden_num), dtype=np.float32))
                word = input_text[i]
                try:
                    word_index = self.word_2_index[word]
                except:
                    word_index = np.random.randint(0, self.word_size, 1)[0]
                    word = self.index_2_word[word_index]
                result += word

                for j in range(6):
                    word_index = self.word_2_index[word]
                    word_tensor = torch.tensor([[word_index]], dtype=torch.long)  # 这里要改，这将创建一个形状为[1, 1,]的张量
                    pre, (h_0, c_0) = self.forward(word_tensor, h_0, c_0)
                    word_index = int(torch.argmax(pre))
                    word= self.index_2_word[word_index]
                    result += word

                result += punctuation_list[i]
            print(result)
