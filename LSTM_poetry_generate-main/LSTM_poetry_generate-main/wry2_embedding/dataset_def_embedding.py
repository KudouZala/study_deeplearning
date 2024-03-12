import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from gensim.models.word2vec import Word2Vec
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, word_2_index, all_data):
        self.word_2_index = word_2_index
        self.all_data = all_data

    def __getitem__(self, index):
        a_poetry = self.all_data[index]
        a_poetry_index = [self.word_2_index[i] for i in a_poetry]
        xs = a_poetry_index[:-1]
        ys = a_poetry_index[1:]
        a3=torch.tensor(xs, dtype=torch.long)
        a4=np.array(xs).astype(np.int64)
        return np.array(xs).astype(np.int64), np.array(ys).astype(np.int64)

    def __len__(self):
        return len(self.all_data)


