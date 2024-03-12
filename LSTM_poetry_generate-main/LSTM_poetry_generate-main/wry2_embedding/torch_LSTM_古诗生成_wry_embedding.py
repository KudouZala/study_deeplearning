
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from gensim.models.word2vec import Word2Vec
from torch.utils.data import Dataset, DataLoader
from model_def_embedding import MyModel
from dataset_def_embedding import MyDataset




if __name__ == "__main__":
    params = {
        "batch_size": 10,
        "epochs": 100,
        "lr": 0.003,
        "hidden_num": 64,
        "embedding_num": 128,
        "train_num": 2000,
        "optimizer": torch.optim.AdamW,
        "batch_num_test": 100,
    }
    model = MyModel(params)
    model = model.to_train()
    model.generate_poetry_acrostic("爱陈品初")
