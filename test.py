import pandas as pd
import torch
from torch.utils.data import Dataset

from tokenizer import Tokenizer
from model import SentimentModel

from constants import DATASET_FNAME

class TestDataset (Dataset):
    def __init__ (self, data_file=DATASET_FNAME):
        self.data = []
        self.tokenizer = Tokenizer.load()

    def __len__ (self):
        return len(self.data)

    def __getitem__ (self, idx):
        x_train = self.tokenizer.to_tensor(self.data[idx])

        return x_train

    def append (self,x):
        self.data.append(x)

if __name__ == "__main__":
    test_dataset = TestDataset()
    model = SentimentModel.load()
    classifier = torch.nn.Softmax(dim=1)

    for i in range(1,1000):
        test_dataset.append(input(f"{i} : "))
        x,lens = test_dataset[-1]
        x_test = torch.unsqueeze(x,dim=0)
        
        y_pred = model((x_test,[lens]))
        p_dist = classifier(y_pred)
        print(p_dist)
        print(torch.argmax(p_dist))
