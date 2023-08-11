import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, Softmax
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt

from tokenizer import Tokenizer
from model import SentimentModel
from dataset import SentimentDataset
from constants import VALIDATION_FNAME, BATCH_SIZE

def compute_accuracy_n_loss (model,fname=VALIDATION_FNAME):
    eval_ds = SentimentDataset(data_file=fname)
    eval_dl = DataLoader(eval_ds, batch_size=128)
    loss_fn = CrossEntropyLoss()
    losses = []
    accs = []
    classifier = Softmax(dim=-1)
    for x_batch, y_batch in tqdm(eval_dl):
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        y_pred = torch.argmax(classifier(logits), dim=-1)
        losses.append(loss.item())
        accs.append((y_pred == y_batch).float().sum().item())
    return np.mean(losses), np.sum(accs) / len(eval_ds)

def get_confusion_matrix (model,fname=VALIDATION_FNAME):
    eval_ds = SentimentDataset(data_file=fname)
    eval_dl = DataLoader(eval_ds, batch_size=128)
    loss_fn = CrossEntropyLoss()
    classifier = Softmax(dim=-1)
    targets = []
    preds = []
    for x_batch, y_batch in tqdm(eval_dl):
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        y_pred = torch.argmax(classifier(logits), dim=-1)
        targets.append(y_batch.numpy())
        preds.append(y_pred.numpy())
    targets, preds = np.concatenate(targets), np.concatenate(preds)
    print(targets.shape)
    return metrics.confusion_matrix(targets, preds)

if __name__ == "__main__":
    model = SentimentModel.load()
    loss, acc = compute_accuracy_n_loss(model)
    confusion_matrix = get_confusion_matrix(model)
    s = f"""\
accuracy (eval) : {100*acc} %
loss     (eval) : {loss}
    """
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()
    print(s)
    plt.show()
