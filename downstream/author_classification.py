import os
import sys
from copy import deepcopy
import random
import logging
import argparse
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import sklearn
from sklearn.metrics import f1_score, roc_auc_score, auc

import torch
from torch import nn, optim
import torch.utils.data as tdata
import torch.nn.functional as F

from IPython import embed

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setuplogging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)

class MLP(torch.nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        self.layers = nn.Sequential(nn.Linear(args.input_size, args.hidden_size),
                                    nn.ReLU(), nn.Linear(args.hidden_size, args.class_num),
                                    nn.Sigmoid())
    def forward(self, x):
        return self.layers(x)

    def test(self, x, labels):
        scores = self.layers(x)

        predictions = torch.argmax(scores, dim=-1)
        prc = sum([labels[i][predictions[i]].item() for i in range(labels.shape[0])]) / labels.shape[0]

        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()

        mrr_all = [mrr_score(labels[i], scores[i]) for i in range(labels.shape[0])]
        mrr = np.mean(mrr_all)

        ndcg_all = [ndcg_score(labels[i], scores[i], labels.shape[1]) for i in range(labels.shape[0])]

        ndcg = np.mean(ndcg_all)

        preds = (scores>0.5).astype(int)
        acc = (preds == labels).sum().sum() / labels.shape[0] / labels.shape[1]

        # preds = (scores>0.5).astype(int)
        # marco_f1 = f1_score(labels, preds, average='macro')
        # micro_f1 = f1_score(labels, preds, average='micro')

        return {
            "main": prc,
            "prc": prc,
            "mrr": mrr,
            "ndcg": ndcg,
            "acc": acc,
            # "marco_f1": marco_f1,
            # "micro_f1": micro_f1
        }, (scores>0.5).astype(int)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='BERT', choices=['GraphformerAbl', 'Heterformer', 'HeterformerD', 'HGT', 'HeterGAT', 'HeterRGCN', 'HeterSHGN'])
    parser.add_argument('--dataset', type=str, default='DBLP', choices=['DBLP', 'book'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--class_num', type=int, default=30) # 30 for DBLP, 10 for Goodreads
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=1027)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--early_stop', type=int, default=10)
    return parser.parse_args()

def acc(y_true, y_hat):
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k) + 1e-10
    actual = dcg_score(y_true, y_score, k) 
    return actual / best

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true + 1e-10)

def calculate(metrics_total_list):
    main = [metrics_total['main'] for metrics_total in metrics_total_list]
    acc = [metrics_total['acc'] for metrics_total in metrics_total_list]
    mrr = [metrics_total['mrr'] for metrics_total in metrics_total_list]
    ndcg = [metrics_total['ndcg'] for metrics_total in metrics_total_list]
    prc = [metrics_total['prc'] for metrics_total in metrics_total_list]
    micro_f1 = [metrics_total['micro_f1'] for metrics_total in metrics_total_list]
    macro_f1 = [metrics_total['macro_f1'] for metrics_total in metrics_total_list]

    res = {'main':main, 'acc':acc, 'mrr':mrr, 'ndcg':ndcg, 'prc':prc, 'micro_f1':micro_f1, 'macro_f1':macro_f1}

    logging.info('###################################################')
    for key in metrics_total_list[0]:
        logging.info("{}:{}±{}".format(key, np.mean(res[key]), np.std(res[key])))

@torch.no_grad()
def validate(args, model, dataloader, device):
    model.eval()

    count = 0
    metrics_total = defaultdict(float)
    labels = np.zeros(args.class_num)[None,:]
    preds = np.zeros(args.class_num)[None,:]

    for step, batch in enumerate(tqdm(dataloader)):
        labels = np.concatenate((labels, batch[1].cpu().numpy()))
        batch = [b.to(device) for b in batch]

        metrics, pred = model.test(*batch)
        for k, v in metrics.items():
            metrics_total[k] += v
        count += 1
        preds = np.concatenate((preds, pred), axis=0)

    for key in metrics_total:
        metrics_total[key] /= count
        logging.info("{}:{}".format(key, metrics_total[key]))

    labels = labels[1:,:]
    preds = preds[1:,:]
    metrics_total['micro_f1'] = f1_score(labels, preds, average='micro')
    metrics_total['macro_f1'] = f1_score(labels, preds, average='macro')
    logging.info("{}:{}".format('micro_f1', metrics_total['micro_f1']))
    logging.info("{}:{}".format('macro_f1', metrics_total['macro_f1']))

    return metrics_total['main'], metrics_total


def main(args):
    # read file
    embedding = np.load(os.path.join(args.dataset+'_embed', args.dataset+'_embed', args.method+'_author.npy'))
    labels = np.load(os.path.join(args.dataset+'_embed', args.dataset+'_embed', 'author_label.npy'))
    assert embedding.shape[0] == labels.shape[0]
    args.class_num = labels.shape[1]

    # delete no label nodes
    not_zero_index = np.where(labels.sum(1) != 0)
    print(f'Zero Class Nodes:{embedding.shape[0]-len(not_zero_index[0])}')
    embedding = embedding[not_zero_index]
    labels = labels[not_zero_index]
    assert embedding.shape[0] == labels.shape[0]
    print(f'All data:{embedding.shape[0]} samples.')

    # split data
    indexes = np.arange(embedding.shape[0])
    np.random.shuffle(indexes)
    # print(indexes[:5])
    train_index = indexes[:int(len(indexes)*args.train_ratio)]
    val_index = indexes[int(len(indexes)*args.train_ratio):int(len(indexes)*(1-args.test_ratio))]
    test_index = indexes[int(len(indexes)*(1-args.test_ratio)):]

    print(f'Train set size: {len(train_index)}, Validation set size: {len(val_index)}, Test set size:{len(test_index)}')

    # construct dataloader
    train_dataloader = tdata.DataLoader(tdata.TensorDataset(torch.FloatTensor(embedding[train_index]), torch.FloatTensor(labels[train_index])),
        batch_size=args.batch_size, shuffle=True)
    val_dataloader = tdata.DataLoader(tdata.TensorDataset(torch.FloatTensor(embedding[val_index]), torch.FloatTensor(labels[val_index])),
        batch_size=args.batch_size, shuffle=False)
    test_dataloader = tdata.DataLoader(tdata.TensorDataset(torch.FloatTensor(embedding[test_index]), torch.FloatTensor(labels[test_index])),
        batch_size=args.batch_size, shuffle=False)

    args.input_size = embedding.shape[1]

    # define model & optimizer
    model = MLP(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.BCELoss()
    best_acc = 0 
    best_count = 0
    ckpt_path = os.path.join('rebuttal', args.dataset+'_embed', 'author-{}-{}-{}-best.pt'.format(args.method, args.lr, args.hidden_size))

    # train
    for i in range(args.epochs):
        total_loss  = []
        model.train()
        for embedding, label in tqdm(train_dataloader):
            logit = model(embedding.to(device))
            loss_train = loss_func(logit, label.to(device))
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            total_loss.append(loss_train.item())
        print("Epoch:{} Train Loss:{}".format(i, np.mean(total_loss)))

        ## start validating
        logging.info("Start validation for epoch-{}".format(i + 1))
        acc, metrics_total = validate(args, model, val_dataloader, device)
        if acc > best_acc:
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"Model saved to {ckpt_path}")
            best_acc = acc
            best_metrics_total = deepcopy(metrics_total)
            best_count = 0
        else:
            best_count += 1
            if best_count >= args.early_stop:
                model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                logging.info("Star testing for best")
                acc, test_metrics_total = validate(args, model, test_dataloader, device)
                return best_metrics_total, test_metrics_total

    # test
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    logging.info('load ckpt:{}'.format(ckpt_path))
    acc, test_metrics_total = validate(args, model, test_dataloader, device)

    return best_metrics_total, test_metrics_total

if __name__ == '__main__':

    # prepare
    args = parse_args()
    # set_seed(args.seed)
    setuplogging()
    
    # print args
    logging.info(args)
    print(args)

    # main function
    # run for 5 times
    val_metrics_total_list = []
    test_metrics_total_list = []

    for i in range(4, -1, -1):
        set_seed(args.seed+i)
        val_metrics_total, test_metrics_total = main(args)
        val_metrics_total_list.append(val_metrics_total)
        test_metrics_total_list.append(test_metrics_total)

    calculate(val_metrics_total_list)
    calculate(test_metrics_total_list)

    # val_metrics_total, test_metrics_total = main(args)
