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

class Scorer(torch.nn.Module):
    def __init__(self, args, embed_size1, embed_size2):
        super(Scorer, self).__init__()
        self.args = args
        self.W = nn.Parameter(torch.FloatTensor(embed_size1, embed_size2))
        nn.init.xavier_normal_(self.W)

    def forward(self, query_embeddings, key_embeddings):
        scores = torch.mm(torch.mm(query_embeddings, self.W), key_embeddings.transpose(0, 1))

        labels = torch.arange(start=0, end=scores.shape[0], dtype=torch.long, device=scores.device)
        loss = F.cross_entropy(scores, labels)

        return loss

    def test(self, query_embeddings, key_embeddings):
        scores = torch.mm(torch.mm(query_embeddings, self.W), key_embeddings.transpose(0, 1))

        labels = torch.arange(start=0, end=scores.shape[0], dtype=torch.long, device=scores.device)

        predictions = torch.argmax(scores, dim=-1)
        acc = (torch.sum((predictions == labels)) / labels.shape[0]).item()

        scores = scores.cpu().numpy()
        labels = F.one_hot(labels).cpu().numpy()
        auc_all = [roc_auc_score(labels[i], scores[i]) for i in range(labels.shape[0])]
        auc = np.mean(auc_all)
        mrr_all = [mrr_score(labels[i], scores[i]) for i in range(labels.shape[0])]
        mrr = np.mean(mrr_all)
        ndcg_all = [ndcg_score(labels[i], scores[i], labels.shape[1]) for i in range(labels.shape[0])]
        ndcg = np.mean(ndcg_all)

        return {
            "main": acc,
            "acc": acc,
            "auc": auc,
            "mrr": mrr,
            "ndcg": ndcg
        }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=None, choices=['book_author', 'book_shelves', 'paper_author', 'paper_venue', 'book_publisher'])
    parser.add_argument('--method', type=str, default='HeterGAT', choices=['GraphformerAbl', 'Heterformer', 'HeterformerD', 'HeterformerDT', 
                                                'HeterformerT', 'HeterGAT', 'HeterMAXSAGE', 'HeterMeanSAGE', 'HeterRGCN', 'HeterSHGN', 'HGT'])
    parser.add_argument('--dataset', type=str, default='book', choices=['DBLP', 'book'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1027)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    # parser.add_argument('--hidden_size', type=int, default=200)
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
    auc = [metrics_total['auc'] for metrics_total in metrics_total_list]
    mrr = [metrics_total['mrr'] for metrics_total in metrics_total_list]
    ndcg = [metrics_total['ndcg'] for metrics_total in metrics_total_list]

    res = {'main':main, 'acc':acc, 'auc':auc, 'mrr':mrr, 'ndcg':ndcg}

    logging.info('###################################################')
    for key in metrics_total_list[0]:
        logging.info("{}:{}±{}".format(key, np.mean(res[key]), np.std(res[key])))

@torch.no_grad()
def validate(args, model, dataloader, device):
    model.eval()

    count = 0
    metrics_total = defaultdict(float)

    for step, batch in enumerate(tqdm(dataloader)):
        # labels = np.concatenate((labels, batch[1].cpu().numpy()))
        batch = [b.to(device) for b in batch]

        metrics = model.test(*batch)
        for k, v in metrics.items():
            metrics_total[k] += v
        count += 1

    for key in metrics_total:
        metrics_total[key] /= count
        logging.info("{}:{}".format(key, metrics_total[key]))

    return metrics_total['main'], metrics_total


def main(args):
    # read file
    q_name, k_name = args.mode.split('_')
    
    q_embedding = np.load(os.path.join(args.dataset, args.mode, args.method+f'_{q_name}.npy'))
    k_embedding_all = np.load(os.path.join(args.dataset, args.mode, args.method+'_'+k_name+'.npy'))
    labels = np.load(os.path.join(args.dataset, args.mode, k_name+'.npy'))
    k_embedding = k_embedding_all[labels]

    assert q_embedding.shape[0] == k_embedding.shape[0]

    # split data
    indexes = np.arange(q_embedding.shape[0])
    np.random.shuffle(indexes)
    train_index = indexes[:int(len(indexes)*args.train_ratio)]
    val_index = indexes[int(len(indexes)*args.train_ratio):int(len(indexes)*(1-args.test_ratio))]
    test_index = indexes[int(len(indexes)*(1-args.test_ratio)):]

    # construct dataloader
    train_dataloader = tdata.DataLoader(tdata.TensorDataset(torch.FloatTensor(q_embedding[train_index]), torch.FloatTensor(k_embedding[train_index])),
        batch_size=args.batch_size, shuffle=True)
    val_dataloader = tdata.DataLoader(tdata.TensorDataset(torch.FloatTensor(q_embedding[val_index]), torch.FloatTensor(k_embedding[val_index])),
        batch_size=args.batch_size, shuffle=False)
    test_dataloader = tdata.DataLoader(tdata.TensorDataset(torch.FloatTensor(q_embedding[test_index]), torch.FloatTensor(k_embedding[test_index])),
        batch_size=args.batch_size, shuffle=False)

    # args.input_size = embedding.shape[1]

    # define model & optimizer
    model = Scorer(args, q_embedding.shape[1], k_embedding.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0 
    best_count = 0
    ckpt_path = os.path.join(args.dataset+'_ckpt', args.mode, '{}-{}-best.pt'.format(args.method, args.lr))

    # train
    for i in range(args.epochs):
        total_loss  = []
        model.train()
        for q_embed, k_embed in tqdm(train_dataloader):
            loss_train = model(q_embed.to(device), k_embed.to(device))
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
                logging.info("Start testing for best")
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
