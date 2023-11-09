import os
import sys
import random
import logging
import argparse
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import sklearn
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score, rand_score
from sklearn.cluster import KMeans

from IPython import embed

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def setuplogging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='transductive', choices=['transductive', 'inductive'])
    parser.add_argument('--method', type=str, default='GraphMeanSage', choices=['GraphMeanSage', 'Heterformer'])
    parser.add_argument('--dataset', type=str, default='DBLP', choices=['DBLP', 'book'])
    parser.add_argument('--class_num', type=int, default=30) # 30 for DBLP, 10 for book
    parser.add_argument('--seed', type=int, default=1027)
    return parser.parse_args()


def main(args):
    # read file
    embedding = np.load(os.path.join(args.dataset+'_embed', args.method+'_'+args.mode+'.npy'))
    labels = np.load(os.path.join(args.dataset+'_embed', args.mode+'_label.npy'))
    assert embedding.shape[0] == labels.shape[0]

    # select single label nodes
    single_label_index = np.where(labels.sum(1) == 1)
    embedding = embedding[single_label_index]
    labels = labels[single_label_index]
    assert embedding.shape[0] == labels.shape[0]
    print(f'Number of Single Class Nodes:{embedding.shape[0]}')

    # sample for DBLP
    if args.dataset == 'DBLP':
        print('Sample 10 classes for DBLP')
        select_label_index = np.where(labels[:,:10].sum(1) == 1)
        embedding = embedding[select_label_index]
        labels = labels[select_label_index]
        assert embedding.shape[0] == labels.shape[0]

        # random sample
        random_label_index = np.arange(20000)
        np.random.shuffle(random_label_index)
        embedding = embedding[random_label_index]
        labels = labels[random_label_index]
        assert embedding.shape[0] == labels.shape[0]

        print(f'Number of Selected Class Nodes:{embedding.shape[0]}')

    # convert label one-hot to label id
    args.class_num = 10
    print(f'Class num:{args.class_num}')
    labels = np.argmax(labels, axis=-1)

    # clustering
    kmeans = KMeans(n_clusters=args.class_num, random_state=args.seed, verbose=1).fit(embedding)
    preds = kmeans.labels_
    
    # save preds
    np.save(os.path.join(args.dataset+'_clustering_result', args.method+'_'+args.mode+'.npy'), preds)

    # evaluation
    RI = rand_score(labels, preds)
    print(f'RI:{RI}')
    logging.info(f'RI:{RI}')
    
    ARI = adjusted_rand_score(labels, preds)
    print(f'ARI:{ARI}')
    logging.info(f'ARI:{ARI}')

    NMI = normalized_mutual_info_score(labels, preds)
    print(f'NMI:{NMI}')
    logging.info(f'NMI:{NMI}')

    V = v_measure_score(labels, preds)
    print(f'V:{V}')
    logging.info(f'V:{V}')

if __name__ == '__main__':

    # prepare
    args = parse_args()
    set_seed(args.seed)
    setuplogging()
    
    # print args
    logging.info(args)
    print(args)

    # main function
    main(args)
