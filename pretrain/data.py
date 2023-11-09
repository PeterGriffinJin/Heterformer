import os
import sys
import math
import pickle
import random
import logging
from tqdm import tqdm

import numpy as np
import torch

from torch.utils.data.dataset import Dataset, TensorDataset
from transformers import BertTokenizer, BertTokenizerFast

from IPython import embed

logger = logging.getLogger(__name__)


def load_dataset(args, tokenizer, evaluate=False, test=False):
    '''
    features : (token_paper, attention_paper), author
    '''
    # load data features from cache or dataset file
    # cached_features_file = os.path.join(args.data_path, 'cached_{}_{}'.format(
    #     list(filter(None, args.model_name_or_path.split('/'))).pop(),
    #     str(args.max_length)))

    evaluation_set_name = 'test' if test else 'val'
    cached_features_file = os.path.join(args.data_path, 'cached_{}_{}_{}_{}'.format(
        evaluation_set_name if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_length),
        args.pretrain_mode))
    
    # exist or not
    if os.path.exists(cached_features_file):
        logger.info(f"Loading features from cached file {cached_features_file}")
        features = pickle.load(open(cached_features_file,'rb'))
    else:
        logger.info("Creating features from dataset file at %s",
                    args.data_path)

        read_file = (evaluation_set_name if evaluate else 'train') + '_' + args.pretrain_mode + '.tsv'
        # read_file = 'train_a.tsv'

        features = read_process_data(os.path.join(args.data_path, read_file), tokenizer, args.max_length)
        logger.info(f"Saving features into cached file {cached_features_file}")
        pickle.dump(features, open(cached_features_file, 'wb'))

    # convert to Tensors and build dataset
    token_paper = torch.LongTensor(features[0][0])
    attention_paper = torch.LongTensor(features[0][1])
    author_list = torch.LongTensor(features[1])
    # assert max(author_list) < args.author_num

    dataset = TensorDataset(token_paper, attention_paper, author_list)

    return dataset

def read_process_data(dir, tokenizer, max_length):
    token_paper = []
    attention_paper = []
    author_list = []

    cnt = 0
    with open(dir) as f:
        data = f.readlines()
        for line in tqdm(data):
            a = line.strip().split('\t')
            if len(a) == 2:
                paper, author = a
            else:
                print(a)
                cnt += 1
                raise ValueError('stop')

            encoded_paper = tokenizer.encode_plus(paper, max_length=max_length, padding='max_length', truncation=True)

            token_paper.append(encoded_paper['input_ids'])
            attention_paper.append(encoded_paper['attention_mask'])
            author_list.append(int(author))

    print('Fail read:{cnt}')
    return (token_paper, attention_paper), author_list
