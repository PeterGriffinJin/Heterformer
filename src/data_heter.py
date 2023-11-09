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


def load_dataset_text(args, tokenizer, evaluate=False, test=False):
    '''
    features : (token_query_and_neighbors, attention_query_and_neighbors, mask_query_and_neighbors), (token_key_and_neighbors, attention_key_and_neighbors, mask_key_and_neighbors)
    '''
    # assert args.data_mode in ['text', 'textf2']

    # block for processes which are not the core process
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # load data features from cache or dataset file
    evaluation_set_name = 'test' if test else 'val'
    cached_features_file = os.path.join(args.data_path, 'cached_{}_{}_{}_{}_{}_{}_{}'.format(
        args.data_mode,
        evaluation_set_name if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_length),
        args.book_neighbour,
        args.shelves_neighbour,
        args.author_neighbour))
    
    # exist or not
    if os.path.exists(cached_features_file):
        if args.local_rank in [-1, 0]:
            logger.info(f"Loading features from cached file {cached_features_file}")
        features = pickle.load(open(cached_features_file,'rb'))
    else:
        if args.local_rank in [-1, 0]:
            logger.info("Creating features from dataset file at %s",
                    args.data_path)

        # read_file = (evaluation_set_name if evaluate else 'train') + '_filter.tsv'
        read_file = (evaluation_set_name if evaluate else 'train') + '.tsv'
        # read_file = (evaluation_set_name if evaluate else 'train') + f'_{args.data_mode}.tsv'

        features = read_process_data_heter(os.path.join(args.data_path, read_file), tokenizer, args.max_length, args.book_neighbour, args.shelves_neighbour, args.author_neighbour)
        logger.info(f"Saving features into cached file {cached_features_file}")
        pickle.dump(features, open(cached_features_file, 'wb'))

    if args.local_rank == 0:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # convert to Tensors and build dataset
    token_query_and_neighbors = torch.LongTensor(features[0][0])
    attention_query_and_neighbors = torch.LongTensor(features[0][1])
    query_mta_neighbors_list = torch.LongTensor(features[0][2])
    mask_query_and_neighbors = torch.LongTensor(features[0][3])
    
    token_key_and_neighbors = torch.LongTensor(features[1][0])
    attention_key_and_neighbors = torch.LongTensor(features[1][1])
    key_mta_neighbors_list = torch.LongTensor(features[1][2])
    mask_key_and_neighbors = torch.LongTensor(features[1][3])

    assert token_query_and_neighbors.shape[1] == 1 + args.book_neighbour
    assert query_mta_neighbors_list.shape[1] == args.shelves_neighbour + args.author_neighbour + 3
    assert token_key_and_neighbors.shape[1] == 1 + args.book_neighbour
    assert key_mta_neighbors_list.shape[1] == args.shelves_neighbour + args.author_neighbour + 3

    dataset = TensorDataset(token_query_and_neighbors, attention_query_and_neighbors, query_mta_neighbors_list, mask_query_and_neighbors,
                            token_key_and_neighbors, attention_key_and_neighbors, key_mta_neighbors_list, mask_key_and_neighbors)

    return dataset

def read_process_data_heter(dir, tokenizer, max_length, book_neighbour, shelves_neighbour, author_neighbour):
    '''
    Each line is a tweet/POI node pair. Each node is made up of [itself, tweet_neighbour * neighbour tweet, mention_neighbour * neighbour mention, tag_neighbour * neighbour tag, author].
    '''
    token_query_and_neighbors = []
    token_key_and_neighbors = []
    attention_query_and_neighbors = []
    attention_key_and_neighbors = []
    query_mta_neighbors_list = []
    key_mta_neighbors_list = []
    mask_query_and_neighbors = []
    mask_key_and_neighbors = []

    with open(dir) as f:
        data = f.readlines()
        for line in tqdm(data):
            a = line.strip().split('\$\$')
            if len(a) == 2:
                query_all, key_all = a
            else:
                print(a)
                raise ValueError('stop')
            query_and_neighbors = query_all.split('\t')
            key_and_neighbors = key_all.split('\t')
            tmp_mask_query_and_neighbors = []
            tmp_mask_key_and_neighbors = []

            # make sure that length is neighbour_num + 1 for query and key
            assert len(query_and_neighbors) == 1 + book_neighbour + shelves_neighbour + author_neighbour + 3
            assert len(key_and_neighbors) == 1 + book_neighbour + shelves_neighbour + author_neighbour + 3

            # split the neighbours
            query_and_text_neighbors = query_and_neighbors[:(1+book_neighbour)]
            query_mta_neighbors = query_and_neighbors[(1+book_neighbour):]
            query_mta_neighbors = [int(v) for v in query_mta_neighbors]

            key_and_text_neighbors = key_and_neighbors[:(1+book_neighbour)]
            key_mta_neighbors = key_and_neighbors[(1+book_neighbour):]
            key_mta_neighbors = [int(v) for v in key_mta_neighbors]

            query_mta_neighbors_list.append(query_mta_neighbors)
            key_mta_neighbors_list.append(key_mta_neighbors)

            # construct neighbour mask
            for v in query_and_text_neighbors:
                if len(v) != 0:
                    tmp_mask_query_and_neighbors.append(1)
                else:
                    tmp_mask_query_and_neighbors.append(0)
            for v in query_mta_neighbors:
                if v != -1:
                    tmp_mask_query_and_neighbors.append(1)
                else:
                    tmp_mask_query_and_neighbors.append(0)

            for v in key_and_text_neighbors:
                if len(v) != 0:
                    tmp_mask_key_and_neighbors.append(1)
                else:
                    tmp_mask_key_and_neighbors.append(0)
            for v in key_mta_neighbors:
                if v != -1:
                    tmp_mask_key_and_neighbors.append(1)
                else:
                    tmp_mask_key_and_neighbors.append(0)
            mask_query_and_neighbors.append(tmp_mask_query_and_neighbors)
            mask_key_and_neighbors.append(tmp_mask_key_and_neighbors)

            encoded_query_and_neighbors = tokenizer.batch_encode_plus(query_and_text_neighbors, max_length=max_length, padding='max_length', truncation=True)
            encoded_key_and_neighbors = tokenizer.batch_encode_plus(key_and_text_neighbors, max_length=max_length, padding='max_length', truncation=True)

            token_query_and_neighbors.append(encoded_query_and_neighbors['input_ids'])
            token_key_and_neighbors.append(encoded_key_and_neighbors['input_ids'])
            attention_query_and_neighbors.append(encoded_query_and_neighbors['attention_mask'])
            attention_key_and_neighbors.append(encoded_key_and_neighbors['attention_mask'])

    return (token_query_and_neighbors, attention_query_and_neighbors, query_mta_neighbors_list, mask_query_and_neighbors), \
        (token_key_and_neighbors, attention_key_and_neighbors, key_mta_neighbors_list, mask_key_and_neighbors)
