import os
import sys
import pickle
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

from transformers import BertTokenizer, BertModel
from DBLP_src.run import load_bert

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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1", "T"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0", "F"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='GraphMeanSage', choices=['GraphMeanSage', 'Heterformer'])
    parser.add_argument('--dataset', type=str, default='DBLP', choices=['DBLP', 'tweet', 'book'])
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased')
    parser.add_argument('--k', type=int, default=10)

    # unimportant args
    parser.add_argument("--pretrain_embed", type=str2bool, default=False) # use pretrain author/venue embedding or not
    parser.add_argument("--pretrain_dir", default="data/book/pretrain_embed", type=str, choices=['data/book/pretrain_embed', 'data/dblp/pretrain_embed']) # pretrain textless node embedding dir
    parser.add_argument("--heter_embed_size", type=int, default=64)
    parser.add_argument("--paper_neighbour", type=int, default=5)
    parser.add_argument("--author_neighbour", type=int, default=3)

    return parser.parse_args()

@ torch.no_grad()
def main(args, input_query, author_idx=None):

    # read file
    train_set_embedding = np.load(os.path.join(args.dataset+'_embed', args.method+'_transductive.npy'))
    test_set_embedding = np.load(os.path.join(args.dataset+'_embed', args.method+'_inductive.npy'))
    document_embedding = np.concatenate((train_set_embedding, test_set_embedding), axis=0)
    print('Embedding Reading Over!')
    with open(os.path.join('data/'+args.dataset+'_embed', 'train_test_text.tsv')) as f:
        data = f.readlines()
    documents = [d.strip() for d in data]
    assert len(documents) == document_embedding.shape[0]
    print('File Reading Over!')

    # encoding input query
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # initialize model
    if args.method in ['GraphMeanSage']:
        load_ckpt_name = f'ckpt/dblp/{args.method}-p-True-1e-05-64-768-cnt-best.pt'
    elif args.method in ['Heterformer']:
        load_ckpt_name = f'ckpt/dblp/{args.method}-pp-True-1e-05-64-768-cnt-best.pt'
        args.author_num, args.venue_num = pickle.load(open('data/dblp/author_venue_num.pkl','rb'))
    else:
        raise ValueError('Incorrect Model Name!')
    args.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_bert(args)
    logging.info('loading model: {}'.format(args.method))
    encoder.to(args.device)
    encoder.load_state_dict(torch.load(load_ckpt_name, map_location="cpu"))
    encoder.eval()
    logging.info('load ckpt:{}'.format(load_ckpt_name))
    print('Model Loading Over!')

    # tokenizing
    # query_embedding = encoder(torch.LongTensor(tokenized_query['input_ids']), torch.LongTensor(tokenized_query['attention_mask']))[1].numpy()
    if args.method in ['GraphMeanSage']:
        tokenized_query = tokenizer.batch_encode_plus([input_query, '', '', '', '', ''], max_length=32, padding='max_length', truncation=True)
        mask_query_and_neighbors = torch.LongTensor([[1,0,0,0,0,0]])
        batch_input = [torch.LongTensor(tokenized_query['input_ids']).unsqueeze(0), 
                        torch.LongTensor(tokenized_query['attention_mask']).unsqueeze(0),
                        mask_query_and_neighbors]
        batch_input = [b.to(args.device) for b in batch_input]
        query_embedding = encoder.infer(*batch_input).cpu().numpy()
    elif args.method in ['Heterformer']:
        tokenized_query = tokenizer.batch_encode_plus([input_query, '', '', '', '', ''], max_length=32, padding='max_length', truncation=True)
        mask_query_and_neighbors = torch.LongTensor([[1,0,0,0,0,0,0,0,0,0]])
        if not author_idx:
            author_venue = torch.LongTensor([[-1,-1,-1,-1]])
        else:
            author_venue = torch.LongTensor([[author_idx,-1,-1,-1]])
        batch_input = [torch.LongTensor(tokenized_query['input_ids']).unsqueeze(0), 
                        torch.LongTensor(tokenized_query['attention_mask']).unsqueeze(0),
                        author_venue,
                        mask_query_and_neighbors]
        batch_input = [b.to(args.device) for b in batch_input]
        query_embedding = encoder.infer(*batch_input).cpu().numpy()
    else:
        raise ValueError('Incorrect Model Name!')

    # calculate score
    score = np.squeeze(np.matmul(document_embedding, query_embedding.transpose()))
    top_id = np.argsort(-score)[:args.k].tolist()

    # print result
    print('***************** Retrieval Result *****************')
    result_texts = [documents[i] for i in top_id]
    for t in result_texts:
        print(t)

if __name__ == '__main__':

    # prepare
    args = parse_args()
    setuplogging()
    dblp_author_id2idx = pickle.load(open('data/dblp/DBLP_neighbour/random_train_author_id2idx.pkl','rb'))

    # print args
    logging.info(args)
    print(args)

    # main function
    input_query = 'xxx' # please put your query here
    input_author = None

    if input_author:
        author_idx = dblp_author_id2idx[input_author]
        main(args, input_query, author_idx)
    else:
        main(args, input_query)
