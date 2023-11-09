import logging
import os
import random
from time import time
from collections import defaultdict
import pickle

from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from utils import setuplogging
from data import load_dataset

from transformers import BertConfig, BertTokenizerFast

from IPython import embed

def load_bert(args):
    config = BertConfig.from_pretrained(args.model_name_or_path, output_hidden_states=True)
    if args.model_type == 'BERTMF':
        from model import BERTMF
        model = BERTMF(args, config)
    elif args.model_type == 'BERTMFM':
        from model import BERTMFM
        model = BERTMFM(args, config)
    else:
        raise ValueError('Input Model Name is Incorrect!')

    return model

def train(args):

    # define tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # load dataset
    ########################## Motify collate_f & think about shuffle in Dataloader
    train_set = load_dataset(args, tokenizer, evaluate=False, test=False)
    val_set = load_dataset(args, tokenizer, evaluate=True, test=False)

    train_sampler = RandomSampler(train_set)
    val_sampler = RandomSampler(val_set)

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, sampler=val_sampler)
    # test_loader = DataLoader(test_set, batch_size=args.test_batch_size, sampler=test_sampler)
    print('Dataset Loading Over!')

    # define model
    model = load_bert(args) ###################################### You may need to modify the length of author embedding (now it's 768, too long)
    logging.info('loading model: {}'.format(args.model_type))
    model = model.cuda()

    if args.load:
        model.load_state_dict(torch.load(args.load_ckpt_name, map_location="cpu"))
        logging.info('load ckpt:{}'.format(args.load_ckpt_name))

    # define optimizer
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': args.lr}])
    ################# You should motify the model name here if using DDP

    loss = 0.0
    global_step = 0
    best_acc, best_count = 0.0, 0
    for ep in range(args.epochs):
        ## start training
        start_time = time()
        model.train() ######################## You should motify it to ddp_model.train when using DDP
        for step, batch in enumerate(tqdm(train_loader)):
            # put data into GPU
            if args.enable_gpu:
                batch = [b.cuda() for b in batch]

            # calculate loss
            batch_loss = model(*batch)
            loss += batch_loss.item()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            global_step += 1

            # logging
            if global_step % args.log_steps == 0:
                logging.info(
                    'cost_time:{} step:{}, lr:{}, train_loss: {:.5f}'.format(
                        time() - start_time, global_step, optimizer.param_groups[0]['lr'],
                        loss / args.log_steps))
                loss = 0.0

        ## start validating
        ckpt_path = os.path.join(args.model_dir, '{}-{}-epoch-{}-{}.pt'.format(args.model_type, args.author_embed, args.lr, ep + 1))
        torch.save(model.state_dict(), ckpt_path)
        logging.info(f"Model saved to {ckpt_path}")

        logging.info("Start validation for epoch-{}".format(ep + 1))
        acc = validate(args, model, val_loader)
        logging.info("validation time:{}".format(time() - start_time))
        if acc > best_acc:
            ckpt_path = os.path.join(args.model_dir, '{}-{}-{}-best.pt'.format(args.model_type, args.author_embed, args.lr))
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"Model saved to {ckpt_path}")
            best_acc = acc
            best_count = 0
        else:
            best_count += 1
            if best_count >= args.early_stop:
                # start_time = time()
                # ckpt_path = os.path.join(args.model_dir, '{}-best.pt'.format(args.model_type))
                # model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                # logging.info("Star testing for best")
                # acc = validate(args, model, test_loader)
                # logging.info("test time:{}".format(time() - start_time))
                exit()

@torch.no_grad()
def validate(args, model, dataloader):
    model.eval()

    count = 0
    metrics_total = defaultdict(float)
    for step, batch in enumerate(dataloader):
        if args.enable_gpu:
                batch = [b.cuda() for b in batch]

        metrics = model.test(*batch)
        for k, v in metrics.items():
            metrics_total[k] += v
        count += 1

    for key in metrics_total:
        metrics_total[key] /= count
        logging.info("{}:{}".format(key, metrics_total[key]))

    return metrics_total['main']

def test(args):
    # define tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # load dataset
    ########################## Motify collate_f & think about shuffle in Dataloader
    args.author_num = len(pickle.load(open(args.author_file,'rb')))
    test_set = load_dataset(args, tokenizer, evaluate=True, test=True)
    # test_sampler = SequentialSampler(test_set) if args.local_rank == -1 else DistributedSampler(test_set)
    test_sampler = RandomSampler(test_set)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, sampler=test_sampler)
    print('Dataset Loading Over!')

    # define model
    model = load_bert(args)
    logging.info('loading model: {}'.format(args.model_type))
    model = model.cuda()

    # load checkpoint
    checkpoint = torch.load(args.load_ckpt_name, map_location="cpu")
    # model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.load_state_dict(checkpoint)
    logging.info('load ckpt:{}'.format(args.load_ckpt_name))

    # test
    validate(args, model, test_loader)
