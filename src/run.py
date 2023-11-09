import logging
import os
import pickle
import random
from time import time
from collections import defaultdict
# from torch._C import device

from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from src.utils import setuplogging
from src.data_heter import load_dataset_text

from transformers import BertConfig, BertTokenizerFast, AdamW, get_linear_schedule_with_warmup

from transformers import BertModel

from IPython import embed

def cleanup():
    dist.destroy_process_group()

def load_bert(args):
    config = BertConfig.from_pretrained(args.model_name_or_path, output_hidden_states=True)
    if args.model_type == 'GraphMeanSage':
        from src.model.Graphsage import GraphMeanSageForNeighborPredict
        model = GraphMeanSageForNeighborPredict.from_pretrained(args.model_name_or_path, config=config) if args.pretrain_LM else GraphMeanSageForNeighborPredict(config)
    elif args.model_type == 'Heterformer':
        from src.model.HeterformerD import HeterFormerDsForNeighborPredict
        model = HeterFormerDsForNeighborPredict.from_pretrained(args.model_name_or_path, config=config) if args.pretrain_LM else HeterFormerDsForNeighborPredict(config)
        model.shelves_num, model.author_num, model.publisher_num, model.language_code_num, model.format_num, model.heter_embed_size = args.shelves_num, args.author_num, args.publisher_num, args.language_code_num, args.format_num, args.heter_embed_size
        model.book_neighbour, model.shelves_neighbour, model.author_neighbour = args.book_neighbour, args.shelves_neighbour, args.author_neighbour
        model.init_mta_embed(args.pretrain_embed, args.pretrain_dir)
    else:
        raise ValueError('Input Model Name is Incorrect!')

    return model

def train(args):

    # # add special token
    #################### Make sure if you want to add special token here. If you want, remember to make further revise on graphformers.#########
    # additional_special_tokens = ["[paper]", "[author]", "[venue]"]
    # args.additional_special_tokens = additional_special_tokens
    print('[Warning] No special token is added in run.py line 56! Take care! If you want to add, remember to make further revision on graphformer!')

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # config.n_gpu = torch.cuda.device_count()
        args.n_gpu = 1
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    logging.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))

    # Load data
    # define tokenizer 
    # ################################################ Make sure if you want to add special token here. If you want, remember to make further revise on graphformers.#########
    # tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", additional_special_tokens=additional_special_tokens)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    # load dataset
    ########################## Motify collate_f & think about shuffle in Dataloader
    if args.data_mode in ['text']:
        train_set = load_dataset_text(args, tokenizer, evaluate=False, test=False)
        val_set = load_dataset_text(args, tokenizer, evaluate=True, test=False)
        test_set = load_dataset_text(args, tokenizer, evaluate=True, test=True)
        args.shelves_num, args.author_num, args.publisher_num, args.language_code_num, args.format_num = pickle.load(open(os.path.join(args.data_path, 'mta_num.pkl'),'rb'))
    else:
        raise ValueError('Data Mode is Incorrect here!')
    # define dataloader
    train_sampler = RandomSampler(train_set) if args.local_rank == -1 else DistributedSampler(train_set)
    val_sampler = SequentialSampler(val_set) if args.local_rank == -1 else DistributedSampler(val_set)
    test_sampler = SequentialSampler(test_set) if args.local_rank == -1 else DistributedSampler(test_set)

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, sampler=val_sampler)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, sampler=test_sampler)
    print(f'[Process:{args.local_rank}] Dataset Loading Over!')

    # define model
    model = load_bert(args)
    if args.local_rank in [-1, 0]:
        logging.info('loading model: {}'.format(args.model_type))
    model.to(args.device)

    if args.load:
        model.load_state_dict(torch.load(args.load_ckpt_name, map_location="cpu"))
        logging.info('load ckpt:{}'.format(args.load_ckpt_name))

    # define DDP here
    if args.local_rank != -1:
        ddp_model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    else:
        ddp_model = model

    # define optimizer
    ###################### You should think more here about the weight_decay and adam_epsilon ##################
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    # optimizer = optim.Adam([{'params': ddp_model.parameters(), 'lr': args.lr}])
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    #train
    loss = 0.0
    global_step = 0
    best_acc, best_count = 0.0, 0

    for ep in range(args.epochs):
        ## start training
        start_time = time()
        ddp_model.train() ######################## You should motify it to ddp_model.train when using DDP
        train_loader_iterator = tqdm(train_loader, desc=f"Epoch:{ep}|Iteration", disable=args.local_rank not in [-1,0])
        for step, batch in enumerate(train_loader_iterator):
            # put data into GPU
            if args.enable_gpu:
                batch = [b.cuda() for b in batch]

            # calculate loss
            batch_loss = ddp_model(*batch)
            loss += batch_loss.item()
            optimizer.zero_grad()
            batch_loss.backward()
            # torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), args.max_grad_norm)
            optimizer.step()
            # scheduler.step()
            global_step += 1

            # logging
            if args.local_rank in [-1, 0]:
                if global_step % args.log_steps == 0:
                    logging.info(
                        'cost_time:{} step:{}, lr:{}, train_loss: {:.5f}'.format(
                            time() - start_time, global_step, optimizer.param_groups[0]['lr'],
                            loss / args.log_steps))
                    loss = 0.0
                if args.local_rank == 0:
                    torch.distributed.barrier()
            else:
                torch.distributed.barrier()

        ## start validating
        if args.local_rank in [-1, 0]:
            ckpt_path = os.path.join(args.model_dir, '{}-{}-{}-{}-{}-{}-{}-epoch-{}.pt'.format(args.model_type, args.data_mode, args.pretrain_LM, args.lr, args.heter_embed_size, args.attr_embed_size, args.attr_vec, ep + 1))
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"Model saved to {ckpt_path}")

            logging.info("Start validation for epoch-{}".format(ep + 1))
            acc = validate(args, model, val_loader)
            logging.info("validation time:{}".format(time() - start_time))
            if acc > best_acc:
                ckpt_path = os.path.join(args.model_dir, '{}-{}-{}-{}-{}-{}-{}-best.pt'.format(args.model_type, args.data_mode, args.pretrain_LM, args.lr, args.heter_embed_size, args.attr_embed_size, args.attr_vec))
                torch.save(model.state_dict(), ckpt_path)
                logging.info(f"Model saved to {ckpt_path}")
                best_acc = acc
                best_count = 0
            else:
                best_count += 1
                if best_count >= args.early_stop:
                    start_time = time()
                    ckpt_path = os.path.join(args.model_dir, '{}-{}-{}-{}-{}-{}-{}-best.pt'.format(args.model_type, args.data_mode, args.pretrain_LM, args.lr, args.heter_embed_size, args.attr_embed_size, args.attr_vec))
                    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                    logging.info("Star testing for best")
                    acc = validate(args, model, test_loader)
                    logging.info("test time:{}".format(time() - start_time))
                    exit()
            if args.local_rank == 0:
                torch.distributed.barrier()
        else:
            torch.distributed.barrier()

    # test
    if args.local_rank in [-1, 0]:
        start_time = time()
        # load checkpoint
        ckpt_path = os.path.join(args.model_dir, '{}-{}-{}-{}-{}-{}-{}-best.pt'.format(args.model_type, args.data_mode, args.pretrain_LM, args.lr, args.heter_embed_size, args.attr_embed_size, args.attr_vec))
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        logging.info('load ckpt:{}'.format(ckpt_path))
        acc = validate(args, model, test_loader)
        logging.info("test time:{}".format(time() - start_time))
        if args.local_rank == 0:
            torch.distributed.barrier()
    else:
        torch.distributed.barrier()
    if args.local_rank != -1:
        cleanup()


@torch.no_grad()
def validate(args, model, dataloader):
    model.eval()

    count = 0
    metrics_total = defaultdict(float)
    for step, batch in enumerate(tqdm(dataloader)):
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

    # add special token
    # additional_special_tokens = ["[paper]", "[author]", "[venue]"]
    # args.additional_special_tokens = additional_special_tokens

    # define tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # load dataset
    ########################## Motify collate_f & think about shuffle in Dataloader
    if args.data_mode in ['text']:
        test_set = load_dataset_text(args, tokenizer, evaluate=True, test=True)
    else:
        raise ValueError('Data Mode is Incorrect here!')

    test_sampler = SequentialSampler(test_set) if args.local_rank == -1 else DistributedSampler(test_set)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, sampler=test_sampler)
    print('Dataset Loading Over!')

    # define model
    model = load_bert(args)
    logging.info('loading model: {}'.format(args.model_type))
    model = model.cuda()

    # load checkpoint
    start_time = time()
    checkpoint = torch.load(args.load_ckpt_name, map_location="cpu")
    model.load_state_dict(checkpoint)
    # model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    logging.info('load ckpt:{}'.format(args.load_ckpt_name))

    # test
    validate(args, model, test_loader)
    logging.info("test time:{}".format(time() - start_time))


@torch.no_grad()
def infer(args):

    # Load data
    # define tokenizer 
    # ################################################ Make sure if you want to add special token here. If you want, remember to make further revise on graphformers.#########
    # tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", additional_special_tokens=additional_special_tokens)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    # load dataset
    ########################## Motify collate_f & think about shuffle in Dataloader
    if args.data_mode in ['text']:
        train_set = load_dataset_text(args, tokenizer, evaluate=False, test=False)
        test_set = load_dataset_text(args, tokenizer, evaluate=True, test=True)
        args.shelves_num, args.author_num, args.publisher_num, args.language_code_num, args.format_num = pickle.load(open(os.path.join(args.data_path, 'mta_num.pkl'),'rb'))
    else:
        raise ValueError('Data Mode is Incorrect here!')
    # define dataloader
    train_sampler = SequentialSampler(train_set) if args.local_rank == -1 else DistributedSampler(train_set)
    test_sampler = SequentialSampler(test_set) if args.local_rank == -1 else DistributedSampler(test_set)

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, sampler=test_sampler)
    print(f'[Process:{args.local_rank}] Dataset Loading Over!')

    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    # config.n_gpu = torch.cuda.device_count()
    args.n_gpu = 1
    args.device = device

    # define model
    model = load_bert(args)
    model.to(args.device)
    model.eval()

    assert args.load == True

    if args.load:
        model.load_state_dict(torch.load(args.load_ckpt_name, map_location="cpu"))
        logging.info('load ckpt:{}'.format(args.load_ckpt_name))

    # obtain embedding on train set for query node
    train_embedding = torch.FloatTensor().to(args.device)
    train_loader_iterator = tqdm(train_loader, desc="Iteration")
    for step, batch in enumerate(train_loader_iterator):
        # put data into GPU
        if args.enable_gpu:
            batch = [b.cuda() for i, b in enumerate(batch) if i< (len(batch) // 2)]

        if args.model_type in ['GraphMeanSage']:
            tweet_neighbour_c = batch[0].shape[1]
            batch[-1] = batch[-1][:,:tweet_neighbour_c]

        # calculate loss
        embedding = model.infer(*batch)
        train_embedding = torch.cat((train_embedding, embedding), dim=0)
    assert train_embedding.shape[0] == len(train_set)

    # obtain embedding on test set for query node
    test_embedding = torch.FloatTensor().to(args.device)
    test_loader_iterator = tqdm(test_loader, desc="Iteration")
    for step, batch in enumerate(test_loader_iterator):
        # put data into GPU
        if args.enable_gpu:
            batch = [b.cuda() for i, b in enumerate(batch) if i< (len(batch) // 2)]

        if args.model_type in ['GraphMeanSage']:
            tweet_neighbour_c = batch[0].shape[1]
            batch[-1] = batch[-1][:,:tweet_neighbour_c]

        # calculate loss
        embedding = model.infer(*batch)
        test_embedding = torch.cat((test_embedding, embedding), dim=0)
    assert test_embedding.shape[0] == len(test_set)

    ######################################### Take care here for the target saving dir ##################################
    np.save(f'output/book_embed/{args.model_type}_transductive.npy', train_embedding.cpu().numpy())
    np.save(f'output/book_embed/{args.model_type}_inductive.npy', test_embedding.cpu().numpy())


@torch.no_grad()
def author_embed(args):

    args.shelves_num, args.author_num, args.publisher_num, args.language_code_num, args.format_num = pickle.load(open(os.path.join(args.data_path, 'mta_num.pkl'),'rb'))
    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    # config.n_gpu = torch.cuda.device_count()
    # args.n_gpu = 1
    args.device = device

    # define model
    model = load_bert(args)
    model.to(args.device)
    model.eval()

    assert args.load == True

    if args.load:
        model.load_state_dict(torch.load(args.load_ckpt_name, map_location="cpu"))
        logging.info('load ckpt:{}'.format(args.load_ckpt_name))

    if args.model_type in ['Heterformer']:
        author_embedding = model.bert.author_embed
    else:
        raise ValueError('Model Type Error!')

    np.save(f'output/book_embed/{args.model_type}_author.npy', author_embedding.cpu().numpy())
    print(f'{args.model_type} Finished!')
