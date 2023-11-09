import os
import logging
import argparse
from pathlib import Path
import torch.multiprocessing as mp

from src.run import train, test, infer, author_embed
from src.utils import setuplogging, str2bool, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", choices=['train', 'test', 'infer', 'author_embed'])
parser.add_argument("--data_path", type=str, default="data/book", choices=['data/book'])
parser.add_argument("--model_dir", type=str, default='ckpt/book', choices=['ckpt/book'])  # path to save
parser.add_argument("--data_mode", default="text", type=str, choices=['text'])
parser.add_argument("--pretrain_embed", type=str2bool, default=False) # use pretrained textless node embedding or not
parser.add_argument("--pretrain_dir", default="data/book/pretrain_embed", type=str, choices=['data/book/pretrain_embed']) # pretrain author/venue embedding dir

# turing
parser.add_argument("--model_type", default="GraphMeanSage", type=str, choices=['GraphMeanSage', 'Heterformer'])
parser.add_argument("--pretrain_LM", type=str2bool, default=True)
parser.add_argument("--heter_embed_size", type=int, default=64)
parser.add_argument("--attr_embed_size", type=int, default=768)
parser.add_argument("--attr_vec", type=str, default='tfidf', choices=['cnt', 'tfidf'])

# some parameters fixed depend on dataset
parser.add_argument("--max_length", type=int, default=64) # this parameter should be the same for all models for one particular dataset
parser.add_argument("--train_batch_size", type=int, default=30)
parser.add_argument("--val_batch_size", type=int, default=100)
parser.add_argument("--test_batch_size", type=int, default=100)

# parser.add_argument("--neighbor_num", type=int, default=5)
parser.add_argument("--book_neighbour", type=int, default=5)
parser.add_argument("--shelves_neighbour", type=int, default=5)
parser.add_argument("--author_neighbour", type=int, default=2)

# distribution
parser.add_argument("--local_rank", type=int, default=-1)

# model training (these parameters can be fixed)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--early_stop", type=int, default=3)
parser.add_argument("--log_steps", type=int, default=100)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--load", type=str2bool, default=False)
parser.add_argument("--max_grad_norm", type=int, default=1)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--adam_epsilon", type=float, default=1e-8)
parser.add_argument("--enable_gpu", type=str2bool, default=True)

# load checkpoint or test
parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str,
                    help="Path to pre-trained model or shortcut name. ")
parser.add_argument(
        "--load_ckpt_name",
        type=str,
        default='ckpt/book/xxx.pt',
        help="choose which ckpt to load and test"
    )

# half float
parser.add_argument("--fp16", type=str2bool, default=True)

args = parser.parse_args()

if args.local_rank in [-1,0]:
    logging.info(args)
    print(args)

# ## make sure that model_type and data_mode fit each other
# p_model = ['GraphMeanSage']
# pp_model = ['HeterformerD']
# text_model = p_model + pp_model

# if args.data_mode in ['text']:
#     assert args.model_type in text_model
# else:
#     raise ValueError('Wrong Data Mode!')

# # make sure the data_dir corresponds to data_mode
# if args.data_mode in ['text']:
#     assert args.data_path in ['data/book']

if __name__ == "__main__":

    set_seed(args.random_seed)
    setuplogging()

    if args.local_rank in [-1,0]:
        print(os.getcwd())
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == 'train':
        if args.local_rank in [-1,0]:
            print('-----------train------------')
        train(args)

    if args.mode == 'test':
        print('-------------test--------------')
        ################## You should use single GPU for testing. ####################
        assert args.local_rank == -1
        test(args)

    if args.mode == 'infer':
        print('-------------infer--------------')
        ################## You should use single GPU for infering. ####################
        assert args.local_rank == -1
        infer(args)

    if args.mode == 'author_embed':
        assert args.model_type in ['HeterformerD']
        print('-------------author embedding--------------')
        ################## You should use single GPU for infering. ####################
        assert args.local_rank == -1
        author_embed(args)
