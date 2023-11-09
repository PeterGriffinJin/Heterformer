import os
import logging
import argparse
from pathlib import Path
import torch.multiprocessing as mp

from run import train, test
from utils import setuplogging, str2bool, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", choices=['train', 'test'])
parser.add_argument("--data_path", type=str, default="book_pretrain/", )

parser.add_argument("--train_batch_size", type=int, default=50)
parser.add_argument("--val_batch_size", type=int, default=100)
parser.add_argument("--test_batch_size", type=int, default=100)

parser.add_argument("--lr", type=float, default=1e-4)

parser.add_argument("--pretrain_mode", type=str, default='s', choices=['s', 'v', 'a', 'p','l', 'f', 'merge'])  # path to save
parser.add_argument("--model_dir", type=str, default='ckpt_movie/', choices=['ckpt_author/', 'ckpt_venue/', 'ckpt_movie/', 'ckpt_Electronics/', 'ckpt_crime_book/', 'ckpt_stackoverflow/'])  # path to save
parser.add_argument("--enable_gpu", type=str2bool, default=True)

parser.add_argument("--max_length", type=int, default=64)
parser.add_argument("--author_embed", type=int, default=64)

# parser.add_argument("--author_num", type=int, default=2717797) # 2717797 for author, 28638 for venue on DBLP
# parser.add_argument("--venue_num", type=int, default=28638)

# parser.add_argument("--mention_num", type=int, default=24089) # 76398 for author, 24089 for mention, 72297 for tag on Tweet
# parser.add_argument("--tag_num", type=int, default=72297) 
# parser.add_argument("--author_num", type=int, default=76398) 

parser.add_argument("--shelves_num", type=int, default=6632) # 6632 for shelves, 205891 for author, 62934 for publisher on Goodread
parser.add_argument("--author_num", type=int, default=205891) # 139 for language_code, 768 for format on Goodread
parser.add_argument("--publisher_num", type=int, default=62934) 
parser.add_argument("--language_code_num", type=int, default=139) 
parser.add_argument("--format_num", type=int, default=768) 

# # parser.add_argument("--node_num", type=int, default=173986) # Amazon movie dataset
# # parser.add_argument("--node_num", type=int, default=255344) # Amazon Electronics dataset
# # parser.add_argument("--node_num", type=int, default=385203) # Goodreads crime dataset
# parser.add_argument("--node_num", type=int, default=240869) # stackoverflow dataset


# model training
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--early_stop", type=int, default=2)
parser.add_argument("--log_steps", type=int, default=1000)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--load", type=str2bool, default=False)
parser.add_argument("--load_bert", type=str2bool, default=True)
parser.add_argument("--fix", type=str2bool, default=True)

# turing
parser.add_argument("--model_type", default="BERTMF", type=str, choices=['BERTMF', 'BERTMFM'], help='Should be in [BERTMF, BERTMFM]')
parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str,
                    help="Path to pre-trained model or shortcut name. ")
parser.add_argument("--bert_load_model", default="bert-base-uncased", type=str,
                    help="Path to pre-trained model or shortcut name. ")

parser.add_argument(
        "--load_ckpt_name",
        type=str,
        default='ckpt/BERTMF-best.pt',
        help="choose which ckpt to load and test"
    )

args = parser.parse_args()

# DBLP
# if args.pretrain_mode == 'a':
#     args.model_dir = 'ckpt_DBLP/ckpt_author/'
# elif args.pretrain_mode == 'v':
#     args.model_dir = 'ckpt_DBLP/ckpt_venue/'
#     args.author_num = args.venue_num
# else:
#     raise ValueError('Pretrain Mode Error!')

# Twitter
# if args.pretrain_mode == 'a':
#     args.model_dir = 'ckpt_Tweet/ckpt_author/'
# elif args.pretrain_mode == 'm':
#     args.model_dir = 'ckpt_Tweet/ckpt_mention/'
#     args.author_num = args.mention_num
# elif args.pretrain_mode == 't':
#     args.model_dir = 'ckpt_Tweet/ckpt_tag/'
#     args.author_num = args.tag_num
# else:
#     raise ValueError('Pretrain Mode Error!')

# Goodreads
if args.pretrain_mode == 'a':
    args.model_dir = 'ckpt_book/ckpt_author/'
elif args.pretrain_mode == 's':
    args.model_dir = 'ckpt_book/ckpt_shelves/'
    args.author_num = args.shelves_num
elif args.pretrain_mode == 'p':
    args.model_dir = 'ckpt_book/ckpt_publisher/'
    args.author_num = args.publisher_num
elif args.pretrain_mode == 'l':
    args.model_dir = 'ckpt_book/ckpt_language/'
    args.author_num = args.language_code_num
elif args.pretrain_mode == 'f':
    args.model_dir = 'ckpt_book/ckpt_format/'
    args.author_num = args.format_num
else:
    raise ValueError('Pretrain Mode Error!')

# # Movie
# if args.pretrain_mode == 'merge':
#     args.author_num = args.node_num
# else:
#     raise ValueError('Pretrain Mode Error!')
logging.info(args)


if __name__ == "__main__":

    set_seed(args.random_seed)
    setuplogging()

    print(os.getcwd())
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == 'train':
        print('-----------train------------')
        train(args)

    if args.mode == 'test':
        print('-------------test--------------')
        test(args)
