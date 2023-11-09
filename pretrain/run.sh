#!/bin/bash

# DBLP
# CUDA_VISIBLE_DEVICES=2 python main.py --data_path DBLP_pretrain/ --pretrain_mode v --model_dir ckpt_venue/ --author_embed 64 --max_length 32
# CUDA_VISIBLE_DEVICES=3 python main.py --data_path DBLP_pretrain/ --pretrain_mode v --model_dir ckpt_venue/ --author_embed 64 --max_length 32


# book
CUDA_VISIBLE_DEVICES=3 python main.py --data_path book_pretrain/ --pretrain_mode s --author_embed 64 --max_length 64
CUDA_VISIBLE_DEVICES=3 python main.py --data_path book_pretrain/ --pretrain_mode a --author_embed 64 --max_length 64
CUDA_VISIBLE_DEVICES=3 python main.py --data_path book_pretrain/ --pretrain_mode p --author_embed 64 --max_length 64
CUDA_VISIBLE_DEVICES=3 python main.py --data_path book_pretrain/ --pretrain_mode l --author_embed 64 --max_length 64
CUDA_VISIBLE_DEVICES=3 python main.py --data_path book_pretrain/ --pretrain_mode f --author_embed 64 --max_length 64
