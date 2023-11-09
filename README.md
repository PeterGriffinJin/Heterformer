# Heterformer

## Data Processing
1. Download raw data from [DBLP](https://originalstatic.aminer.cn/misc/dblp.v12.7z), [Twitter](https://drive.google.com/file/d/0Byrzhr4bOatCRHdmRVZ1YVZqSzA/view?resourcekey=0-3_R5EWrLYjaVuysxPTqe5A) and [Goodreads](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home).
2. Data processing: Run the cells in data/$dataset/data_processing.ipynb for first step data processing.
3. Network Sampling: Run the cells in data/$dataset/sampling.ipynb for ego-network sampling and train/val/test data generation.
4. Pretrain data: Run the cells in data/$dataset/generate_pretrain_data.ipynb for textless node pretraining data generation.

## Train
1. Pretrain textless node embeddings.
Take Goodreads dataset as an example.
```
cd pretrain/
bash run.sh
```

2. Prepare textless node embedding file for Heterformer training.

Run the cells in pretrain/transfer_embed.ipynb

3. Heterformer training.
```
cd ..
python main.py --data_path data/$dataset --model_type Heterformer --pretrain_embed True --pretrain_dir data/$dataset/pretrain_embed
```

## Test
```
python main.py --data_path data/$dataset --model_type Heterformer --mode test --load_ckpt_name $load_ckpt_dir
```

## Inference
```
python main.py --data_path data/$dataset --model_type Heterformer --mode infer --load 1 --load_ckpt_name $load_ckpt_dir
```

## Downstream Tasks
#### Transductive Text-rich node classification
```
cd downstream/
python classification.py --mode transductive --dataset $dataset --method Heterformer
```

#### Inductive Text-rich node classification
```
python classification.py --mode inductive --dataset $dataset --method Heterformer
```

#### Textless node classification
```
python author_classification.py --dataset $dataset --method Heterformer
```

#### Node Clustering
```
python clustering.py --mode transductive --dataset $dataset --method Heterformer
```

#### Retrieval
```
python retrieval.py --method Heterformer
```
