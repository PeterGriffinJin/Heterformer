import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel, BertModel

from utils import roc_auc_score, mrr_score, ndcg_score

from IPython import embed

class BERTMF(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.author_num = args.author_num
        self.bert = BertModel(config)
        print(f'author number:{args.author_num}')
        
        self.author_embeddings = nn.Parameter(torch.FloatTensor(args.author_num, args.author_embed))
        nn.init.xavier_normal_(self.author_embeddings)
        self.linear = nn.Linear(args.author_embed, config.hidden_size)
        # self.linear = nn.Parameter(torch.FloatTensor(args.author_embed, config.hidden_size))
        # nn.init.xavier_normal_(self.linear)

        if args.load_bert:
            print(f'Loading Bert:{args.bert_load_model}')
            self.bert.from_pretrained(args.bert_load_model)

        if args.fix:
            for p in self.bert.parameters():
                p.requires_grad = False

    def test(self, token_paper, attention_paper, author_list, **kwargs):

        paper_embeddings = self.bert(input_ids=token_paper, attention_mask=attention_paper)[0][:,0]
        author_embeddings = self.linear(self.author_embeddings[author_list])
        # paper_embeddings = torch.mm(paper_embeddings, self.linear)

        scores = torch.matmul(paper_embeddings, author_embeddings.transpose(0, 1))
        labels = torch.arange(start=0, end=scores.shape[0], dtype=torch.long, device=scores.device)

        predictions = torch.argmax(scores, dim=-1)
        acc = (torch.sum((predictions == labels)) / labels.shape[0]).item()

        scores = scores.cpu().numpy()
        labels = F.one_hot(labels).cpu().numpy()
        auc_all = [roc_auc_score(labels[i], scores[i]) for i in range(labels.shape[0])]
        auc = np.mean(auc_all)
        mrr_all = [mrr_score(labels[i], scores[i]) for i in range(labels.shape[0])]
        mrr = np.mean(mrr_all)
        ndcg_all = [ndcg_score(labels[i], scores[i], labels.shape[1]) for i in range(labels.shape[0])]
        ndcg = np.mean(ndcg_all)

        return {
            "main": acc,
            "acc": acc,
            "auc": auc,
            "mrr": mrr,
            "ndcg": ndcg
        }

    def forward(self, token_paper, attention_paper, author_list, **kwargs):

        paper_embeddings = self.bert(input_ids=token_paper, attention_mask=attention_paper)[0][:,0]
        author_embeddings = self.linear(self.author_embeddings[author_list])
        # author_embeddings = self.author_embeddings[author_list]
        # paper_embeddings = torch.mm(paper_embeddings, self.linear)

        score = torch.matmul(paper_embeddings, author_embeddings.transpose(0, 1))
        labels = torch.arange(start=0, end=score.shape[0], dtype=torch.long, device=score.device)
        loss = F.cross_entropy(score, labels)

        return loss


class BERTMFM(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.author_num = args.author_num
        self.bert = BertModel(config)
        print(f'author number:{args.author_num}')
        
        self.author_embeddings = nn.Parameter(torch.FloatTensor(args.author_num, args.author_embed))
        nn.init.xavier_normal_(self.author_embeddings)
        self.linear_list = nn.ModuleList([nn.Linear(args.author_embed, config.hidden_size) for _ in range(config.num_hidden_layers)])

        if args.load_bert:
            print(f'Loading Bert:{args.bert_load_model}')
            self.bert.from_pretrained(args.bert_load_model)

        if args.fix:
            for p in self.bert.parameters():
                p.requires_grad = False

    def test(self, token_paper, attention_paper, author_list, **kwargs):

        paper_embeddings_list = self.bert(input_ids=token_paper, attention_mask=attention_paper)[2]
        for i, l in enumerate(self.linear_list):
            if i == 0:
                paper_embeddings = paper_embeddings_list[i+1][:,0]
                author_embeddings = l(self.author_embeddings[author_list])
            else:
                paper_embeddings = torch.cat((paper_embeddings, paper_embeddings_list[i+1][:,0]), dim=-1)
                author_embeddings = torch.cat((author_embeddings, l(self.author_embeddings[author_list])), dim=-1)

        scores = torch.matmul(paper_embeddings, author_embeddings.transpose(0, 1))
        labels = torch.arange(start=0, end=scores.shape[0], dtype=torch.long, device=scores.device)

        predictions = torch.argmax(scores, dim=-1)
        acc = (torch.sum((predictions == labels)) / labels.shape[0]).item()

        scores = scores.cpu().numpy()
        labels = F.one_hot(labels).cpu().numpy()
        auc_all = [roc_auc_score(labels[i], scores[i]) for i in range(labels.shape[0])]
        auc = np.mean(auc_all)
        mrr_all = [mrr_score(labels[i], scores[i]) for i in range(labels.shape[0])]
        mrr = np.mean(mrr_all)
        ndcg_all = [ndcg_score(labels[i], scores[i], labels.shape[1]) for i in range(labels.shape[0])]
        ndcg = np.mean(ndcg_all)

        return {
            "main": acc,
            "acc": acc,
            "auc": auc,
            "mrr": mrr,
            "ndcg": ndcg
        }

    def forward(self, token_paper, attention_paper, author_list, **kwargs):

        paper_embeddings_list = self.bert(input_ids=token_paper, attention_mask=attention_paper)[2]
        for i, l in enumerate(self.linear_list):
            if i == 0:
                paper_embeddings = paper_embeddings_list[i+1][:,0]
                author_embeddings = l(self.author_embeddings[author_list])
            else:
                paper_embeddings = torch.cat((paper_embeddings, paper_embeddings_list[i+1][:,0]), dim=-1)
                author_embeddings = torch.cat((author_embeddings, l(self.author_embeddings[author_list])), dim=-1)

        score = torch.matmul(paper_embeddings, author_embeddings.transpose(0, 1))
        labels = torch.arange(start=0, end=score.shape[0], dtype=torch.long, device=score.device)
        loss = F.cross_entropy(score, labels)

        return loss
