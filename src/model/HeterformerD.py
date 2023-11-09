import os
import math
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_bert import BertSelfAttention, BertLayer, BertEmbeddings, BertPreTrainedModel

from src.utils import roc_auc_score, mrr_score, ndcg_score

from IPython import embed

############################ Take care that we assume the last one is venue #########################
############################ Take care that here we assume that there are three relations #########################
class GraphAggregation(BertSelfAttention):
    def __init__(self, config):
        super(GraphAggregation, self).__init__(config)
        self.output_attentions = False

    def forward(self, hidden_states, attention_mask):
        
        query = self.query(hidden_states[:, :1])  # B 1 D
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        station_embed = self.multi_head_attention(query=query,
                                                    key=key,
                                                    value=value,
                                                    attention_mask=attention_mask)[0]  # B 1 D
        
        station_embed = station_embed.squeeze(1)

        return station_embed

    def multi_head_attention(self, query, key, value, attention_mask):
        query_layer = self.transpose_for_scores(query)
        key_layer = self.transpose_for_scores(key)
        value_layer = self.transpose_for_scores(value)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return (context_layer, attention_probs) if self.output_attentions else (context_layer,)


class HGraphBertEncoderD(nn.Module):
    def __init__(self, config):
        super(HGraphBertEncoderD, self).__init__()

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

        self.graph_attention = GraphAggregation(config=config)

    def forward(self,
                hidden_states,
                attention_mask,
                shelves_embed,
                author_embed,
                publisher_embed,
                language_code_embed,
                format_embed,
                node_mask):

        all_hidden_states = ()
        all_attentions = ()

        all_nodes_num, seq_length, emb_dim = hidden_states.shape
        batch_size, _, _, _ = node_mask.shape
        subgraph_node_num = self.book_neighbour + 1

        mta_embed = torch.cat((shelves_embed, author_embed, publisher_embed, language_code_embed, format_embed), dim=1)

        paper_mask = node_mask[:,:,:,:(1+self.book_neighbour)]
        mta_mask = torch.cat((node_mask[:,:,:,:1],node_mask[:,:,:,(1+self.book_neighbour):]), dim=-1)

        # second point
        # embed()

        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if i > 0:

                hidden_states = hidden_states.view(batch_size, subgraph_node_num, seq_length, emb_dim)  # B SN L D
                cls_emb = hidden_states[:, :, 2].clone()  # B SN D
                ############# maybe here you have to author_embed.clone(), venue_embed.clone(), I'm not sure. #############

                ################### You may add author_embed transfer and venue_embed transfer here. ######################
                #  xxxxx
                ################### You may add author_embed transfer and venue_embed transfer here. ######################

                # prepare for the graph aggregation
                center_and_author_venue_embed = torch.cat((cls_emb[:,:1], mta_embed), dim=1)

                station_emb_paper = self.graph_attention(hidden_states=cls_emb, attention_mask=paper_mask)  # B D
                station_emb_mta = self.graph_attention(hidden_states=center_and_author_venue_embed, attention_mask=mta_mask)  # B D

                # update the station in the query/key
                hidden_states[:, 0, 0] = station_emb_paper
                hidden_states[:, 0, 1] = station_emb_mta
                hidden_states = hidden_states.view(all_nodes_num, seq_length, emb_dim)

                layer_outputs = layer_module(hidden_states, attention_mask=attention_mask)

            else:
                temp_attention_mask = attention_mask.clone()
                temp_attention_mask[::subgraph_node_num, :, :, :2] = -10000.0
                layer_outputs = layer_module(hidden_states, attention_mask=temp_attention_mask)

            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class HeterFormerDs(BertPreTrainedModel):
    def __init__(self, config):
        super(HeterFormerDs, self).__init__(config=config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.embeddings = BertEmbeddings(config=config)
        self.encoder = HGraphBertEncoderD(config=config)

    def init_mta_embed(self, pretrain_embed, pretrain_dir, book_neighbour, shelves_neighbour, author_neighbour, shelves_num, author_num, publisher_num, language_code_num, format_num, heter_embed_size):
        self.book_neighbour = book_neighbour
        self.shelves_neighbour = shelves_neighbour
        self.author_neighbour = author_neighbour
        self.shelves_num = shelves_num
        self.author_num = author_num
        self.publisher_num = publisher_num
        self.language_code_num = language_code_num
        self.format_num = format_num
        self.heter_embed_size = heter_embed_size

        if not pretrain_embed:
            self.shelves_embed = nn.Parameter(torch.FloatTensor(self.shelves_num, self.heter_embed_size))
            self.author_embed = nn.Parameter(torch.FloatTensor(self.author_num, self.heter_embed_size))
            self.publisher_embed = nn.Parameter(torch.FloatTensor(self.publisher_num, self.heter_embed_size))
            self.language_code_embed = nn.Parameter(torch.FloatTensor(self.language_code_num, self.heter_embed_size))
            self.format_embed = nn.Parameter(torch.FloatTensor(self.format_num, self.heter_embed_size))

            nn.init.xavier_normal_(self.shelves_embed)
            nn.init.xavier_normal_(self.author_embed)
            nn.init.xavier_normal_(self.publisher_embed)
            nn.init.xavier_normal_(self.language_code_embed)
            nn.init.xavier_normal_(self.format_embed)

            self.shelves_to_text_transform = nn.Linear(self.heter_embed_size, self.hidden_size)
            self.author_to_text_transform = nn.Linear(self.heter_embed_size, self.hidden_size)
            self.publisher_to_text_transform = nn.Linear(self.heter_embed_size, self.hidden_size)
            self.language_code_to_text_transform = nn.Linear(self.heter_embed_size, self.hidden_size)
            self.format_to_text_transform = nn.Linear(self.heter_embed_size, self.hidden_size)

        else:
            checkpoint = pickle.load(open(os.path.join(pretrain_dir, f'shelves_MF_{self.heter_embed_size}.pt'),'rb'))
            self.shelves_embed = nn.Parameter(checkpoint['author_embeddings']) # this "author_embedding" name might be ambiguous, but it's not a bug, relax
            self.shelves_to_text_transform = nn.Linear(self.heter_embed_size, self.hidden_size)
            with torch.no_grad():
                self.shelves_to_text_transform.weight.copy_(checkpoint['linear.weight'])
                self.shelves_to_text_transform.bias.copy_(checkpoint['linear.bias'])

            checkpoint = pickle.load(open(os.path.join(pretrain_dir, f'author_MF_{self.heter_embed_size}.pt'),'rb'))
            self.author_embed = nn.Parameter(checkpoint['author_embeddings']) # this "author_embedding" name might be ambiguous, but it's not a bug, relax
            self.author_to_text_transform = nn.Linear(self.heter_embed_size, self.hidden_size)
            with torch.no_grad():
                self.author_to_text_transform.weight.copy_(checkpoint['linear.weight'])
                self.author_to_text_transform.bias.copy_(checkpoint['linear.bias'])

            checkpoint = pickle.load(open(os.path.join(pretrain_dir, f'publisher_MF_{self.heter_embed_size}.pt'),'rb'))
            self.publisher_embed = nn.Parameter(checkpoint['author_embeddings']) # this "author_embedding" name might be ambiguous, but it's not a bug, relax
            self.publisher_to_text_transform = nn.Linear(self.heter_embed_size, self.hidden_size)
            with torch.no_grad():
                self.publisher_to_text_transform.weight.copy_(checkpoint['linear.weight'])
                self.publisher_to_text_transform.bias.copy_(checkpoint['linear.bias'])

            checkpoint = pickle.load(open(os.path.join(pretrain_dir, f'language_code_MF_{self.heter_embed_size}.pt'),'rb'))
            self.language_code_embed = nn.Parameter(checkpoint['author_embeddings']) # this "author_embedding" name might be ambiguous, but it's not a bug, relax
            self.language_code_to_text_transform = nn.Linear(self.heter_embed_size, self.hidden_size)
            with torch.no_grad():
                self.language_code_to_text_transform.weight.copy_(checkpoint['linear.weight'])
                self.language_code_to_text_transform.bias.copy_(checkpoint['linear.bias'])

            checkpoint = pickle.load(open(os.path.join(pretrain_dir, f'format_MF_{self.heter_embed_size}.pt'),'rb'))
            self.format_embed = nn.Parameter(checkpoint['author_embeddings']) # this "author_embedding" name might be ambiguous, but it's not a bug, relax
            self.format_to_text_transform = nn.Linear(self.heter_embed_size, self.hidden_size)
            with torch.no_grad():
                self.format_to_text_transform.weight.copy_(checkpoint['linear.weight'])
                self.format_to_text_transform.bias.copy_(checkpoint['linear.bias'])

        self.encoder.book_neighbour = self.book_neighbour

    def forward(self,
                input_ids,
                attention_mask,
                mta_neighbors,
                neighbor_mask=None):
        all_nodes_num, seq_length = input_ids.shape
        # batch_size, subgraph_node_num = neighbor_mask.shape
        subgraph_node_num = self.book_neighbour + 1

        # # split neighbour_mask
        # paper_neighbour_mask, author_neighbour_mask, venue_neighbour_mask = neighbor_mask[:,:(1+self.paper_neighbour)], neighbor_mask[:,(1+self.paper_neighbour):(1+self.paper_neighbour+self.author_neighbour)], neighbor_mask[:,-1:]

        # obtain embedding
        embedding_output = self.embeddings(input_ids=input_ids)
        shelves_embed = self.shelves_to_text_transform(self.shelves_embed[mta_neighbors[:, :self.shelves_neighbour]])
        author_embed = self.author_to_text_transform(self.author_embed[mta_neighbors[:, self.shelves_neighbour:(self.shelves_neighbour+self.author_neighbour)]])
        publisher_embed = self.publisher_to_text_transform(self.publisher_embed[mta_neighbors[:, -3:-2]])
        language_code_embed = self.language_code_to_text_transform(self.language_code_embed[mta_neighbors[:, -2:-1]])
        format_embed = self.format_to_text_transform(self.format_embed[mta_neighbors[:, -1:]])

        # Add station attention mask
        ############################## we add two new position, one for text neighbour, one for non-text neighbour ####################################
        station_mask = torch.zeros((all_nodes_num, 2), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([station_mask, attention_mask], dim=-1)  # N 2+L
        attention_mask[::(subgraph_node_num), :2] = 1.0  # only use the station for main nodes

        extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0
        node_mask = (1.0 - neighbor_mask[:, None, None, :]) * -10000.0
        # node_mask = (1.0 - paper_neighbour_mask[:, None, None, :]) * -10000.0
        # author_neighbour_mask = (1.0 - author_neighbour_mask[:, None, None, :]) * -10000.0
        # venue_neighbour_mask = (1.0 - venue_neighbour_mask[:, None, None, :]) * -10000.0

        # Add station_placeholder
        ############################# we add two new position, one for text neighbour, one for non-text neighbour ######################################
        station_placeholder = torch.zeros(all_nodes_num, 2, embedding_output.size(-1)).type(
            embedding_output.dtype).to(embedding_output.device)
        embedding_output = torch.cat([station_placeholder, embedding_output], dim=1)  # N 2+L D

        # stop point 1
        # embed()

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            shelves_embed=shelves_embed,
            author_embed=author_embed,
            publisher_embed=publisher_embed,
            language_code_embed=language_code_embed,
            format_embed=format_embed,
            node_mask=node_mask)

        # stop point 3
        # embed()

        return encoder_outputs


class HeterFormerDsForNeighborPredict(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = HeterFormerDs(config)
        self.hidden_size = config.hidden_size
        self.init_weights()

    def init_mta_embed(self, pretrain_embed, pretrain_dir):
        self.bert.init_mta_embed(pretrain_embed, pretrain_dir, self.book_neighbour, self.shelves_neighbour, self.author_neighbour, self.shelves_num, self.author_num, self.publisher_num, self.language_code_num, self.format_num, self.heter_embed_size)

    def infer(self, input_ids_node_and_neighbors_batch, attention_mask_node_and_neighbors_batch,
                mta_neighbors_batch, mask_node_and_neighbors_batch):
        '''
        B: batch size, N: 1 + neighbour_num, L: max_token_len, D: hidden dimmension
        '''

        B, N, L = input_ids_node_and_neighbors_batch.shape
        D = self.hidden_size
        input_ids = input_ids_node_and_neighbors_batch.view(B * N, L)
        attention_mask = attention_mask_node_and_neighbors_batch.view(B * N, L)

        hidden_states = self.bert(input_ids, attention_mask, mta_neighbors_batch, mask_node_and_neighbors_batch)
        
        last_hidden_states = hidden_states[0]
        
        cls_embeddings = last_hidden_states[:, 2].view(B, N, D)  # [B,N,D]
        node_embeddings = cls_embeddings[:, 0, :]  # [B,D]

        # stop point 4
        # embed()

        return node_embeddings

    def test(self, input_ids_query_and_neighbors_batch, attention_mask_query_and_neighbors_batch,
                query_mta_neighbors_batch, mask_query_and_neighbors_batch,
                input_ids_key_and_neighbors_batch, attention_mask_key_and_neighbors_batch, 
                key_mta_neighbors_batch, mask_key_and_neighbors_batch,
                **kwargs):
        query_embeddings = self.infer(input_ids_query_and_neighbors_batch, attention_mask_query_and_neighbors_batch,
                                        query_mta_neighbors_batch, mask_query_and_neighbors_batch)
        key_embeddings = self.infer(input_ids_key_and_neighbors_batch, attention_mask_key_and_neighbors_batch,
                                    key_mta_neighbors_batch, mask_key_and_neighbors_batch)
        scores = torch.matmul(query_embeddings, key_embeddings.transpose(0, 1))
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

    def forward(self, input_ids_query_and_neighbors_batch, attention_mask_query_and_neighbors_batch,
                query_mta_neighbors_batch, mask_query_and_neighbors_batch, \
                input_ids_key_and_neighbors_batch, attention_mask_key_and_neighbors_batch, 
                key_mta_neighbors_batch, mask_key_and_neighbors_batch,
                **kwargs):
        
        query_embeddings = self.infer(input_ids_query_and_neighbors_batch, attention_mask_query_and_neighbors_batch,
                                        query_mta_neighbors_batch, mask_query_and_neighbors_batch)
        key_embeddings = self.infer(input_ids_key_and_neighbors_batch, attention_mask_key_and_neighbors_batch,
                                    key_mta_neighbors_batch, mask_key_and_neighbors_batch)
        score = torch.matmul(query_embeddings, key_embeddings.transpose(0, 1))
        labels = torch.arange(start=0, end=score.shape[0], dtype=torch.long, device=score.device)
        loss = F.cross_entropy(score, labels)
        return loss
