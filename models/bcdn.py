
import torch
import torch.nn as nn
import copy
from transformers import BertModel,BertForSequenceClassification, AutoConfig
import math
import torch.nn.functional as F
from torch.autograd import Variable, Function
from typing import Any, Optional, Tuple
from models.model_utils import BertSelfEncoder, BertCrossEncoder_AttnMap
import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = torch.dot(vec1, vec2)
    norm_vec1 = torch.linalg.norm(vec1)
    norm_vec2 = torch.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

class FP_nn(nn.Module):
    def __init__(self,opt):
        super(FP_nn,self).__init__()
        self.opt = opt
        self.fp_weight_list = nn.ModuleList()
        self.layers = opt.num_layers
        self.mem_dim = opt.bert_dim
        self.attention_heads = opt.attention_heads
        self.bert_dim = opt.bert_dim
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)

        # gcn layer
        self.fp_W = nn.ModuleList()
    
        for layer in range(self.layers):
            fp_input_dim = self.bert_dim if layer == 0 else self.mem_dim
            self.fp_W.append(nn.Linear(fp_input_dim, self.mem_dim))

        self.fp_attn = MultiHeadAttention(opt.attention_heads, opt.bert_dim)

        for j in range(self.layers):
            fp_input_dim = self.bert_dim if j == 0 else self.mem_dim
            self.fp_weight_list.append(nn.Linear(fp_input_dim, self.mem_dim))

    def forward(self, att_inputs, src_mask): 
        fp_attn_tensor = self.fp_attn(att_inputs, att_inputs, src_mask)
        fp_attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(fp_attn_tensor, 1, dim=1)]
        fp_adj = None

        # * Average Multi-head Attention matrixes
        for i in range(self.attention_heads):
            if fp_adj is None:
                fp_adj = fp_attn_adj_list[i]
            else:
                fp_adj += fp_attn_adj_list[i]
        fp_adj = fp_adj / self.attention_heads

        for j in range(fp_adj.size(0)):
            fp_adj[j] = fp_adj[j] - torch.diag(torch.diag(fp_adj[j]))
            fp_adj[j] = fp_adj[j] + torch.eye(fp_adj[j].size(0)).to(self.opt.device)
        fp_adj = src_mask.transpose(1, 2) * fp_adj

        fp_denom = fp_adj.sum(2).unsqueeze(2) + 1
        fp_outputs = att_inputs
        for l in range(self.layers):
            fp_Ax = fp_adj.bmm(fp_outputs)
            fp_AxW = self.fp_weight_list[l](fp_Ax)
            fp_AxW = fp_AxW / fp_denom
            fp_gAxW = F.relu(fp_AxW)
            fp_outputs = self.gcn_drop(fp_gAxW) if l < self.layers - 1 else fp_gAxW
        
        return fp_outputs, fp_adj

class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]

        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn

def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class BCDN(nn.Module):
    def __init__(self, bert, opt):
        super(BCDN, self).__init__()
        self.bert = bert
        self.opt = opt
        self.img_feat_dim = opt.img_feat_dim
        self.roi_num = opt.roi_num
        self.dropout = nn.Dropout(opt.dropout)
        self.line_c = nn.Linear(512, opt.trans_dim)
        self.text_dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.visual_dense = nn.Linear(opt.trans_dim, opt.polarities_dim)
       
        self.image_share = nn.Linear(opt.trans_dim, opt.trans_dim)
        self.text_share = nn.Linear(opt.trans_dim, opt.trans_dim)
        self.fianl_class = nn.Linear(opt.trans_dim, opt.polarities_dim)
        self.fusion = nn.Sequential(
            nn.Linear(opt.trans_dim*4, opt.trans_dim*2),
            nn.Sigmoid(),
            nn.Linear(opt.trans_dim*2, opt.polarities_dim)
        )
        self.mlp1 = nn.Linear(opt.bert_dim, opt.trans_dim)
        self.mlp2 = nn.Linear(opt.bert_dim, opt.trans_dim)
        self.image_class = nn.Linear(opt.trans_dim, opt.polarities_dim)
        self.text_class = nn.Linear(opt.trans_dim, opt.polarities_dim)
        self.self_attn = FP_nn(opt)
        
    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, src_mask, img_feat, full_img_feat = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
        sequence_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids)[0]
        pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids)[1]
        pooled_output = self.dropout(pooled_output)
        
        src_mask = src_mask.unsqueeze(-2)
        self_attn_tensor, _ = self.self_attn(sequence_output, src_mask)
        text_attn = self_attn_tensor.sum(dim=1)
        
        text_sim = self.mlp1(text_attn).unsqueeze(1) 
        text_cls = self.mlp2(text_attn) 
        ori_transformer_inputs = torch.cat((full_img_feat, img_feat), 1).to(dtype=next(self.parameters()).dtype) 
        transformer_inputs = self.line_c(ori_transformer_inputs) 

        text_sim = text_sim.expand(-1, 101, -1) 
        cosine_similarity = F.cosine_similarity(transformer_inputs, text_sim, dim=-1)

        similarity_weights = F.softmax(cosine_similarity, dim=-1) 
        similarity_weights = similarity_weights.unsqueeze(-1)
        image_feat = torch.sum(transformer_inputs * similarity_weights, dim=1)

        image_common = self.image_share(image_feat)
        image_private = image_feat - image_common
        text_common = self.text_share(text_cls)
        text_private = text_cls - text_common #[16,512]
        
        all_features = torch.cat([image_common, image_private, text_common, text_private], dim=1)
        logits = self.fusion(all_features)
        image_logits = self.image_class(image_feat)
    
        return logits, image_common, image_private, text_common, text_private, ori_transformer_inputs, image_feat, image_logits#, pooled_output, text_logits#, final_output#, image_common_logits, text_common_logits, image_specific_logits, text_specific_logits
