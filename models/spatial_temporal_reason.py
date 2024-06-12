# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
""" Spatial-temporal Reasoning Module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.structures import Instances
from .utils import time_position_embedding
from util.misc import inverse_sigmoid
from typing import Optional
from torch import Tensor
import copy
from .deformable_transformer_plus import VisionLanguageFusionModule

class SpatialTemporalReasoner(nn.Module):
    def __init__(self, 
                 history_reasoning=True,
                 embed_dims=256, 
                 hist_len=4, 
                 num_classes=1):
        super(SpatialTemporalReasoner,self).__init__()
        self.embed_dims = embed_dims
        self.hist_len = hist_len


        self.num_classes = num_classes

        # for timing modeling
        self.history_reasoning = history_reasoning


        # fusion modeling
        self.fusion_module = VisionLanguageFusionModule(d_model=embed_dims, nhead=8)

        self.init_params_and_layers()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return 
    
    def init_params_and_layers(self):
        # Modules for history reasoning
        if self.history_reasoning:
            # temporal transformer
            hist_transformerlayer = TempTransformerLayer()
            self.hist_transformer = TempTransformer(hist_transformerlayer,3)
            object_transfrmerlayer = TempTransformerLayer()
            self.obiect_attention = TempTransformer(object_transfrmerlayer,3)
           
            # classification refinement
            self.track_cls = nn.Linear(self.embed_dims, 1)
            # reference refinement
            self.track_ref = nn.Linear(self.embed_dims, 1)
            # localization refinement
            self.track_box = MLP(self.embed_dims, self.embed_dims, 4, 3)
            prior_prob = 0.01 # positive prior probability

            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.track_cls.bias.data = torch.ones(1) * bias_value
            nn.init.constant_(self.track_box.layers[-1].weight.data, 0)
            nn.init.constant_(self.track_box.layers[-1].bias.data, 0)
            self.track_ref.bias.data = torch.ones(1) * bias_value
            
            self.ts_query_embed = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )
                       

            
        return

    def forward(self, track_instances,text_info,training=False):
        # 1. Prepare the spatial-temporal features
        track_instances = self.frame_shift(track_instances)

        fusion_change = False

        # 2. History reasoning
        if self.history_reasoning:
            track_instances = self.forward_history_reasoning(track_instances)
            track_instances = self.forward_history_refine(track_instances,training)

        return track_instances


    def forward_history_reasoning(self, track_instances: Instances):
        """Using history information to refine the current frame features
        """
        if len(track_instances) == 0:
            return track_instances
        
        embed = track_instances.output_embedding

        assert len(embed)==len(track_instances),"wrongwrongwrong,vslid_idxes func"
        if len(embed) == 0:
            return track_instances
        
        hist_embed = track_instances.hist_embeds 
        hist_padding_mask = track_instances.hist_padding_masks 
        
        # get position embedding for cross attention
        ts_pe = time_position_embedding(hist_embed.shape[0],self.hist_len,
                                        self.embed_dims,hist_embed.device)
        ts_pe = self.ts_query_embed(ts_pe)
        # time modeling
        dim = track_instances.query_pos.shape[1]
        hist_pe = track_instances.query_pos[:, :dim // 2][:,None,:]
        temp_embed = self.hist_transformer(embed[:,None,:],hist_embed,query_embed=ts_pe[:, -1:, :],pos_embed=ts_pe,key_padding_mask=hist_padding_mask)# 300,1,256
        final_embed = self.obiect_attention(temp_embed.transpose(0, 1),temp_embed.transpose(0, 1),query_embed=hist_pe.transpose(0, 1),pos_embed=hist_pe.transpose(0, 1))[0]
        # update track_instances
        track_instances.output_embedding = final_embed.clone()
        track_instances.hist_embeds[:,-1] = final_embed.clone().detach()
        return track_instances
    
    def forward_history_refine(self, track_instances: Instances,training):
        if len(track_instances) == 0:
            return track_instances
        embed = track_instances.output_embedding 

        if len(embed) == 0:
            return track_instances
        """Classification"""
        logits = self.track_cls(track_instances.output_embedding) # 8,1
        track_instances.cache_pred_logits = logits
        if training:
            track_instances.cache_scores = logits.sigmoid().max(dim=-1).values
        else:
            track_instances.cache_scores = logits.sigmoid().max(dim=-1).values
        '''reference scores'''
        reference_scores = self.track_ref(track_instances.output_embedding)
        track_instances.cache_pred_refers = reference_scores.clone()
        """Localization"""
        deltas = self.track_box(track_instances.output_embedding) # 8,4
        box = inverse_sigmoid(track_instances.cache_pred_boxes.clone())
        deltas = deltas+box
        track_instances.cache_pred_boxes = deltas.sigmoid()
        

        return track_instances
    
    def frame_shift(self, track_instances: Instances):
        device =  track_instances.query_pos.device

        """History reasoning"""
        # embeds
        track_instances.hist_embeds = track_instances.hist_embeds.clone()
        track_instances.hist_embeds = torch.cat((
            track_instances.hist_embeds[:,1:,:],track_instances.output_embedding[:,None,:]),dim=1)
        # padding masks
        track_instances.hist_padding_masks = torch.cat((
            track_instances.hist_padding_masks[:, 1:], 
            torch.zeros((len(track_instances), 1), dtype=torch.bool, device=device)), 
            dim=1)  
        return track_instances

    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TempTransformer(nn.Module):
    def __init__(self, decoder_layer, num_layers=2):
        super(TempTransformer,self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self,embed,hist_embed,query_embed, pos_embed,key_padding_mask=None):
        query = embed.transpose(0,1) 
        memory = hist_embed.transpose(0,1)
        query_embed = query_embed.transpose(0,1) 
        pos_embed = pos_embed.transpose(0,1) 

        for layer in self.layers:
            query = layer(query,memory,query_embed,pos_embed,key_padding_mask=key_padding_mask)
        
        return query.transpose(0,1)


class TempTransformerLayer(nn.Module):
    def __init__(self,dim_in=256,heads=8,dropout=0.1,hidden_dim=1024):
        super(TempTransformerLayer,self).__init__()
        # cross Attention
        self.cross_attn = nn.MultiheadAttention(dim_in,heads,dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim_in)
        # self Attention
        self.self_attn = nn.MultiheadAttention(dim_in,heads,dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim_in)

        # ffn
        self.linear1 = nn.Linear(dim_in,hidden_dim)
        self.activation = nn.ReLU(True)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim,dim_in)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(dim_in)
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def _forward_self_attn(self, tgt, query_pos, attn_mask=None):

        q = k = self.with_pos_embed(tgt, query_pos)
        if attn_mask is not None:
            tgt2 = self.self_attn(q, k, tgt,
                                  attn_mask=attn_mask)[0]
        else:
            tgt2 = self.self_attn(q, k, tgt)[0]
        tgt = tgt + self.dropout2(tgt2)
        return self.norm2(tgt)

    def forward(self,target,memory,query_embed, pos_embed, key_padding_mask=None):

        # self
        tgt = self._forward_self_attn(target,query_embed)
        # cross
        tgt2 = self.cross_attn(self.with_pos_embed(tgt,query_embed),self.with_pos_embed(memory,pos_embed),memory,key_padding_mask=key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


    