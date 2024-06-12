# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
DETR model and criterion classes.
"""
import os
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List
import copy

from util import box_ops, checkpoint
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from models.structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou
from transformers import RobertaModel, RobertaTokenizerFast
from transformers import BertTokenizerFast, BertModel
from transformers import AutoModel, AutoTokenizer
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe, Vocab, FastText
from einops import rearrange, repeat

from .backbone import build_backbone
from .matcher import build_matcher
from .deformable_transformer_plus import build_deforamble_transformer, FeatureResizer, VisionLanguageFusionModule
from .qim_pro import build as build_query_interaction_layer
# from .qim import build as build_query_interaction_layer
from .memory_bank import build_memory_bank
from .deformable_detr import SetCriterion, MLP
from .segmentation import sigmoid_focal_loss
from .position_encoding import PositionEmbeddingSine1D
from torch.cuda.amp import autocast as autocast
from .spatial_temporal_reason import SpatialTemporalReasoner
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)

class ClipMatcher(SetCriterion):
    def __init__(self, num_classes,
                        matcher,
                        weight_dict,
                        losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__(num_classes, matcher, weight_dict, losses)
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_loss = True
        self.losses_dict = {}
        self._current_frame_idx = 0

    def initialize_for_single_clip(self, gt_instances: List[Instances], dataset_name=None):
        self.gt_instances = gt_instances
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}
        self.dataset_name = dataset_name

    def _step(self):
        self._current_frame_idx += 1

    def calc_loss_for_track_scores(self, track_instances: Instances):
        frame_id = self._current_frame_idx - 1
        gt_instances = self.gt_instances[frame_id]
        outputs = {
            'pred_logits': track_instances.track_scores[None],
        }
        device = track_instances.track_scores.device

        num_tracks = len(track_instances)
        src_idx = torch.arange(num_tracks, dtype=torch.long, device=device)
        tgt_idx = track_instances.matched_gt_idxes  # -1 for FP tracks and disappeared tracks

        track_losses = self.get_loss('labels',
                                     outputs=outputs,
                                     gt_instances=[gt_instances],
                                     indices=[(src_idx, tgt_idx)],
                                     num_boxes=1)
        self.losses_dict.update(
            {'frame_{}_track_{}'.format(frame_id, key): value for key, value in
             track_losses.items()})

    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

    def get_loss(self, loss, outputs, gt_instances, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'refers': self.loss_refers,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, gt_instances, indices, num_boxes, **kwargs)

    def loss_boxes(self, outputs, gt_instances: List[Instances], indices: List[tuple], num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        # We ignore the regression loss of the track-disappear slots.
        #TODO: Make this filter process more elegant.
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([gt_per_img.boxes[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)

        # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        target_obj_ids = torch.cat([gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0) # size(16)
        mask = (target_obj_ids != -1)

        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[mask]),
            box_ops.box_cxcywh_to_xyxy(target_boxes[mask])))

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_labels(self, outputs, gt_instances: List[Instances], indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # The matched gt for disappear track query is set -1.
        labels = []
        for gt_per_img, (_, J) in zip(gt_instances, indices):
            labels_per_img = torch.ones_like(J)
            # set labels of track-appear slots to 0.
            # labels_per_img = torch.full_like(J, self.num_classes)  # fixed
            if len(gt_per_img) > 0:
                labels_per_img[J != -1] = gt_per_img.labels[J[J != -1]]
            labels.append(labels_per_img)
        target_classes_o = torch.cat(labels)
        target_classes[idx] = target_classes_o
        if self.focal_loss:
            gt_labels_target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[:, :, :-1]  # no loss for the last (background) class
            gt_labels_target = gt_labels_target.to(src_logits)
            loss_ce = sigmoid_focal_loss(src_logits.flatten(1),
                                             gt_labels_target.flatten(1),
                                             alpha=0.25,
                                             gamma=2,
                                             num_boxes=num_boxes, mean_in_dim1=False)
            loss_ce = loss_ce.sum()
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this self.trainingone here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    def loss_refers(self,  outputs, gt_instances: List[Instances], indices: List[tuple], num_boxes):

        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_refs = outputs['pred_refers'][idx]
        target_refs = torch.cat([gt_per_img.is_ref[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)
        gt_labels_target = target_refs.view(src_refs.size()[0], src_refs.size()[1])

        if self.focal_loss:
            loss_ce = sigmoid_focal_loss(src_refs.flatten(1),
                                         gt_labels_target.flatten(1),
                                         alpha=0.25,
                                         gamma=2,
                                         num_boxes=num_boxes, mean_in_dim1=False)
            loss_ce = loss_ce.sum()
        else:
            loss_ce = F.cross_entropy(src_refs.transpose(1, 2), gt_labels_target, self.empty_weight)
        losses = {'loss_refer': loss_ce}

        return losses

    def match_for_single_frame(self, outputs: dict):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        gt_instances_i = self.gt_instances[self._current_frame_idx]  # gt instances of i-th image.
        track_instances: Instances = outputs_without_aux['track_instances']
        pred_logits_i = outputs_without_aux['pred_logits'][0]  # predicted logits of i-th image.
        pred_boxes_i = outputs_without_aux['pred_boxes'][0]  # predicted boxes of i-th image.
        pred_refers_i = outputs_without_aux['pred_refers'][0] # predicted refer scores of i-th image

        obj_idxes = gt_instances_i.obj_ids
        obj_idxes_list = obj_idxes.detach().cpu().numpy().tolist()
        obj_idx_to_gt_idx = {obj_idx: gt_idx for gt_idx, obj_idx in enumerate(obj_idxes_list)}
        outputs_i = {
            'pred_logits': pred_logits_i.unsqueeze(0), 
            'pred_boxes': pred_boxes_i.unsqueeze(0), 
            'pred_refers': pred_refers_i.unsqueeze(0), 
        }


        # step1. inherit and update the previous tracks.
        num_disappear_track = 0
        for j in range(len(track_instances)):
            obj_id = track_instances.obj_idxes[j].item()
            # set new target idx.
            if obj_id >= 0:
                if obj_id in obj_idx_to_gt_idx:
                    track_instances.matched_gt_idxes[j] = obj_idx_to_gt_idx[obj_id]
                else:
                    num_disappear_track += 1
                    track_instances.matched_gt_idxes[j] = -1  # track-disappear case.
            else:
                track_instances.matched_gt_idxes[j] = -1

        full_track_idxes = torch.arange(len(track_instances), dtype=torch.long).to(pred_logits_i.device)
        matched_track_idxes = (track_instances.obj_idxes >= 0)  # occu
        prev_matched_indices = torch.stack(
            [full_track_idxes[matched_track_idxes], track_instances.matched_gt_idxes[matched_track_idxes]], dim=1).to(
            pred_logits_i.device)

        # step2. select the unmatched slots.
        # note that the FP tracks whose obj_idxes are -2 will not be selected here.
        unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]

        # step3. select the untracked gt instances (new tracks).
        tgt_indexes = track_instances.matched_gt_idxes
        tgt_indexes = tgt_indexes[tgt_indexes != -1]

        tgt_state = torch.zeros(len(gt_instances_i)).to(pred_logits_i.device)
        tgt_state[tgt_indexes] = 1
        untracked_tgt_indexes = torch.arange(len(gt_instances_i)).to(pred_logits_i.device)[tgt_state == 0]
        # untracked_tgt_indexes = select_unmatched_indexes(tgt_indexes, len(gt_instances_i))
        untracked_gt_instances = gt_instances_i[untracked_tgt_indexes]

        def match_for_single_decoder_layer(unmatched_outputs, matcher):
            new_track_indices = matcher(unmatched_outputs,
                                             [untracked_gt_instances])  # list[tuple(src_idx, tgt_idx)]

            src_idx = new_track_indices[0][0]
            tgt_idx = new_track_indices[0][1]
            # concat src and tgt.
            new_matched_indices = torch.stack([unmatched_track_idxes[src_idx], untracked_tgt_indexes[tgt_idx]],
                                              dim=1).to(pred_logits_i.device)
            return new_matched_indices

        # step4. do matching between the unmatched slots and GTs.
        unmatched_outputs = {
            'pred_logits': track_instances.cache_pred_logits[unmatched_track_idxes].unsqueeze(0),
            'pred_boxes': track_instances.cache_pred_boxes[unmatched_track_idxes].unsqueeze(0),
            'pred_refers': track_instances.cache_pred_refers[unmatched_track_idxes].unsqueeze(0)
        }
        new_matched_indices = match_for_single_decoder_layer(unmatched_outputs, self.matcher)

        # step5. update obj_idxes according to the new matching result.
        track_instances.obj_idxes[new_matched_indices[:, 0]] = gt_instances_i.obj_ids[new_matched_indices[:, 1]].long()
        track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]

        # step6. calculate iou.
        active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.matched_gt_idxes >= 0)
        active_track_boxes = track_instances.cache_pred_boxes[active_idxes]
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_i.boxes[track_instances.matched_gt_idxes[active_idxes]]
            active_track_boxes = box_ops.box_cxcywh_to_xyxy(active_track_boxes)
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            track_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))

        # step7. merge the unmatched pairs and the matched pairs.
        matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)

        # step8. calculate losses.
        self.num_samples += len(gt_instances_i) + num_disappear_track
        self.sample_device = pred_logits_i.device
        # if is_crowdhuman:
        #     losses = ['labels', 'boxes']
        # else:
        losses = ['labels', 'boxes', 'refers']
        for loss in losses:
            new_track_loss = self.get_loss(loss,
                                           outputs=outputs_i,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices[:, 0], matched_indices[:, 1])],
                                           num_boxes=1)
            self.losses_dict.update(
                {'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in new_track_loss.items()})

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                unmatched_outputs_layer = {
                    'pred_logits': aux_outputs['pred_logits'][0, unmatched_track_idxes].unsqueeze(0),
                    'pred_boxes': aux_outputs['pred_boxes'][0, unmatched_track_idxes].unsqueeze(0),
                }
                new_matched_indices_layer = match_for_single_decoder_layer(unmatched_outputs_layer, self.matcher)
                matched_indices_layer = torch.cat([new_matched_indices_layer, prev_matched_indices], dim=0)
                # losses = ['labels', 'boxes', 'scores']
                for loss in losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss,
                                           aux_outputs,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices_layer[:, 0], matched_indices_layer[:, 1])],
                                           num_boxes=1, )
                    self.losses_dict.update(
                        {'frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in
                         l_dict.items()})
        self._step()
        return track_instances


    def loss_hist_reasoning(self,track_instances:Instances):
        
        frame_id = self._current_frame_idx-1
        gt_instances_i = self.gt_instances[frame_id]
        pred_logits_i = track_instances.cache_pred_logits  
        pred_boxes_i = track_instances.cache_pred_boxes  
        pred_refers_i = track_instances.cache_pred_refers 
        outputs_i = {
            'pred_logits': pred_logits_i.unsqueeze(0), 
            'pred_boxes': pred_boxes_i.unsqueeze(0), 
            'pred_refers': pred_refers_i.unsqueeze(0), 
        }

        num_tracks = len(track_instances)
        device = track_instances.track_scores.device
        src_idx = torch.arange(num_tracks,dtype=torch.long,device=device) 
        tgt_idx = track_instances.matched_gt_idxes 

        losses = ['labels', 'boxes', 'refers']
        for loss in losses:
            new_track_loss = self.get_loss(loss,
                                           outputs=outputs_i,
                                           gt_instances=[gt_instances_i],
                                           indices=[(src_idx, tgt_idx)],
                                           num_boxes=1)
            self.losses_dict.update(
                {'frame_{}_temporal_{}'.format(frame_id, key): value for key, value in new_track_loss.items()})
        return
    
    def calc_iou(self,track_instances:Instances):
        frame_id = self._current_frame_idx-1
        gt_instances_i = self.gt_instances[frame_id]
        active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.matched_gt_idxes >= 0)
        active_track_boxes = track_instances.cache_pred_boxes[active_idxes]
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_i.boxes[track_instances.matched_gt_idxes[active_idxes]]
            active_track_boxes = box_ops.box_cxcywh_to_xyxy(active_track_boxes)
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            track_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))
        return track_instances

    def forward(self, outputs, input_data: dict):
        # losses of each frame are calculated during the model's forwarding and are outputted by the model as outputs['losses_dict].
        losses = outputs.pop("losses_dict")
        num_samples = self.get_num_boxes(self.num_samples)
        for loss_name, loss in losses.items():
            losses[loss_name] /= num_samples
        return losses


class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.6, filter_score_thresh=0.5, miss_tolerance=10):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        device = track_instances.obj_idxes.device

        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        new_obj = (track_instances.obj_idxes == -1) & (track_instances.scores >= self.score_thresh)
        disappeared_obj = (track_instances.obj_idxes >= 0) & (track_instances.scores < self.filter_score_thresh)
        num_new_objs = new_obj.sum().item()

        track_instances.obj_idxes[new_obj] = self.max_obj_id + torch.arange(num_new_objs, device=device)
        self.max_obj_id += num_new_objs

        track_instances.disappear_time[disappeared_obj] += 1
        to_del = disappeared_obj & (track_instances.disappear_time >= self.miss_tolerance)
        track_instances.obj_idxes[to_del] = -1


class TrackerPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, track_instances: Instances, target_size) -> Instances:
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits = track_instances.pred_logits
        out_bbox = track_instances.pred_boxes
        out_refers = track_instances.pred_refers

        # prob = out_logits.sigmoid()
        # prob = out_logits[...,:1].sigmoid()
        scores = out_logits[..., 0].sigmoid()
        # scores, labels = prob.max(-1)
        refers = out_refers.sigmoid().max(-1)[0]

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_size
        scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
        boxes = boxes * scale_fct[None, :]

        track_instances.boxes = boxes
        track_instances.scores = scores
        track_instances.labels = torch.full_like(scores, 0)
        track_instances.refers = refers
        track_instances.remove('pred_logits')
        track_instances.remove('pred_boxes')
        track_instances.remove('pred_refers')
        return track_instances


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransRMOT(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, criterion, track_embed,
                 aux_loss=True, with_box_refine=False, two_stage=False, memory_bank=None, use_checkpoint=False,tracking=False,hist_len=4):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.tracking = tracking
        self.num_queries = num_queries
        self.track_embed = track_embed # QIM
        self.transformer = transformer # deformable detr plus
        hidden_dim = transformer.d_model 
        self.num_classes = num_classes 
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.refer_embed = nn.Linear(hidden_dim, 1) # this is referring branch
        self.num_feature_levels = num_feature_levels
        self.use_checkpoint = use_checkpoint
        
        # Temporal Enhancement Module
        self.STReasoner = SpatialTemporalReasoner(hist_len=hist_len)
        self.hist_len = hist_len

        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2) #(300,256)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss # true
        self.with_box_refine = with_box_refine # true
        self.two_stage = two_stage #False

        self.init_params_and_layers(hidden_dim)
        # language encoder
        # self.tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder_type)
        # self.text_encoder = RobertaModel.from_pretrained(text_encoder_type)
        # self.text_encoder.pooler = None  # this pooler is never used, this is a hack to avoid DDP problems...

        self.tokenizer = RobertaTokenizerFast.from_pretrained('/data_2/zyn/Data4RMOT/FairMOT/src/roberta_base/',
                                                              local_files_only=True)
        self.text_encoder = RobertaModel.from_pretrained('/data_2/zyn/Data4RMOT/FairMOT/src/roberta_base/',
                                                         local_files_only=True)

        freeze_text_encoder = True
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        self.txt_proj = FeatureResizer(
            input_feat_size=self.text_encoder.config.hidden_size,
            output_feat_size=hidden_dim,
            dropout=True,
        )

        self.fusion_module = VisionLanguageFusionModule(d_model=hidden_dim, nhead=8)
        self.text_pos = PositionEmbeddingSine1D(hidden_dim, normalize=True)

        prior_prob = 0.01 # positive prior probability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        self.refer_embed.bias.data = torch.ones(1) * bias_value

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine: # True
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
            self.refer_embed = _get_clones(self.refer_embed, num_pred)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
            self.refer_embed = nn.ModuleList([self.refer_embed for _ in range(num_pred)])

        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        self.post_process = TrackerPostProcess()
        self.track_base = RuntimeTrackerBase()
        self.criterion = criterion
        self.memory_bank = memory_bank
        self.mem_bank_len = 0 if memory_bank is None else memory_bank.max_his_length

    def _generate_empty_tracks(self):
        track_instances = Instances((1, 1))
        num_queries, dim = self.query_embed.weight.shape  # (300, 256)
        device = self.query_embed.weight.device
        query_embeds = self.query_embed.weight
        reference_points = self.transformer.reference_points(self.query_embed.weight[:, :dim // 2])

        """Detection queries 3,3"""
        track_instances.ref_pts = reference_points.clone() # first hidden_dim for init point prediction
        track_instances.query_pos = query_embeds.clone()

        
        """Tracking information 3,4"""
        track_instances.obj_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device) # id for the tracks
        track_instances.matched_gt_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)# matched gt indexes, for loss computation
        track_instances.disappear_time = torch.zeros((len(track_instances), ), dtype=torch.long, device=device)# life cycle management
        track_instances.track_query_mask = torch.zeros(
            (len(track_instances), ), dtype=torch.bool, device=device)
        """Current frame information """   
        track_instances.iou = torch.ones((len(track_instances),), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((len(track_instances), 4), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros((len(track_instances), self.num_classes), dtype=torch.float, device=device)
        track_instances.pred_refers = torch.zeros((len(track_instances), 1), dtype=torch.float, device=device)

        """Cache for current frame information, loading temporary data for qim"""
        track_instances.cache_scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.cache_pred_boxes = torch.zeros((len(track_instances), 4), dtype=torch.float, device=device)
        track_instances.cache_pred_logits = torch.zeros((len(track_instances), self.num_classes), dtype=torch.float, device=device)
        track_instances.cache_pred_refers = torch.zeros((len(track_instances), 1), dtype=torch.float, device=device)

        # embedding 
        track_instances.hist_embeds = torch.zeros(
            (len(track_instances),self.hist_len,dim // 2),dtype=torch.float32,device=device)
        # padding mask, follow MultiHeadAttention, 1 indicates padded
        track_instances.hist_padding_masks = torch.ones(
            (len(track_instances), self.hist_len), dtype=torch.bool, device=device)
       
        track_instances.output_embedding = torch.zeros((num_queries, dim //2 ), device=device)
        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros((len(track_instances), mem_bank_len, dim), dtype=torch.float32, device=device)
        track_instances.mem_padding_mask = torch.ones((len(track_instances), mem_bank_len), dtype=torch.bool, device=device)
        track_instances.save_period = torch.zeros((len(track_instances), ), dtype=torch.float32, device=device)

        return track_instances.to(self.query_embed.weight.device)

    def clear(self):
        self.track_base.clear()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_refer):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_refers': c, }
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_refer[:-1])]

    def forward_text_simple(self, text_querires, device):
        tokenized_queries = text_querires[0].split(' ')
        indices = [self.glove_vocab[k] for k in tokenized_queries if k in self.glove_vocab.keys()]
        indices = torch.tensor(np.asarray(indices), dtype=torch.int64).to(device)
        text_features = self.text_encoder(indices).unsqueeze(0)
        text_pad_mask = torch.zeros(text_features.size()[:2]).type(torch.BoolTensor).to(device)

        text_features = self.txt_proj(text_features)
        return text_features, text_pad_mask, text_features

    def forward_text(self, text_queries, device):
        tokenized_queries = self.tokenizer.batch_encode_plus(text_queries, padding="longest", return_tensors='pt').to(device)
        # with torch.inference_mode(mode=self.freeze_text_encoder):
        encoded_text = self.text_encoder(**tokenized_queries)
        # Transpose memory because pytorch's attention expects sequence first
        text_features = encoded_text.last_hidden_state.clone()
        text_features = self.txt_proj(text_features)  # change text embeddings dim to model dim
        # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
        text_pad_mask = tokenized_queries.attention_mask.ne(1).bool()  # [B, S]

        text_sentence_features = text_features
        return text_features, text_pad_mask, text_sentence_features

    def _forward_single_image(self, samples, track_instances: Instances, sentences):
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None

        # Extract linguistic features
        text_word_features, text_word_mask, text_sentence_features = self.forward_text(sentences, src.device)
        text_word_features = text_word_features.flatten(0, 1).unsqueeze(0)
        text_word_mask = text_word_mask.flatten(0, 1).unsqueeze(0)
        text_pos = self.text_pos(NestedTensor(text_word_features, text_word_mask)).permute(2, 0, 1)
        text_word_features = text_word_features.permute(1, 0, 2)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            src_proj_l = self.input_proj[l](src) # 
            n, c, h, w = src_proj_l.shape

            # vision and language fusion
            src_proj_l = rearrange(src_proj_l, 'b c h w -> (h w) b c')
            src_proj_l = self.fusion_module(tgt=src_proj_l,
                                            memory=text_word_features,
                                            memory_key_padding_mask=text_word_mask,
                                            pos=text_pos,
                                            query_pos=None
                                            )
            src_proj_l = rearrange(src_proj_l, '(h w) b c -> b c h w', h=h, w=w)
            srcs.append(src_proj_l)
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs): # add smallest scale
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                n, c, h, w = src.shape

                src = rearrange(src, 'b c h w -> (h w) b c')
                src = self.fusion_module(tgt=src,
                                         memory=text_word_features,
                                         memory_key_padding_mask=text_word_mask,
                                         pos=text_pos,
                                         query_pos=None
                                         )
                src = rearrange(src, '(h w) b c -> b c h w', h=h, w=w)

                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.transformer(srcs, masks, pos, track_instances.query_pos, text_sentence_features, ref_pts=track_instances.ref_pts)
        outputs_classes = []
        outputs_coords = []
        outputs_refers = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_refer = self.refer_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_refers.append(outputs_refer)

        outputs_class = torch.stack(outputs_classes) 
        outputs_coord = torch.stack(outputs_coords) 
        outputs_refer = torch.stack(outputs_refers)
        ref_pts_all = torch.cat([init_reference[None], inter_references[:, :, :, :2]], dim=0)
        last_query_feats = hs[-1]
        # last_query_embeds = track_instances.query_embeds.clone()
        out = {
            'pred_logits': outputs_class[-1], 
            'pred_boxes': outputs_coord[-1], 
            'ref_pts': ref_pts_all[5], 
            'pred_refers': outputs_refer[-1],
            'query_feats':last_query_feats,
            }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_refer)
        out['text_word_mask'] = text_word_mask 
        out['text_pos'] = text_pos 
        out['text_word_features'] = text_word_features 
        return out


    def _post_process_single_image(self, frame_res, track_instances, is_last):
        frame_res['track_instances'] = track_instances
        text_result = {
            'text_word_mask':frame_res.pop('text_word_mask'),
            'text_pos':frame_res.pop('text_pos'),
            'text_word_features':frame_res.pop('text_word_features')
        }
        if self.training:
            # the track id will be assigned by the mather.
            # Loss computation for the detection
            track_instances = self.criterion.match_for_single_frame(frame_res)
        
        if self.memory_bank is not None: #False
            track_instances = self.memory_bank(track_instances)
            if self.training:
                self.criterion.calc_loss_for_track_scores(track_instances)
        
           
        if self.training:
            track_instances = self.STReasoner(track_instances,text_result,training=True)
            track_instances = self.criterion.calc_iou(track_instances)
            self.criterion.loss_hist_reasoning(track_instances)
            track_instances = self.frame_summarization(track_instances, tracking=False)
            frame_res['pred_logits'] = track_instances.pred_logits
            frame_res['pred_boxes'] = track_instances.pred_boxes
            frame_res['pred_refers'] = track_instances.pred_refers
        else:
            track_instances = self.STReasoner(track_instances,text_result,training=False)
            track_instances = self.frame_summarization(track_instances,tracking=True)
            self.track_base.update(track_instances)
        
        tmp = {}
        tmp['init_track_instances'] = self._generate_empty_tracks()
        tmp['track_instances'] = track_instances
        if not is_last:
            out_track_instances = self.track_embed(tmp)
            frame_res['track_instances'] = out_track_instances
        else:
            frame_res['track_instances'] = None
        return frame_res


    @torch.no_grad()
    def inference_single_image(self, img, sentence, ori_img_size, track_instances=None):
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        if track_instances is None:
            track_instances = self._generate_empty_tracks()
        res = self._forward_single_image(img, track_instances=track_instances, sentences=sentence)
        track_instances = self.load_detection_output_into_cache(track_instances,res)
        res = self._post_process_single_image(res, track_instances, False)

        track_instances = res['track_instances']
        track_instances = self.post_process(track_instances, ori_img_size)
        ret = {'track_instances': track_instances}
        if 'ref_pts' in res:
            ref_pts = res['ref_pts']
            img_h, img_w = ori_img_size
            scale_fct = torch.Tensor([img_w, img_h]).to(ref_pts)
            ref_pts = ref_pts * scale_fct[None]
            ret['ref_pts'] = ref_pts
        return ret

    def init_params_and_layers(self,hidden_dim):
        """Generate the instances for tracking, especially the object queries
        """
        # query initialization for detection
        # reference points, mapping fourier encoding to embed_dims
        if self.tracking:
            self.query_feat_embedding = nn.Embedding(self.num_queries,hidden_dim)
            nn.init.zeros_(self.query_feat_embedding.weight)
    
    def load_detection_output_into_cache(self,track_instances:Instances,out):
        """ Load output of the detection head into the track_instances cache (inplace)
        """
        query_feats = out.pop('query_feats')
        # query_embeds = out.pop('query_embeds')
        with torch.no_grad():
            if self.training:# True
                track_scores = out['pred_logits'][0, :].sigmoid().max(dim=-1).values
            else:
                track_scores = out['pred_logits'][0, :, 0].sigmoid()
        track_instances.cache_scores = track_scores.clone()
        track_instances.cache_pred_logits = out['pred_logits'][0].clone() 
        track_instances.output_embedding = query_feats[0].clone() 
        track_instances.cache_pred_boxes = out['pred_boxes'][0].clone() 
        track_instances.cache_pred_refers = out['pred_refers'][0].clone()
        return track_instances


    def frame_summarization(self, track_instances, tracking):
        """ Load the results after spatial-temporal reasoning into track instances
        """
        track_instances.pred_logits= track_instances.cache_pred_logits 
        track_instances.scores = track_instances.cache_scores
        track_instances.pred_refers = track_instances.cache_pred_refers 
        track_instances.pred_boxes = track_instances.cache_pred_boxes
        return track_instances


    # @autocast()
    def forward(self, data: dict):
        # data_dict = copy.deepcopy(data)
        if self.training:
            self.criterion.initialize_for_single_clip(data['gt_instances'], data['dataset_name'])
        frames = data['imgs']  # list of Tensor.
        sentences = data['sentences']
        self.sentences_len = len(sentences)
        outputs = {
            'pred_logits': [],
            'pred_boxes': [],
            'pred_refers': [],
        }

        track_instances = self._generate_empty_tracks()
        keys = list(track_instances._fields.keys())
        for frame_index, frame in enumerate(frames):
            frame.requires_grad = False
            is_last = frame_index == len(frames) - 1
            if self.use_checkpoint and frame_index < len(frames) - 1:
                def fn(frame, *args):
                    frame = nested_tensor_from_tensor_list([frame])
                    # frame.requires_grad = False
                    tmp = Instances((1, 1), **dict(zip(keys, args)))
                    frame_res = self._forward_single_image(frame, tmp, sentences)
                    return (
                        frame_res['pred_logits'],
                        frame_res['pred_boxes'],
                        frame_res['pred_refers'],
                        frame_res['ref_pts'],
                        frame_res['query_feats'],
                        frame_res['text_word_mask'],
                        frame_res['text_pos'],
                        frame_res['text_word_features'],
                        *[aux['pred_logits'] for aux in frame_res['aux_outputs']],
                        *[aux['pred_boxes'] for aux in frame_res['aux_outputs']],
                        *[aux['pred_refers'] for aux in frame_res['aux_outputs']],
                    )

                args = [frame] + [track_instances.get(k) for k in keys]
                params = tuple((p for p in self.parameters() if p.requires_grad))
                tmp = checkpoint.CheckpointFunction.apply(fn, len(args), *args, *params)
                frame_res = {
                    'pred_logits': tmp[0],
                    'pred_boxes': tmp[1],
                    'pred_refers': tmp[2],
                    'ref_pts': tmp[3],
                    'query_feats': tmp[4],
                    'text_word_mask': tmp[5] ,
                    'text_pos': tmp[6],
                    'text_word_features': tmp[7],
                    'aux_outputs': [{
                        'pred_logits': tmp[8 + i],
                        'pred_boxes': tmp[8 + 5 + i],
                        'pred_refers': tmp[8 + 5 + 5 + i]
                    } for i in range(5)],
                }
            else:
                frame = nested_tensor_from_tensor_list([frame])
                # frame.requires_grad = False
                frame_res = self._forward_single_image(frame, track_instances, sentences)


            track_instances = self.load_detection_output_into_cache(track_instances, frame_res)
            frame_res = self._post_process_single_image(frame_res, track_instances, is_last)

            track_instances = frame_res['track_instances']
            outputs['pred_logits'].append(frame_res['pred_logits'])
            outputs['pred_boxes'].append(frame_res['pred_boxes'])
            outputs['pred_refers'].append(frame_res['pred_refers'])

        if not self.training:
            outputs['track_instances'] = track_instances
        else:
            outputs['losses_dict'] = self.criterion.losses_dict
        return outputs


def build(args):
    dataset_to_num_classes = {
        'e2e_rmot': 1,
        'e2e_dance': 1,
        'e2e_t2t': 1,
    }
    assert args.dataset_file in dataset_to_num_classes
    num_classes = dataset_to_num_classes[args.dataset_file]
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    d_model = transformer.d_model
    hidden_dim = args.dim_feedforward 
    query_interaction_layer = build_query_interaction_layer(args, args.query_interaction_layer, d_model, hidden_dim, d_model*2)

    img_matcher = build_matcher(args)
    print(args)
    num_frames_per_batch = max(args.sampler_lengths)# 2
    weight_dict = {}
    for i in range(num_frames_per_batch):
        weight_dict.update({"frame_{}_loss_ce".format(i): args.cls_loss_coef,
                            'frame_{}_loss_bbox'.format(i): args.bbox_loss_coef,
                            'frame_{}_loss_giou'.format(i): args.giou_loss_coef,
                            'frame_{}_loss_refer'.format(i): args.refer_loss_coef,
                            "frame_{}_temporal_loss_ce".format(i): args.cls_loss_coef,
                            'frame_{}_temporal_loss_bbox'.format(i): args.bbox_loss_coef,
                            'frame_{}_temporal_loss_giou'.format(i): args.giou_loss_coef,
                            'frame_{}_temporal_loss_refer'.format(i): args.refer_loss_coef,
                            })

    # TODO this is a hack
    if args.aux_loss:
        for i in range(num_frames_per_batch):
            for j in range(args.dec_layers - 1):
                weight_dict.update({"frame_{}_aux{}_loss_ce".format(i, j): args.cls_loss_coef,
                                    'frame_{}_aux{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_aux{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    'frame_{}_aux{}_loss_refer'.format(i, j): args.refer_loss_coef,
                                    })
    if args.memory_bank_type is not None and len(args.memory_bank_type) > 0:
        memory_bank = build_memory_bank(args, d_model, hidden_dim, d_model * 2)
        for i in range(num_frames_per_batch):
            weight_dict.update({"frame_{}_track_loss_ce".format(i): args.cls_loss_coef})
    else:
        memory_bank = None
    losses = ['labels', 'boxes', 'refers']
    criterion = ClipMatcher(num_classes, matcher=img_matcher, weight_dict=weight_dict, losses=losses)
    criterion.to(device)
    postprocessors = {}
    model = TransRMOT(
        backbone,
        transformer,
        track_embed=query_interaction_layer,
        num_feature_levels=args.num_feature_levels,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        criterion=criterion,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        memory_bank=memory_bank,
        use_checkpoint=args.use_checkpoint,
        tracking = args.tracking,
        hist_len = args.hist_len
    )
    return model, criterion, postprocessors
