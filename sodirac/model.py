#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/17/23 2:58 PM
# @Author  : Chang Xu
# @File    : model.py
# @Email   : changxu@nus.edu.sg
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
# from easydl import aToBSheduler
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Function
from torch_geometric.nn import (BatchNorm, GCNConv, SAGEConv, GATConv, 
                    Sequential, TAGConv, GraphConv, 
                    GatedGraphConv, ResGatedGraphConv, TransformerConv, 
                    ARMAConv, SGConv, MFConv, RGCNConv, FeaStConv, LEConv, 
                    ClusterGCNConv, GraphNorm, LayerNorm, PairNorm, InstanceNorm,
                    GraphSizeNorm, MessageNorm, VGAE) 
from typing import Callable, Iterable, Union, Tuple, Optional
import collections
import logging
from itertools import combinations
logger = logging.getLogger(__name__)


OPTIMIZERS = {
            "GCN": GCNConv,
            "SAGE": SAGEConv,
            "GAT": GATConv,
            "TAG": TAGConv,
            "Graph": GraphConv,
            "GatedGraph": GatedGraphConv,
            "ResGatedGraph": ResGatedGraphConv,
            "Transformer": TransformerConv, 
            "ARMA": ARMAConv,
            "SG": SGConv,
            "MF": MFConv, 
            "RGCN": RGCNConv, 
            "FeaSt": FeaStConv,
            "LE": LEConv,
            "ClusterGCN": ClusterGCNConv,
            }


# def aToBScheduler(step, min_val=0.0, max_val=1.0, gamma=10, max_iter=10000):
#     """
#     A custom scheduler that linearly scales from `min_val` to `max_val` over `max_iter` steps.
#     It adjusts the rate of scaling based on the `gamma` parameter, which introduces a non-linear behavior.
    
#     Parameters:
#     - step: Current training step.
#     - min_val: Minimum value of the coefficient.
#     - max_val: Maximum value of the coefficient.
#     - gamma: A scaling factor for controlling the speed of change (higher gamma -> faster change).
#     - max_iter: Maximum number of steps.
    
#     Returns:
#     - The adjusted coefficient value at the current step.
#     """
#     # Calculate the progress (0 to 1) based on the current step
#     progress = min(step / max_iter, 1.0)
    
#     # Apply a non-linear scaling controlled by gamma
#     # This formula can be adjusted based on the desired behavior.
#     scale_factor = (max_val - min_val) * (1.0 - math.exp(-gamma * progress)) + min_val
    
#     return float(scale_factor)


def aToBSheduler(step, A, B, gamma=10, max_iter=10000):
    '''
    change gradually from A to B, according to the formula (from <Importance Weighted Adversarial Nets for Partial Domain Adaptation>)
    A + (2.0 / (1 + exp(- gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)

    =code to see how it changes(almost reaches B at %40 * max_iter under default arg)::

        from matplotlib import pyplot as plt

        ys = [aToBSheduler(x, 1, 3) for x in range(10000)]
        xs = [x for x in range(10000)]

        plt.plot(xs, ys)
        plt.show()

    '''
    ans = A + (2.0 / (1 + np.exp(- gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)
    return float(np.copy(ans))


class NormedLinear(nn.Module):

    def __init__(
        self, 
        in_features, 
        out_features,
        ):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(
        self, 
        x
        ):
        out = 5 * F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return  out


class GradientReverseLayer(Function):
    """Layer that reverses and scales gradients before
    passing them up to earlier ops in the computation graph
    during backpropogation.
    """
    @staticmethod
    def forward(ctx, coeff, input):
        """
        Perform a no-op forward pass that stores a weight for later
        gradient scaling during backprop.

        Parameters
        ----------
        x : torch.FloatTensor
            [Batch, Features]
        weight : float
            weight for scaling gradients during backpropogation.
            stored in the "context" ctx variable.

        Notes
        -----
        We subclass `Function` and use only @staticmethod as specified
        in the newstyle pytorch autograd functions.
        https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function

        We define a "context" ctx of the class that will hold any values
        passed during forward for use in the backward pass.

        `x.view_as(x)` and `*1` are necessary so that `GradReverse`
        is actually called
        `torch.autograd` tries to optimize backprop and
        excludes no-ops, so we have to trick it :)
        """
        # store the weight we'll use in backward in the context
        ctx.coeff = coeff
        return input

    @staticmethod
    def backward(ctx, grad_outputs):
        """Return gradients

        Returns
        -------
        rev_grad : torch.FloatTensor
            reversed gradients scaled by `weight` passed in `.forward()`
        None : None
            a dummy "gradient" required since we passed a weight float
            in `.forward()`.
        """
        # here scale the gradient and multiply by -1
        # to reverse the gradients
        coeff = ctx.coeff
        return None, -coeff * grad_outputs

class GradientReverseModule(nn.Module):
    def __init__(
        self, 
        scheduler,
        ):
        super(GradientReverseModule, self).__init__()
        self.scheduler = scheduler
        self.global_step = 0.0
        self.coeff = 0.0
        self.grl = GradientReverseLayer.apply

    def forward(
        self, 
        x,
        ):
        # Calculate the coefficient based on the scheduler
        self.coeff = self.scheduler(self.global_step)
        self.global_step += 1.0
        return self.grl(self.coeff, x)


class integrate_model(nn.Module):
    def __init__(
        self, 
        n_inputs_list,
        n_domains,
        n_hiddens: int = 128,
        n_outputs: int = 64,
        opt_GNN: str = "GCN"
        ):
        super(integrate_model, self).__init__()
        self.n_inputs_list = n_inputs_list
        self.n_hiddens = n_hiddens
        self.n_outputs = n_outputs
        opt_GNN_model = OPTIMIZERS[opt_GNN]
        ######## private encoders and decoders
        encoders = []
        decoders = []
        for i in range(len(n_inputs_list)):
            encoder = Sequential('x, edge_index, batch', [
                (opt_GNN_model(n_inputs_list[i], n_hiddens), 'x, edge_index-> x1'), 
                (nn.LayerNorm(n_hiddens), 'x1 -> x1'),
                nn.Tanh(), 
                nn.Dropout(p=0.1),
                (opt_GNN_model(n_hiddens, n_outputs), 'x1, edge_index -> x2'),
                ])
            decoder = Sequential('x, edge_index', [
                (opt_GNN_model(n_outputs, n_inputs_list[i]), 'x, edge_index -> x1'),
                ])
            encoders.append(encoder)
            decoders.append(decoder)
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.combine_encoder = Sequential('x, edge_index',[
                        (opt_GNN_model(len(n_inputs_list) * n_outputs, len(n_inputs_list) * n_outputs // 3), 'x, edge_index -> x1'),
                        (nn.LayerNorm(len(n_inputs_list) * n_outputs // 3), 'x1 -> x1'),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.1),
                        (opt_GNN_model(len(n_inputs_list) * n_outputs // 3, n_outputs), 'x1, edge_index -> x2'),
                        ])
        self.clf_domain = NormedLinear(n_outputs, n_domains)
        self.max_iter = 10000.0
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1, gamma = 10, max_iter = self.max_iter))  

    def forward(
        self, 
        x_list,
        batch_list,
        edge_index,
        reverse: bool = True,
        ):
        feats = []
        domain_preds = []
        recon_feats = []
        for i in range(len(self.n_inputs_list)):
            feat = self.encoders[i](x_list[i], edge_index, batch_list[i])
            feats.append(feat)
            recon_feat = self.decoders[i](feat, edge_index)
            recon_feats.append(recon_feat)
            if reverse:
                feat_re = self.grl(feat)
                domain_pred = self.clf_domain(feat_re)
            else:
                domain_pred = self.clf_domain(feat)
            domain_preds.append(domain_pred)
        combine_feats = torch.cat(feats, dim=1)
        combine_recon = self.combine_encoder(combine_feats, edge_index)
        return feats, domain_preds, recon_feats, combine_recon


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
    """
    def __init__(
        self, 
        n_inputs, 
        n_labels, 
        n_hiddens: int = 64,
        n_outputs: int = 32,
        s = 64.0, 
        m = 0.20, 
        easy_margin = False, 
        ):
        super(ArcMarginProduct, self).__init__()
        self.n_inputs = n_inputs
        self.n_labels = n_labels
        self.s = s
        self.m = m
        self.linear1 = SAGEConv(n_inputs, n_outputs)
        self.relu = nn.ReLU()
        self.weight = nn.Parameter(torch.FloatTensor(n_labels, n_outputs))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(
        self, 
        x,
        edge_index,
        label,
        ):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        x = self.linear1(x, edge_index)
        feat = x
        x  = self.relu(x)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=self.weight.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # ------------- torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return feat, output
    
    def predict(
        self, 
        x,
        edge_index,
        ):
        x = self.linear1(x, edge_index)
        feat = x
        x  = self.relu(x)
        return feat, F.linear(F.normalize(x), F.normalize(self.weight))


class MovingAverage(nn.Module):
    def __init__(
        self, 
        size: Tuple[int, ...], 
        buffer_size: int = 128, 
        init_value: float = 0
    ):
        super().__init__()

        self.register_buffer(
            "buffer", torch.full((buffer_size,) + size, fill_value=init_value)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.buffer = torch.cat([self.buffer[1:], x[None]])

        return self.buffer.mean(dim=0)


class ExponentialMovingAverage(nn.Module):
    def __init__(
        self, size: Tuple[int, ...], momentum: float = 0.999, init_value: float = 0
    ):
        super().__init__()

        self.momentum = momentum
        self.register_buffer("avg", torch.full(size, fill_value=init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.avg += (self.avg - x) * (self.momentum - 1)

        return self.avg


class annotate_model(nn.Module):
    def __init__(
        self, 
        n_inputs,
        n_labels,
        n_domains,
        n_hiddens: int = 128,
        n_outputs: int = 64,
        opt_GNN: str = "GCN",
        s: int = 64,
        m: float = 0.1,
        easy_margin: bool = False,
        ):
        super(annotate_model, self).__init__()
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.n_outputs = n_outputs
        self.n_domains = n_domains
        opt_GNN_model = OPTIMIZERS[opt_GNN]
        ####### private encoders and decoders
        self.super_encoder = ArcMarginProduct(
                                n_inputs = n_inputs,
                                n_labels = n_labels,
                                n_hiddens = n_hiddens,
                                n_outputs = n_outputs,
                                s = s, 
                                m = m, 
                                easy_margin = easy_margin
                                )
        # self.super_encoder = NormedLinear(x_dim, num_cls)
        self.encoder = Sequential('x, edge_index', [
                                (opt_GNN_model(n_inputs, n_hiddens), 'x, edge_index -> x1'),
                                (nn.LayerNorm(n_hiddens), 'x1 -> x1'),
                                nn.ReLU(),
                                nn.Dropout(p=0.1),
                                (opt_GNN_model(n_hiddens, n_outputs), 'x1, edge_index -> x2' ),
                                ])
        self.decoder = Sequential('x, edge_index', [
                                (opt_GNN_model(n_outputs, n_inputs), 'x, edge_index -> x1'),
                                ])
        ######### share space advertised network (discriminator) 
        self.clf_domain = NormedLinear(n_outputs, n_domains)
        ######### share space advertised network (discriminator) 
        self.clf_label = NormedLinear(n_outputs, n_labels)
        self.max_iter = 10000.0
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=self.max_iter))  

    def forward(
        self, 
        x_list,
        edge_index_list,
        reverse: bool = True,
        ):
        feats = []
        label_preds = []
        domain_preds = []
        recon_feats = []
        for i in range(len(x_list)):
            feat = self.encoder(x_list[i], edge_index_list[i])
            feats.append(feat)
            recon_feat = self.decoder(feat, edge_index_list[i])
            recon_feats.append(recon_feat)
            label_pred = self.clf_label(feat)
            label_preds.append(label_pred)      
            if reverse:
                feat_re = self.grl(feat)
                domain_pred = self.clf_domain(feat_re)
            else:
                domain_pred = self.clf_domain(feat)
            domain_preds.append(domain_pred)
        return feats, domain_preds, recon_feats, label_preds











