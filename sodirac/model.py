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
from torch_geometric.nn import (BatchNorm, GCNConv, SAGEConv, GATConv, Sequential, TAGConv, GraphConv, 
                    GatedGraphConv, ResGatedGraphConv, TransformerConv, ARMAConv, SGConv, MFConv, FeaStConv,
                    ClusterGCNConv, GraphNorm, LayerNorm) 
from typing import Callable, Iterable, Union, Tuple, Optional, List
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
            "FeaSt": FeaStConv,
            "ClusterGCN": ClusterGCNConv,
            }


class integrate_model(nn.Module):
    def __init__(
        self, 
        n_inputs_list: List[int],
        n_domains: int,
        n_hiddens: int = 64,  # Increased default hidden size
        n_outputs: int = 16,   # Increased default output size
        opt_GNN: str = "GAT",   # Changed default to GAT for attention
        dropout_rate: float = 0.1,
        use_skip_connections: bool = True,
        use_attention: bool = True,
        n_attention_heads: int = 4,
        use_layer_scale: bool = False,
        layer_scale_init: float = 1e-2,
        use_stochastic_depth: bool = False,
        stochastic_depth_rate: float = 0.1,
        combine_method: str = 'concat',  # 'concat', 'sum', 'attention'
    ):
        """
        Advanced integrated model for multi-domain graph learning with modern techniques.
        
        Args:
            n_inputs_list: List of input dimensions for each modality
            n_domains: Number of domains for domain classification
            n_hiddens: Hidden layer dimension
            n_outputs: Output feature dimension
            opt_GNN: Type of GNN layer to use ('GCN', 'GAT', 'GraphSAGE', etc.)
            dropout_rate: Dropout rate for regularization
            use_skip_connections: Whether to use skip connections
            use_attention: Whether to use attention mechanisms
            n_attention_heads: Number of attention heads if using attention
            use_layer_scale: Whether to use layer scale (from ConvNeXt)
            layer_scale_init: Initial value for layer scale
            use_stochastic_depth: Whether to use stochastic depth regularization
            stochastic_depth_rate: Base rate for stochastic depth
        """
        super().__init__()
        
        # Input validation
        if not isinstance(n_inputs_list, list) or len(n_inputs_list) == 0:
            raise ValueError("n_inputs_list must be a non-empty list")
            
        self.n_inputs_list = n_inputs_list
        self.n_hiddens = n_hiddens
        self.n_outputs = n_outputs
        self.n_domains = n_domains
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.use_attention = use_attention
        self.use_layer_scale = use_layer_scale
        self.use_stochastic_depth = use_stochastic_depth
        self.layer_scale_init = layer_scale_init
        self.combine_method = combine_method
        self.stochastic_depth_rate = stochastic_depth_rate
        
        # Get GNN layer constructor
        self.gnn_layer = self._get_gnn_layer(opt_GNN, n_attention_heads)
        
        ######## Private Encoders ########
        self.encoders = nn.ModuleList()
        for i, n_inputs in enumerate(n_inputs_list):
            encoder = self._build_encoder(
                n_inputs, 
                n_hiddens, 
                n_outputs,
                f"encoder_{i}"
            )
            self.encoders.append(encoder)
        
        ######## Private Decoders ########
        self.decoders = nn.ModuleList()
        for i, n_inputs in enumerate(n_inputs_list):
            decoder = Sequential('x, edge_index', [
            (self.gnn_layer(n_outputs, n_inputs), 'x, edge_index -> x'),
            (nn.Dropout(dropout_rate), 'x -> x')
            ])
            self.decoders.append(decoder)

        # Feature combination attention if needed
        if combine_method == "attention":
            combined_dim = n_outputs
            self.combine_attention = nn.MultiheadAttention(
                combined_dim, n_attention_heads, dropout=dropout_rate, batch_first=True
            )
            self.combine_ln = LayerNorm(n_outputs)
        elif combine_method == "sum":
            combined_dim = n_outputs
        elif combine_method == "concat":
            combined_dim = len(n_inputs_list) * n_outputs
        
        self.combine_encoder = self._build_combiner(combined_dim, n_outputs)
        ######## Domain Classifier ########
        self.clf_domain = self._build_domain_classifier(n_outputs, n_hiddens, n_domains)
        
        ######## Gradient Reversal ########
        self.max_iter = 10000.0
        self.grl = GradientReverseModule(
            lambda step: aToBSheduler(
                step, 0.0, 1, gamma=10, max_iter=self.max_iter
            )
        )
    
        # Initialize weights
        self.apply(self._init_weights)

    def _get_gnn_layer(self, opt_GNN: str, n_heads: int = 4):
        """Get the appropriate GNN layer constructor"""
        if opt_GNN not in OPTIMIZERS:
            raise ValueError(f"Unsupported GNN type: {opt_GNN}")
            
        if opt_GNN == "GAT" and not self.use_attention:
            return OPTIMIZERS["GCN"]  # Fall back to GCN if attention disabled
            
        if opt_GNN == "GAT":
            # Wrap GAT to ensure output dimension matches expected n_outputs
            def wrapped_gat(in_dim, out_dim):
                return OPTIMIZERS[opt_GNN](
                    in_dim, 
                    out_dim,  # Split dimension across heads
                    heads = n_heads,
                    concat = False,
                )
            return wrapped_gat
            
        return OPTIMIZERS[opt_GNN]

    def _build_encoder(self, in_dim: int, hidden_dim: int, out_dim: int, name: str = ""):
        """Build an encoder block with modern techniques"""
          
        layers = [
            (nn.Linear(in_dim, hidden_dim), 'x -> x'),
            (LayerNorm(hidden_dim), 'x, batch -> x'),
            (GELU(), 'x -> x'),
            (nn.Dropout(self.dropout_rate), 'x -> x'),
            (nn.Linear(hidden_dim, out_dim), 'x  -> x'),
        ]
        
        if self.use_skip_connections:
            layers.append((ResidualConnection(out_dim), 'x, x -> x'))
        
        if self.use_layer_scale:
            layers.append((LayerScale(out_dim, init_value=self.layer_scale_init), 'x -> x'))
        
        return Sequential('x, batch', layers)

    def _build_combiner(self, in_dim: int, out_dim: int):
        """Build the feature combiner with modern techniques"""
        layers = [
            (self.gnn_layer(in_dim, in_dim), 'x, edge_index -> x'),
            (BatchNorm(in_dim), 'x -> x'),
            (GELU(), 'x -> x'),
            (nn.Dropout(self.dropout_rate), 'x -> x'),
            (self.gnn_layer(in_dim, out_dim), 'x, edge_index -> x'),
        ]
        
        if self.use_skip_connections:
            layers.insert(3, (ResidualConnection(in_dim), 'x, x -> x'))
        
        return Sequential('x, edge_index', layers)

    def _build_domain_classifier(self, in_dim: int, hidden_dim: int, out_dim: int):
        """Build the domain classifier with modern techniques"""
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            NormedLinear(hidden_dim, out_dim)
        )

    def _init_weights(self, module):
        """Initialize weights with modern schemes"""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

    def _combine_features(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """Enhanced feature combination with multiple methods"""
        if self.combine_method == "concat":
            return torch.cat(feats, dim=1)
        elif self.combine_method == "sum":
            return torch.stack(feats, dim=0).sum(dim=0)
        elif self.combine_method == "attention":
            # Reshape for attention: [batch_size, num_modalities, feature_dim]
            feat_stack = torch.stack(feats, dim=1)
            attn_out, _ = self.combine_attention(
                feat_stack, feat_stack, feat_stack
            )
            return self.combine_ln(attn_out.mean(dim=1))
        else:
            raise ValueError(f"Unknown combine_method: {self.combine_method}")

    def forward(
        self, 
        x_list: List[torch.Tensor],
        batch_list: List[torch.Tensor],
        edge_index: torch.Tensor,
        reverse: bool = True
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Forward pass with modern features.
        
        Args:
            x_list: List of input feature tensors
            # batch_list: List of batch index tensors
            edge_index: Graph connectivity
            reverse: Whether to use gradient reversal
            
        Returns:
            tuple: (features, domain_preds, recon_features, combined_features)
        """
        # Input validation
        self._validate_inputs(x_list, batch_list)
        
        feats = []
        domain_preds = []
        recon_feats = []
        
        # Process each modality with potential stochastic depth
        for i, (encoder, decoder) in enumerate(zip(self.encoders, self.decoders)):
            # Apply stochastic depth if enabled
            if self.use_stochastic_depth and self.training:
                keep_prob = 1.0 - self.stochastic_depth_rate * (i / len(self.encoders))
                if torch.rand(1).item() > keep_prob:
                    # Skip this encoder during training
                    dummy_feat = torch.zeros(
                        x_list[i].size(0), self.n_outputs, 
                        device=x_list[i].device
                    )
                    feats.append(dummy_feat)
                    recon_feats.append(x_list[i])  # Identity reconstruction
                    domain_preds.append(
                        torch.zeros(x_list[i].size(0), self.n_domains, 
                                  device=x_list[i].device)
                    )
                    continue
            
            # Encode features
            feat = encoder(x_list[i], batch_list[i])
            feats.append(feat)
            
            # Reconstruct features
            recon_feat = decoder(feat, edge_index)
            recon_feats.append(recon_feat)
            
            # Domain classification
            feat_re = self.grl(feat) if reverse else feat
            domain_pred = self.clf_domain(feat_re)
            domain_preds.append(domain_pred)

        # Combine features from all modalities
        combine_feats = self._combine_features(feats)
        combine_recon = self.combine_encoder(combine_feats, edge_index)
        
        return feats, domain_preds, recon_feats, combine_recon

    def _validate_inputs(self, x_list, batch_list):
        """Validate input tensors"""
        if len(x_list) != len(self.n_inputs_list):
            raise ValueError(
                f"Expected {len(self.n_inputs_list)} input tensors, got {len(x_list)}"
            )
        if len(batch_list) != len(self.n_inputs_list):
            raise ValueError(
                f"Expected {len(self.n_inputs_list)} batch tensors, got {len(batch_list)}"
            )
        for i, (x, n_inputs) in enumerate(zip(x_list, self.n_inputs_list)):
            if x.size(1) != n_inputs:
                raise ValueError(
                    f"Input {i} has dimension {x.size(1)}, expected {n_inputs}"
                )

                
class GELU(nn.Module):
    """Gaussian Error Linear Unit with optional approximate version"""
    def __init__(self, approximate: str = 'none'):
        super().__init__()
        self.approximate = approximate
        
    def forward(self, x):
        return F.gelu(x, approximate=self.approximate)


class ResidualConnection(nn.Module):
    """Enhanced residual connection with optional layer scaling"""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = LayerNorm(dim)
        
    def forward(self, x: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        return self.norm(x + res)


class LayerScale(nn.Module):
    """Layer scale from ConvNeXt paper"""
    def __init__(self, dim: int, init_value: float = 1e-2):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class GradientReverseModule(nn.Module):
    """Enhanced gradient reversal layer with scheduling"""
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler
        self.register_buffer('step', torch.zeros(1))
        
    def forward(self, x):
        lambd = self.scheduler(self.step.item())
        self.step += 1
        return GradientReverseFunction.apply(x, lambd)


class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


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
        out = 10 * F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return  out

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
        n_inputs: Union[int, List[int]],  # Can be single int or list of ints
        n_labels: int,
        n_domains: int,
        n_hiddens: int = 64,
        n_outputs: int = 16,
        opt_GNN: str = "SAGE",  
        dropout_rate: float = 0.1,
        use_skip_connections: bool = True,
        use_attention: bool = True,
        n_attention_heads: int = 2,
        use_layer_scale: bool = False,
        layer_scale_init: float = 1e-2,
        use_stochastic_depth: bool = False,
        stochastic_depth_rate: float = 0.1,
        s: int = 64, 
        m: float = 0.1,
        easy_margin: bool = False,
        combine_method: str = 'concat',  # 'concat', 'sum', 'attention'
        input_split: Optional[List[Tuple[int, int]]] = None,  # [(start1, end1), (start2, end2), ...]
    ):
        """
        Enhanced annotation model for both single-modal and multi-modal cases.
        
        Args:
            n_inputs: Input feature dimension (int for single-modal, list for multi-modal)
            n_labels: Number of classes for annotation
            n_domains: Number of domains for domain adaptation
            n_hiddens: Hidden layer dimension
            n_outputs: Output feature dimension
            opt_GNN: Type of GNN layer to use
            dropout_rate: Dropout rate
            use_skip_connections: Whether to use residual connections
            use_attention: Whether to use attention mechanisms
            n_attention_heads: Number of attention heads if using attention
            use_layer_scale: Whether to use layer scale
            layer_scale_init: Initial value for layer scale
            use_stochastic_depth: Whether to use stochastic depth
            stochastic_depth_rate: Rate for stochastic depth (0-1)
            s: Scale factor for ArcFace
            m: Margin for ArcFace
            easy_margin: Use easy margin for ArcFace
            combine_method: How to combine features ('concat', 'sum', 'attention')
            input_split: List of (start, end) indices for each modality in input matrix
                        None for single-modal or already separated inputs
        """
        super().__init__()
        
        # Handle input configuration
        self.is_multi_modal = isinstance(n_inputs, list)
        self.combine_method = combine_method
        self.input_split = input_split
        
        if self.is_multi_modal:
            self.n_inputs_list = n_inputs
            self.n_modalities = len(n_inputs)
        else:
            self.n_inputs_list = [n_inputs]
            self.n_modalities = 1
        self.total_input_dim = sum(self.n_inputs_list)
            
        self.n_labels = n_labels
        self.n_domains = n_domains
        self.n_hiddens = n_hiddens
        self.n_outputs = n_outputs
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.use_attention = use_attention
        self.n_attention_heads = n_attention_heads
        self.gnn_layer = self._get_gnn_layer(opt_GNN, n_attention_heads)
        self.use_layer_scale = use_layer_scale
        self.use_stochastic_depth = use_stochastic_depth
        self.layer_scale_init = layer_scale_init
        self.stochastic_depth_rate = stochastic_depth_rate
        
        ######## Modality-specific Encoders ########
        self.encoder = nn.ModuleList()
        for i, n_in in enumerate(self.n_inputs_list):
            encoder = self._build_encoder(
                n_in, n_hiddens, n_outputs, f"encoder_{i}"
            )
            self.encoder.append(encoder)
        
        ######## Feature Combiner ########
        if self.is_multi_modal:
            if combine_method == "concat":
                combiner_input_dim = n_outputs * self.n_modalities
            elif combine_method in ["sum", "attention"]:
                combiner_input_dim = n_outputs
            else:
                raise ValueError(f"Unknown combine_method: {combine_method}")
                
            self.combiner = self._build_combiner(combiner_input_dim, n_outputs)
            
            if combine_method == "attention":
                self.combine_attention = nn.MultiheadAttention(
                    n_outputs, n_attention_heads, dropout=dropout_rate, batch_first=True
                )
                self.combine_ln = LayerNorm(n_outputs)
        else:     
            self.combiner = self._build_combiner(n_outputs, n_outputs)
        
        ######## ArcFace Margin Classifier ########
        self.super_encoder = ArcMarginProduct(
            n_inputs=n_outputs,
            n_labels=n_labels,
            n_hiddens=n_hiddens,
            n_outputs=n_outputs,
            s=s,
            m=m,
            easy_margin=easy_margin
        )
        
        ######## Decoders ########
        self.decoder = Sequential('x, edge_index', [
                    (self.gnn_layer(n_outputs, self.total_input_dim), 'x, edge_index -> x'),
                    (nn.Dropout(dropout_rate), 'x -> x')
                ])
        
        ######## Domain Classifier ########
        self.clf_domain = self._build_domain_classifier(n_outputs, n_hiddens, n_domains)
        
        ######## Label Classifier ########
        self.clf_label = NormedLinear(n_outputs, n_labels)
        
        ######## Gradient Reversal ########
        self.max_iter = 10000.0
        self.grl = GradientReverseModule(
            lambda step: aToBSheduler(
                step, 0.0, 1, gamma=10, max_iter=self.max_iter
            )
        )
        
        # Initialize weights
        self.apply(self._init_weights)

    def _get_gnn_layer(self, opt_GNN: str, n_heads: int = 4):
        """Get the appropriate GNN layer constructor"""
        if opt_GNN not in OPTIMIZERS:
            raise ValueError(f"Unsupported GNN type: {opt_GNN}")
            
        if opt_GNN == "GAT" and not self.use_attention:
            return OPTIMIZERS["GCN"]  # Fall back to GCN if attention disabled
            
        if opt_GNN == "GAT":
            # Wrap GAT to ensure output dimension matches expected n_outputs
            def wrapped_gat(in_dim, out_dim):
                return OPTIMIZERS[opt_GNN](
                    in_dim, 
                    out_dim,  # Split dimension across heads
                    heads=n_heads,
                    concat=False
                )
            return wrapped_gat
            
        return OPTIMIZERS[opt_GNN]

    def _build_encoder(self, in_dim: int, hidden_dim: int, out_dim: int, name: str = ""):
        """Build an encoder block with modern techniques"""
        layers = [
            (self.gnn_layer(in_dim, hidden_dim), 'x, edge_index -> x'),
            (LayerNorm(hidden_dim), 'x -> x'),
            (GELU(), 'x -> x'),
            (nn.Dropout(self.dropout_rate), 'x -> x'),
            (self.gnn_layer(hidden_dim, out_dim), 'x, edge_index -> x'),
        ]
        
        if self.use_skip_connections:
            layers.append((ResidualConnection(out_dim), 'x, x -> x'))
        
        if self.use_layer_scale:
            layers.append((LayerScale(out_dim, init_value=self.layer_scale_init), 'x -> x'))
        
        return Sequential('x, edge_index', layers)

    def _build_combiner(self, in_dim: int, out_dim: int):
        """Build the feature combiner with modern techniques"""
        layers = [
            (self.gnn_layer(in_dim, in_dim), 'x, edge_index -> x'),
            (BatchNorm(in_dim), 'x -> x'),
            (GELU(), 'x -> x'),
            (nn.Dropout(self.dropout_rate), 'x -> x'),
            (self.gnn_layer(in_dim, out_dim), 'x, edge_index -> x'),
        ]
        
        if self.use_skip_connections:
            layers.insert(4, (ResidualConnection(in_dim), 'x, x -> x'))
        
        return Sequential('x, edge_index', layers)

    def _build_domain_classifier(self, in_dim: int, hidden_dim: int, out_dim: int):
        """Build the domain classifier with modern techniques"""
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),  # Swish activation
            nn.Dropout(self.dropout_rate),
            NormedLinear(hidden_dim, out_dim)
        )

    def _init_weights(self, module):
        """Initialize weights with modern schemes"""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

    def _split_inputs(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Split input tensor into modalities according to input_split"""
        if not self.is_multi_modal:
            return [x]
            
        if self.input_split is None:
            raise ValueError("For multi-modal inputs, input_split must be provided")
            
        if len(self.input_split) != self.n_modalities:
            raise ValueError(f"input_split length ({len(self.input_split)}) must match number of modalities ({self.n_modalities})")
            
        return [x[:, start:end] for start, end in self.input_split]

    def _combine_features(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """Combine features from different modalities"""
        if not self.is_multi_modal:
            return feats[0]
            
        if self.combine_method == "concat":
            return torch.cat(feats, dim=1)
        elif self.combine_method == "sum":
            return torch.stack(feats, dim=0).sum(dim=0)
        elif self.combine_method == "attention":
            # Reshape for attention: [batch_size, num_modalities, feature_dim]
            feat_stack = torch.stack(feats, dim=1)
            attn_out, _ = self.combine_attention(
                feat_stack, feat_stack, feat_stack
            )
            return self.combine_ln(attn_out.mean(dim=1))
        else:
            raise ValueError(f"Unknown combine_method: {self.combine_method}")

    def forward(
        self,
        x_list: List[torch.Tensor],
        edge_index_list: List[torch.Tensor], 
        label_list: Optional[List[torch.Tensor]] = None,
        reverse: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Forward pass handling both source/target domains and multi-modal cases.
    
        Args:
            x_list: List containing source and target data tensors
            edge_index_list: List containing source and target edge indices
            label_list: Optional list containing source and target labels
            reverse: Whether to use gradient reversal
        
        Returns:
            tuple: (combined_features, domain_pred, recon_features, label_pred)
        """
        # Process source and target separately
        all_feats = []
        all_recon_feats = []
        all_domain_preds = []
        all_label_preds = []
    
        for domain_idx, (x, edge_index) in enumerate(zip(x_list, edge_index_list)):
            # Split into modalities if multi-modal
            if self.is_multi_modal:
                x_modalities = self._split_inputs(x)
            else:
                x_modalities = [x]
        
            # Process each modality
            feats = []
            for i, encoder in enumerate(self.encoder):
                x_i = x_modalities[i]
            
                # Apply stochastic depth if enabled
                if self.use_stochastic_depth and self.training:
                    keep_prob = 1.0 - self.stochastic_depth_rate * (i / len(self.encoder))
                    if torch.rand(1).item() > keep_prob:
                        # Skip this encoder during training
                        dummy_feat = torch.zeros(
                            x_i.size(0), self.n_outputs, 
                            device=x_i.device
                        )
                        feats.append(dummy_feat)
                        recon_feats.append(x_i)  # Identity reconstruction
                        continue
            
                # Encode features
                feat = encoder(x_i, edge_index)
                feats.append(feat)
        
            # Combine modalities
            combined_feat = self._combine_features(feats)
            if self.is_multi_modal:
                combined_feat = self.combiner(combined_feat, edge_index)

            # Reconstruct features
            recon_feat = self.decoder(combined_feat, edge_index)
            all_recon_feats.append(recon_feat)
        
            # Store results for this domain
            all_feats.append(combined_feat)
    
            # Domain classification (only for source if doing adaptation)
            if domain_idx == 0 or reverse:  # Always classify source, target only when not reversing
                feat_re = self.grl(combined_feat) if reverse else combined_feat
                domain_pred = self.clf_domain(feat_re)
                all_domain_preds.append(domain_pred)
        
            # Label prediction (only for source if labels available)
            label_pred = self.clf_label(combined_feat)
            all_label_preds.append(label_pred)
    
        return all_feats, all_domain_preds, all_recon_feats, all_label_preds
