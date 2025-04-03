import os
import time
import copy
import math
import anndata
import scanpy as sc
from timeit import default_timer as timer
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import sklearn.metrics
import seaborn as sns
import logging
from functools import partial
from sklearn.decomposition import PCA
from itertools import combinations

from scipy.spatial.distance import cdist 
from torch.autograd import grad

from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import random
from typing import Union, Callable, Any, Iterable, List, Optional

from .model import MovingAverage, ExponentialMovingAverage


logger = logging.getLogger(__name__)

OPTIMIZERS = {
        "adadelta": optim.Adadelta,
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
            }

def correlation_loss_func(
    z1: torch.Tensor, 
    z2: torch.Tensor, 
    lamb: float = 5e-3, 
    scale_loss: float = 0.025,
) -> torch.Tensor:
    """Computes Correlation loss given batch of projected features z1 from view 1 and projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        lamb (float, optional): off-diagonal scaling factor for the cross-covariance matrix.
            Defaults to 5e-4.
        scale_loss (float, optional): final scaling factor of the loss. Defaults to 0.5.

    Returns:
        torch.Tensor: Correlation Loss.
    """

    N, D = z1.size()

    # Batch normalization along the features (D dimension) for each instance (N dimension)
    bn = torch.nn.BatchNorm1d(N, affine=False).to(z1.device)
    z1 = bn(z1.T).T
    z2 = bn(z2.T).T

    # Compute the correlation matrix
    corr = (z1 @ z2.T) / N

    # Distributed computing: all_reduce and normalize by world_size
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(corr)
        world_size = dist.get_world_size()
        corr /= world_size

    # Create a diagonal matrix
    diag = torch.eye(N, device=corr.device)

    # Compute the squared differences between the correlation matrix and the diagonal matrix
    cdif = (corr - diag).pow(2)

    # Apply the scaling factor for off-diagonal elements
    cdif[~diag.bool()] *= lamb

    # Sum up the squared differences and scale by the final scaling factor
    loss = scale_loss * cdif.sum()

    return loss


class train_integrate(object):
    def __init__(
        self,
        minemodel,
        save_path: str,
        device = "cpu",
        )-> None:
        self.device = device
        self.minemodel = minemodel.to(self.device) 
        self.save_path = save_path

    def _get_optimizer(
        self,
        hyperparams: dict,
        epochs,
        optimizer_name: str = "adam",
        ):
        lr = hyperparams['learning_rate']
        wd = hyperparams['weight_decay']
        self.step_scheduler = hyperparams['step_scheduler']
        self.tau = hyperparams['tau']
        parameter_mine_model = self.minemodel.parameters()
        opt_callable = OPTIMIZERS[optimizer_name.lower()]
        ######## configure optimizer and scheduler
        self.optimizer = opt_callable(list(parameter_mine_model),
                                       lr = lr, 
                                       weight_decay = wd
                                        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                                optimizer = self.optimizer,
                                T_max = epochs, 
                                eta_min= lr / 10000,
                                )
        self.scaler = torch.cuda.amp.GradScaler()

    def _train(
        self,
        samples: dict,
        epochs: int,
        hyperparams: dict,
        optimizer_name: str = "adam",
        lamb: float = 5e-3, 
        scale_loss: float = 0.025,
        ):
        criterion = nn.CrossEntropyLoss()
        criterion_re = nn.MSELoss(reduction="mean")
        # criterion_diff = DiffLoss()
        # configure optimizer
        self._get_optimizer(hyperparams = hyperparams, epochs = epochs, optimizer_name = optimizer_name)
        #### early stopping varables
        start_epoch = 0 
        self.history = {
                "epoch_loss": [],
                "epoch_Diff_loss": [],
                "epoch_MMD_loss": [],
                "epoch_domain_loss": [],
                "epoch_re_loss": [],
                "epoch_com_loss": [],
                }
        iters = len(samples["graph_dl"])
        now = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
        ###### training loop
        pbar = tqdm(range(1, epochs+1), desc="Project..")
        for epoch in pbar:
            running_loss = 0.0
            running_Diff_loss = 0.0
            running_MMD_loss = 0.0
            running_domain_loss = 0.0
            running_re_loss = 0.0 
            running_com_loss = 0.0 
            for graph in samples["graph_dl"]:
                ########### set network to train mode
                self.minemodel.train()
                ####### all steps have similar beggining 
                ####### phase 1: train networks to minimize loss on source
                self.optimizer.zero_grad()
                datas = []
                domains = []
                batches = []
                for i in range(samples["n_samples"]):
                    data = graph[f"data_{i}"].to(self.device)
                    domain = graph[f"domain_{i}"].to(self.device)
                    batch = graph[f"batch_{i}"].to(self.device)
                    datas.append(data)
                    domains.append(domain)
                    batches.append(batch)
                edge_index = graph["edge_index"].to(self.device)
                ####### convert target to source
                feats, domain_preds, recon_feats, combine_recon = self.minemodel(datas, batches, edge_index)
                loss_MMD = self._compute_MMD_loss(combined_data = feats)
                # Within-view Reconstruction Loss
                loss_re = 0
                loss_com = 0
                for i in range(len(datas)):
                    loss_re += criterion_re(datas[i], recon_feats[i])
                    loss_com += correlation_loss_func(feats[i], combine_recon, lamb = lamb, scale_loss = scale_loss)

                ###### Across privates loss
                combs = combinations(range(len(feats)), 2)
                loss_Diff = 0
                for comb in list(combs):
                    loss_Diff += correlation_loss_func(feats[comb[0]], feats[comb[1]], lamb = lamb, scale_loss = scale_loss)

                loss_domain = criterion(
                        torch.cat([domain_pred for domain_pred in domain_preds], dim = 0),
                        torch.cat([domain for domain in domains], dim = 0))
                ######## get total loss
                loss_total =  loss_re + loss_domain + loss_com + loss_Diff + loss_MMD
                ############ backpropagate and update weights
                loss_total.backward()
                self.optimizer.step()
                # self.scaler.scale(loss_total).backward()
                # self.scaler.step(self.optimizer)
                # self.scaler.update()
                ### metric
                running_loss += loss_total.item()
                running_Diff_loss += loss_Diff.item()
                running_MMD_loss += loss_MMD.item()
                running_domain_loss += loss_domain.item()
                running_re_loss += loss_re.item()
                running_com_loss += loss_com.item()

            # get losses
            epoch_loss = running_loss / iters
            epoch_MMD_loss = running_MMD_loss / iters
            epoch_Diff_loss = running_Diff_loss / iters
            epoch_domain_loss = running_domain_loss / iters 
            epoch_re_loss = running_re_loss / iters 
            epoch_com_loss = running_com_loss / iters 

            if self.step_scheduler:
                self.scheduler.step()
            self.history["epoch_loss"].append(epoch_loss)
            self.history["epoch_Diff_loss"].append(epoch_Diff_loss)
            self.history["epoch_MMD_loss"].append(epoch_MMD_loss)
            self.history["epoch_domain_loss"].append(epoch_domain_loss)
            self.history["epoch_re_loss"].append(epoch_re_loss)
            self.history["epoch_com_loss"].append(epoch_com_loss)
            pbar.set_postfix(
                        {"Loss": epoch_loss, "Diff": epoch_Diff_loss, 
                         "Domain": epoch_domain_loss, "RE": epoch_re_loss, 
                         "Combine": epoch_com_loss, "MMD": epoch_MMD_loss
                         } 
                        )
        torch.save({'integrate_model_weights': self.minemodel.state_dict(),
                }, os.path.join(self.save_path, f"Dirac_integration_{now}.pt"))
        return now

    def evaluate(
        self, 
        samples,
        ):
        """
        Evaluates model on `dataloader`.
        Arguments:
        ----------
        dataloader: PyTorch DataLoader
            DataLoader with test data.
        return_lists_roc: bool
            If True returns also list of labels, a list of outputs and a list of predictions.
            Useful for some metrics.
        Returns:
        --------
        accuracy: float
            Accuracy achieved over `dataloader`.
        """

        # set network to evaluation mode
        self.minemodel.eval()
        datas = []
        batches = []
        for i in range(samples["n_samples"]):
            data = samples["graph_ds"][f"data_{i}"].to(self.device)
            batch = samples["graph_ds"][f"batch_{i}"].to(self.device)       
            datas.append(data)
            batches.append(batch)
        edge_index = samples["graph_ds"]["edge_index"].to(self.device)
        with torch.no_grad():
            feats, _, _, combine_recon = self.minemodel(datas, batches, edge_index)
            datas_z = []
            for i in range(samples["n_samples"]):
                data_z = feats[i].cpu().detach().numpy()
                datas_z.append(data_z)
            # all_z = np.vstack((z for z in datas_z))
            all_z = np.vstack(datas_z)
        return all_z, combine_recon.cpu().numpy()  

    def _compute_dist_loss(
        self,
        latent_z,
        diff_sim,
        mask = None,
        mode = "mse",
        ):
        latent_sim = self._compute_pairwise_distance(latent_z, latent_z)
        if mode == "mse":
            latent_sim = latent_sim / torch.norm(latent_sim, p='fro')
            diff_sim = diff_sim / torch.norm(diff_sim, p = 'fro')
            if mask is not None:
                loss = torch.norm((diff_sim - latent_sim) * mask, p = 'fro')
            else:
                loss = torch.norm(diff_sim - latent_sim, p = 'fro')
        elif mode == "kl":
            latent_dist = latent_sim / torch.sum(latent_sim) + 1e-12
            diff_dist = diff_sim / torch.sum(diff_sim) + 1e-12
            loss = torch.sum(latent_dist * torch.log(latent_dist / diff_dist))
        return loss

    @staticmethod
    def _compute_pairwise_distance(
        x,
        y,
        ):
        x_norm = (x**2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.clamp(dist, 0.0, np.inf)

    def _compute_gaussian_kernel(
        self,
        x,
        y,
        ):
        sigmas = torch.FloatTensor([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]).to(self.device)
        dist = self._compute_pairwise_distance(x, y)
        beta = 1. / (2. * sigmas[:,None])
        s = - beta.mm(dist.reshape((1, -1)))
        gaussian_matrix =  torch.sum(torch.exp(s), dim = 0)
        return gaussian_matrix

    def _compute_MMD_loss(
        self,
        combined_data,
        ):
        n_batches = len(combined_data)
        start_batch = 0 
        loss = 0
        for batch in range(n_batches):
            if batch == start_batch:
                loss += torch.mean(self._compute_gaussian_kernel(combined_data[batch], combined_data[batch]))
            else:
                loss += torch.mean(self._compute_gaussian_kernel(combined_data[batch], combined_data[batch]))
        #########
        for batch in range(1, n_batches):
            loss -= 2.0 * torch.mean(self._compute_gaussian_kernel(combined_data[start_batch], combined_data[batch]))

        loss = torch.sqrt(loss ** 2 + 1e-12)
        if loss.data.item()<0:
            loss = torch.FloatTensor([0.0]).to(self.device)
        return loss

    def Noise_Cross_Entropy(
        self, 
        pred_sp, 
        emb_sp
        ):
        '''\
        Calculate noise cross entropy. Considering spatial neighbors as positive pairs for each spot
            
        Parameters
        ----------
        pred_sp : torch tensor
            Predicted spatial gene expression matrix.
        emb_sp : torch tensor
            Reconstructed spatial gene expression matrix.

        Returns
        -------
        loss : float
            Loss value.

        '''
        
        mat = self.cosine_similarity(pred_sp, emb_sp) 
        k = torch.exp(mat).sum(axis=1) - torch.exp(torch.diag(mat, 0))
        
        # positive pairs
        p = torch.exp(mat).sum(axis=1)
        # p = torch.mul(p, self.graph_neigh).sum(axis=1)
        
        ave = torch.div(p, k)
        loss = - torch.log(ave).mean()
        
        return loss
    
    def cosine_similarity(
        self, 
        pred_sp, 
        emb_sp
        ):  #pres_sp: spot x gene; emb_sp: spot x gene
        '''\
        Calculate cosine similarity based on predicted and reconstructed gene expression matrix.    
        '''
        M = torch.matmul(pred_sp, emb_sp.T)
        Norm_c = torch.norm(pred_sp, p=2, dim=1)
        Norm_s = torch.norm(emb_sp, p=2, dim=1)
        Norm = torch.matmul(Norm_c.reshape((pred_sp.shape[0], 1)), Norm_s.reshape((emb_sp.shape[0], 1)).T) + -5e-12
        M = torch.div(M, Norm)
        
        if torch.any(torch.isnan(M)):
           M = torch.where(torch.isnan(M), torch.full_like(M, 0.4868), M)

        return M 


class MarginLoss(nn.Module):
    
    def __init__(
        self, 
        m=0.2, 
        weight=None, 
        s=10
        ):
        super(MarginLoss, self).__init__()
        self.m = m
        self.s = s
        self.weight = weight

    def forward(
        self, 
        x, 
        target
        ):
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.data.view(-1, 1), 1)
        x_m = x - self.m * self.s
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(output, target, weight=self.weight)


def entropy(x):
    """ 
    Helper function to compute the entropy over the batch 
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """
    EPS = 1e-5
    x_ =  torch.clamp(x, min = EPS)
    b =  x_ * torch.log(x_)

    if len(b.size()) == 2: # Sample-wise entropy
        return - b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))


class BarlowLoss(nn.Module):
    def __init__(
        self, 
        lmbda: float=5e-3, 
        reduction='mean'
        ):
        super().__init__()
        self.lmbda = lmbda
        self.reduction = reduction
        
    def _off_diagonal(
        self, 
        x
        ):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        
    def forward(
        self, 
        z1, 
        z2
        ):
        n, d = z1.shape
        # normalize along batch dim
        z1 = (z1 - z1.mean(0)) / z1.std(0) # NxD
        z2 = (z2 - z2.mean(0)) / z2.std(0) # NxD
        
        # cross correltation matrix
        cor = torch.mm(z1.T, z2)
        cor.div_(n)
        
        # loss
        on_diag = torch.diagonal(cor).add_(-1).pow_(2).sum()
        off_diag = self._off_diagonal(cor).pow_(2).sum()
        
        loss = on_diag + self.lmbda * off_diag
        
        if self.reduction == 'mean':
            return loss
        else:
            raise ValueError

class DistillLoss(nn.Module):
    def __init__(
        self, 
        warmup_teacher_temp_epochs, 
        nepochs, 
        ncrops=2, 
        warmup_teacher_temp=0.07, 
        teacher_temp=0.04,
        student_temp=0.1
        ):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(
        self, 
        student_output, 
        teacher_output, 
        epoch
        ):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss


class CenterLoss(nn.Module):

    def __init__(
        self, 
        num_classes, 
        feat_dim
        ):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(
        self, 
        x, 
        labels
        ):
        '''
        Parameters:
            x: input tensor with shape (batch_size, feat_dim)
            labels: ground truth label with shape (batch_size)
        Return:
            loss of centers
        '''
        # Move centers to the same device as input x
        centers = self.centers.to(x.device)
        
        # compute the distance of (x-center)^2
        batch_size = x.size(0)
        x_norm = torch.sum(x**2, dim=1, keepdim=True)
        centers_norm = torch.sum(centers**2, dim=1, keepdim=True)
        distmat = x_norm - 2 * torch.matmul(x, centers.t()) + centers_norm.t()

        # get one_hot matrix
        one_hot = torch.zeros(batch_size, self.num_classes, device=x.device)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Compute loss using masked distance matrix
        dist = torch.sum(distmat * one_hot, dim=1)
        loss = torch.mean(dist)

        return loss



class train_annotate(train_integrate):
    """
    Paper: AdaMatch: A Unified Approach to Semi-Supervised Learning and Domain Adaptation
    Authors: David Berthelot, Rebecca Roelofs, Kihyuk Sohn, Nicholas Carlini, Alex Kurakin
    """
    def _train_supervised(
        self, 
        samples,
        graph_dl_source,
        class_weight,
        epochs: int = 50,
        ):
        self.novel_model = self.minemodel.super_encoder
        optimizer = optim.Adam(self.novel_model.parameters(), lr=1e-3, weight_decay=5e-2)
        start_epoch = 0
        center = CenterLoss(num_classes = samples["n_labels"], feat_dim = samples["n_outputs"])
        pbar = tqdm(range(start_epoch+1, epochs+1), desc = "Dirac Find novel cell type..")
        for epoch in pbar:
            sum_loss = 0.0
            for batch_idx, labeled_x in enumerate(graph_dl_source):
                self.novel_model.train()
                source_x = labeled_x.data.to(self.device)
                source_edge_index = labeled_x.edge_index.to(self.device)
                source_label = labeled_x.label.to(self.device)
                optimizer.zero_grad()
                feat, output = self.novel_model(source_x, source_edge_index, source_label)
                center_loss = center(feat, source_label) 
                ce_loss = self._compute_source_loss(output, source_label, class_weight = class_weight)
                loss = ce_loss + center_loss
                sum_loss += loss.item()
                loss.backward()
                optimizer.step()

            pbar.set_postfix({'Loss': sum_loss / (batch_idx + 1)})
        return self.novel_model

    def _est_seeds(
        self,
        source_graph,
        target_graph,
        clusters,
        num_novel_class: int = 3,
        precent = 0.95,
        ):
        self.novel_model.eval()
        entrs = np.array([])
        with torch.no_grad():
            for _, graph in enumerate(target_graph):
                target_graph_cp = copy.deepcopy(graph.data).to(self.device)
                target_edge_index = copy.deepcopy(graph.edge_index).to(self.device)
                _, output = self.novel_model.predict(target_graph_cp, target_edge_index)
                prob = F.softmax(output)
                sorted_prob, _ = torch.sort(prob, dim=1)
                entr1 = -torch.abs(sorted_prob[:, -1] - sorted_prob[:, -2])
                entr2 = -torch.sum(prob * torch.log(prob), 1)
                normalized_entr1 = (entr1 - entr1.min()) / (entr1.max() - entr1.min())
                normalized_entr2 = (entr2 - entr2.min()) / (entr2.max() - entr2.min())
                entr = normalized_entr1 * normalized_entr2
                entrs = np.append(entrs, entr.cpu().numpy())

        entrs_per_cluster = []
        for i in range(np.max(clusters)+1):
            locs = np.where(clusters == i)[0]
            entrs_per_cluster.append(np.mean(entrs[locs]))
        entrs_per_cluster = np.array(entrs_per_cluster)
        if num_novel_class > 0:
            novel_cluster_idxs = np.argsort(entrs_per_cluster)[-num_novel_class:]
        else:
            novel_cluster_idxs = []
        novel_label_seeds = np.zeros_like(clusters)
        largest_seen_id = torch.max(source_graph.label)
        for j, idx in enumerate(novel_cluster_idxs):
            sub_entrs = np.argsort(entrs[clusters == idx])[-int(len(clusters[clusters == idx]) * precent):]
            locs = np.where(clusters == idx)[0][sub_entrs]
            novel_label_seeds[locs] = largest_seen_id + j + 1
        return novel_label_seeds, entrs

    def _generate_center(
        self,
        pre_model, 
        samples, 
        ):
        all_feat = []
        all_labels = []

        class_mean = torch.zeros(samples["n_labels"], samples["n_outputs"]).to(self.device)
        class_sig = torch.zeros(samples["n_labels"], samples["n_outputs"]).to(self.device)

        for batch_idx, labeled_x in enumerate(samples["source_graph_dl"]):
            source_x = labeled_x.data.to(self.device)
            source_label = labeled_x.label.to(self.device)
            source_edge_index = labeled_x.edge_index.to(self.device)
            feats, _ = pre_model(source_x, source_edge_index, source_label)

            all_feat.append(feats.detach().clone().to(self.device))
            all_labels.append(source_label.detach().clone().to(self.device))

        all_feat = torch.cat(all_feat, dim=0).to(self.device)
        all_labels = torch.cat(all_labels, dim=0).to(self.device)

        for i in range(samples["n_labels"]):
            this_feat = all_feat[all_labels==i]
            this_mean = this_feat.mean(dim=0)
            this_var = this_feat.var(dim=0)
            class_mean[i, :] = this_mean
            class_sig[i, :] = (this_var + 1e-5).sqrt()

        class_mean, class_sig, class_cov = class_mean.to(self.device), class_sig.to(self.device), 0

        return class_mean, class_sig, class_cov


    def _sample_labeled_features(
        self,
        samples,
        class_mean, 
        class_sig, 
        num_per_class: int = 20,
        ):
        feats = []
        labels = []
        for i in range(samples["n_labels"]):
            dist = torch.distributions.Normal(class_mean[i].to(self.device), class_sig.mean(dim=0).to(self.device))
            this_feat = dist.sample((num_per_class,)).to(self.device)  # new API
            this_label = torch.ones(this_feat.size(0)).to(self.device) * i

            feats.append(this_feat)
            labels.append(this_label)

        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0).long()

        return feats, labels

    def _calculate_similarity_loss(
        self,
        feat,
        source_label,
        source_len,
        batch_size,
        prob,
        ):
        # Similarity labels
        bce = nn.BCELoss()
        feat_detach = feat.detach()
        feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
        cosine_dist = torch.mm(feat_norm, feat_norm.t())

        pos_pairs = []
        target = source_label
        target_np = target.cpu().numpy()
            
        for i in range(source_len):
            target_i = target_np[i]
            idxs = np.where(target_np == target_i)[0]
            if len(idxs) == 1:
                pos_pairs.append(idxs[0])
            else:
                selec_idx = np.random.choice(idxs, 1)
                while selec_idx == i:
                    selec_idx = np.random.choice(idxs, 1)
                pos_pairs.append(int(selec_idx))
            
        unlabel_cosine_dist = cosine_dist[source_len:, :]
        vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)
        pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
        pos_pairs.extend(pos_idx)
            
        pos_prob = prob[pos_pairs, :]
        pos_sim = torch.bmm(prob.view(batch_size, 1, -1), pos_prob.view(batch_size, -1, 1)).squeeze()
        ones = torch.ones_like(pos_sim)
        bce_loss = bce(pos_sim, ones)

        return bce_loss

    def _train_novel(
        self, 
        pre_model,
        samples,
        epochs: int, 
        hyperparams: dict, 
        weights: dict,
        optimizer_name: str = "adam",
        ):
        """
        Trains the model (encoder + classifier).
        Arguments:
        ----------
        source_dataloader_weak: PyTorch DataLoader
            DataLoader with source domain training data with weak augmentations.
        source_dataloader_strong: PyTorch DataLoader
            DataLoader with source domain training data with strong augmentations.
        target_dataloader_weak: PyTorch DataLoader
            DataLoader with target domain training data with weak augmentations.
            THIS DATALOADER'S BATCH SIZE MUST BE 3 * SOURCE_DATALOADER_BATCH_SIZE.
        target_dataloader_strong: PyTorch DataLoader
            DataLoader with target domain training data with strong augmentations.
            THIS DATALOADER'S BATCH SIZE MUST BE 3 * SOURCE_DATALOADER_BATCH_SIZE. 
        source_dataloader_test: PyTorch DataLoader
            DataLoader with target domain validation data, used for early stopping.
        epochs: int
            Amount of epochs to train the model for.
        hyperparams: dict
            Dictionary containing hyperparameters for this algorithm. Check `data/hyperparams.py`.
        save_path: str
            Path to store model weights.
        Returns:
        --------
        encoder: PyTorch neural network
            Neural network that receives images and encodes them into an array of size X.
        classifier: PyTorch neural network
            Neural network that receives an array of size X and classifies it into N classes.
        """
        ###### load source and target data
        graph_dl_source = samples["source_graph_dl"]
        graph_dl_target = samples["target_graph_dl"]
        graph_dl_test = samples["source_graph_ds"]
        class_weight = samples["class_weight"]

        ###### load pre-model
        pre_model = copy.deepcopy(pre_model)
        pre_model = pre_model.to(self.device)
        pre_model.eval()

        ###### calculate class mean、sig and cov
        class_mean, class_sig, class_cov = self._generate_center(pre_model = pre_model, samples = samples)

        iters = max(len(graph_dl_source), 
                    len(graph_dl_target), 
                    )
        #############
        nclass = samples["n_novel_labels"]
        ####### define loss function
        # criterion_re = SIMSE()
        criterion_re = nn.MSELoss(reduction="mean")
        bce = nn.BCELoss()
        criterion = nn.CrossEntropyLoss()
        ce = MarginLoss(m=-0.2)
        center = CenterLoss(num_classes = samples["n_novel_labels"], feat_dim = samples["n_outputs"])
        # configure optimizer
        self._get_optimizer(hyperparams = hyperparams, epochs = epochs, optimizer_name = optimizer_name)
        # mu related stuff
        steps_per_epoch = iters 
        total_steps = epochs * steps_per_epoch 
        current_step = 0

        # early stopping variables
        start_epoch = 0
        best_acc = 0.0
        patience = 50
        bad_epochs = 0
        self.history = {
                'epoch_loss': [],
                'accuracy_test': [],
                }
        # training loop
        pbar = tqdm(range(start_epoch+1, epochs+1), desc = "Dirac novel training...")
        for epoch in pbar:
            tic = timer()
            running_loss = 0.0
            # this is where the unsupervised learning comes in, as such, we're not interested in labels
            self.minemodel.train()
            for iter_id, (graph_source, graph_target) in enumerate(zip(graph_dl_source, graph_dl_target)):
                ####### load source and target data
                source_data = graph_source["data"].to(self.device)
                source_label = graph_source["label"].to(self.device)
                source_domain = graph_source["domain"].to(self.device)
                source_edge_index = graph_source["edge_index"].to(self.device)

                target_data = graph_target["data"].to(self.device)
                target_ce_idx = torch.where(graph_target["label"]>0)[0]
                target_label = graph_target["label"].to(self.device)
                target_domain = graph_target["domain"].to(self.device)
                target_edge_index = graph_target["edge_index"].to(self.device)
                
                # concatenate data (in case of low GPU power this could be done after classifying with the model)
                self.optimizer.zero_grad()

                # forward pass: calls the model once for both source and target and once for source only
                feats, domain_preds, recon_feats, label_preds  = self.minemodel([source_data, target_data], [source_edge_index, target_edge_index])
                logits_source_p = label_preds[0]

                ###### integrate data
                source_len = len(label_preds[0])
                batch_size = len(label_preds[0]) + len(label_preds[1])
                feat = torch.cat([feats[0], feats[1]], dim=0)
                output = torch.cat((label_preds[0], label_preds[1]))
              
                ######## calculate reconstruction loss
                loss_re = criterion_re(torch.cat([source_data, target_data], dim = 0), torch.cat([recon_feats[0], recon_feats[1]], dim = 0))
                ###### Across privates domain loss
                loss_domain = criterion(torch.cat([domain_preds[0], domain_preds[1]], dim = 0), torch.cat([source_domain, target_domain], dim = 0))

                # from https://github.com/yizhe-ang/AdaMatch-PyTorch/blob/main/trainers/adamatch.py
                self._disable_batchnorm_tracking(self.minemodel.encoder)
                self._disable_batchnorm_tracking(self.minemodel.clf_label)

                source_label_preds = self.minemodel.clf_label(self.minemodel.encoder(source_data, source_edge_index))
                logits_source_pp = source_label_preds

                self._enable_batchnorm_tracking(self.minemodel.encoder)
                self._enable_batchnorm_tracking(self.minemodel.clf_label)

                # perform random logit interpolation
                lambd = torch.rand_like(logits_source_p).to(self.device)
                final_logits_source = (lambd * logits_source_p) + ((1-lambd) * logits_source_pp)

                # distribution allignment
                ## softmax for logits of source
                pseudolabels_source = F.softmax(final_logits_source, 1)

                ## softmax for logits of target
                pseudolabels_target = F.softmax(label_preds[1][~target_ce_idx], 1)

                ## allign target label distribtion to source label (moving average module)
                expectation_ratio = (1e-6 + torch.mean(pseudolabels_source)) / (1e-6 + torch.mean(pseudolabels_target)) ##### 
                final_logits_target = F.normalize((pseudolabels_target * expectation_ratio), p=2, dim=1) # L2 normalization

                # perform relative confidence thresholding
                row_wise_max, _ = torch.max(pseudolabels_source, dim=1)
                final_sum = torch.mean(row_wise_max, 0)
                
                ## define relative confidence threshold
                c_tau = self.tau * final_sum

                max_values, _ = torch.max(final_logits_target, dim=1)
                mask = (max_values >= c_tau).float()
                # compute loss
                source_loss = self._compute_source_loss(logits_weak = final_logits_source, 
                                                        labels = source_label,
                                                        class_weight = class_weight,
                                                        )
                final_target_pseudolabels = torch.max(final_logits_target, 1)[1] # argmax
                target_loss = self._compute_target_loss(final_target_pseudolabels,
                                                        label_preds[1][~target_ce_idx],
                                                        mask,
                                                        )

                ## compute target loss weight (mu)
                pi = torch.tensor(np.pi, dtype=torch.float).to(self.device)
                step = torch.tensor(current_step, dtype=torch.float).to(self.device)
                mu = 0.5 - torch.cos(torch.minimum(pi, (2*pi*step) / total_steps)) / 2

                ## get total loss
                adamatch_loss = source_loss + mu * target_loss

                # #### calculate margin loss
                ce_idx = torch.cat((torch.arange(len(source_label)), len(source_label) + target_ce_idx))
                true_label = torch.cat((source_label, target_label))
                ce_loss = criterion(output[ce_idx], true_label[ce_idx])

                ### calculate entropy loss
                prob = F.softmax(output, dim=1)
                entropy_loss = entropy(torch.mean(prob, 0))

                bce_loss = self._calculate_similarity_loss(feat, source_label, source_len, batch_size, prob)

                ###############
                labeled_feats, labeled_labels = self._sample_labeled_features(samples = samples, 
                                                                              class_mean = class_mean, 
                                                                              class_sig = class_sig, 
                                                                              num_per_class = 20,
                                                                              )
                labeled_output = self.minemodel.clf_label(labeled_feats)
                loss_ce = ce(labeled_output, labeled_labels)

                ###############
                pre_feats, _ = pre_model.predict(target_data, target_edge_index)
                size_1, size_2 = pre_feats[target_ce_idx].size()
                loss_kd = torch.dist(F.normalize(pre_feats[target_ce_idx].view(size_1 * size_2, 1), dim=0),
                                     F.normalize(feats[1][target_ce_idx].view(size_1 * size_2, 1), dim=0))

                current_step += 1

                loss_total = weights["alpha1"]*ce_loss - weights["alpha2"]*entropy_loss + \
                            weights["alpha3"]*adamatch_loss + weights["alpha4"]*loss_domain +\
                            weights["alpha5"]*loss_re + weights["alpha6"]*loss_ce +\
                            weights["alpha7"]*loss_kd + weights["alpha8"]*bce_loss
                # backpropagate and update weights
                loss_total.backward()
                self.optimizer.step()
                # metrics
                running_loss += loss_total.item()

            # get losses
            epoch_loss = running_loss / iters
            if self.step_scheduler:
                self.scheduler.step()

            self.history["epoch_loss"].append(epoch_loss)

            # self.evaluate on testing data (target domain)
            test_epoch_accuracy = self.evaluate_source(graph_dl_test)
            self.history['accuracy_test'].append(test_epoch_accuracy)

            # save checkpoint
            if epoch > epochs // 4 * 3:
                if test_epoch_accuracy > best_acc:
                    torch.save({'Dirac_weights': self.minemodel.state_dict(),
                                }, os.path.join(self.save_path, "Dirac_annotation.pt"))
                    best_acc = test_epoch_accuracy
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                
            pbar.set_postfix(
                {"Loss": epoch_loss,"Loss": epoch_loss}
                )
            
            if bad_epochs >= patience:
                print(f"reached {bad_epochs} bad epochs, stopping training with best test accuracy of {best_acc}!")
                break

        best = torch.load(os.path.join(self.save_path, "Dirac_annotation.pt"), map_location=torch.device('cpu'))
        self.minemodel.load_state_dict(best['Dirac_weights'])
        self.minemodel.to(self.device)
        return self.minemodel


    @torch.no_grad()
    def evaluate_novel_target(
        self, 
        graph_dl, 
        return_lists_roc: bool = False,
        ):
        """
        Evaluates model on `graph_dl_test`.
        Arguments:
        ----------
        graph_dl_test: PyTorch DataLoader
            DataLoader with test data.
        return_lists_roc: bool
            If True returns also list of labels, a list of outputs and a list of predictions.
            Useful for some metrics.
        Returns:
        --------
        accuracy: float
            Accuracy achieved over `dataloader`.
        """

        # set network to evaluation mode
        self.minemodel.eval()
        preds = np.array([])
        confs = np.array([])
        with torch.no_grad():
            all_outputs = []
            all_feats = []
            idxs =[]
            for _, graph in enumerate(graph_dl):
                graph_cp = copy.deepcopy(graph.data).to(self.device)
                edge_index_cp = copy.deepcopy(graph.edge_index).to(self.device)
                idx_cp = copy.deepcopy(graph.idx).to(self.device)
                feat = self.minemodel.encoder(graph_cp, edge_index_cp)
                output = self.minemodel.clf_label(feat)
                all_outputs.append(output)
                all_feats.append(feat)
                idxs.append(idx_cp)

            all_outputs = torch.cat(all_outputs)
            all_feats = torch.cat(all_feats)
            idxs = torch.cat(idxs)
            sorted_indices = torch.argsort(idxs)
            all_outputs = all_outputs[sorted_indices]
            all_feats = all_feats[sorted_indices]

            # save most likely prediction and output probabilities
            prob = F.softmax(all_outputs, dim = 1)
            conf, pred = prob.max(1)
            preds = np.append(preds, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
            # numpify
            feats_numpy = all_feats.detach().cpu().numpy()
            prob_numpy = prob.detach().cpu().numpy()

        preds = preds.astype(int)
        mean_uncert = 1 - np.mean(confs)
        if return_lists_roc:
            return feats_numpy, output, prob_numpy, preds, confs, mean_uncert
        return preds

    def _train(
        self, 
        samples,
        epochs: int, 
        hyperparams: dict, 
        optimizer_name: str = "adam",
        ):
        """
        Trains the model (encoder + classifier).
        Arguments:
        ----------
        source_dataloader_weak: PyTorch DataLoader
            DataLoader with source domain training data with weak augmentations.
        source_dataloader_strong: PyTorch DataLoader
            DataLoader with source domain training data with strong augmentations.
        target_dataloader_weak: PyTorch DataLoader
            DataLoader with target domain training data with weak augmentations.
            THIS DATALOADER'S BATCH SIZE MUST BE 3 * SOURCE_DATALOADER_BATCH_SIZE.
        target_dataloader_strong: PyTorch DataLoader
            DataLoader with target domain training data with strong augmentations.
            THIS DATALOADER'S BATCH SIZE MUST BE 3 * SOURCE_DATALOADER_BATCH_SIZE. 
        source_dataloader_test: PyTorch DataLoader
            DataLoader with target domain validation data, used for early stopping.
        epochs: int
            Amount of epochs to train the model for.
        hyperparams: dict
            Dictionary containing hyperparameters for this algorithm. Check `data/hyperparams.py`.
        save_path: str
            Path to store model weights.
        Returns:
        --------
        encoder: PyTorch neural network
            Neural network that receives images and encodes them into an array of size X.
        classifier: PyTorch neural network
            Neural network that receives an array of size X and classifies it into N classes.
        """
        graph_dl_source = samples["source_graph_dl"]
        graph_dl_target = samples["target_graph_dl"]
        graph_dl_test = samples["source_graph_ds"]
        class_weight = samples["class_weight"]

        iters = max(len(graph_dl_source), 
                    len(graph_dl_target), 
                    )
        self.num_classes = torch.max(graph_dl_test["label"])
        ####### define loss function
        criterion_re = nn.MSELoss(reduction="mean")
        criterion = nn.CrossEntropyLoss()
        center = CenterLoss(num_classes = samples["n_labels"], feat_dim = samples["n_outputs"])
        # configure optimizer
        self._get_optimizer(hyperparams = hyperparams, epochs = epochs, optimizer_name = optimizer_name)
        # mu related stuff
        steps_per_epoch = iters 
        total_steps = epochs * steps_per_epoch 
        current_step = 0

        # early stopping variables
        start_epoch = 0
        best_acc = 0.0
        patience = 50
        bad_epochs = 0
        self.history = {
                'epoch_loss': [],
                'epoch_adamatch_loss': [], 
                'epoch_domain_loss': [], 
                'epoch_re_loss': [], 
                'accuracy_test': []
                }
        # training loop
        pbar = tqdm(range(start_epoch+1, epochs+1), desc = "Dirac annotate training..")
        for epoch in pbar:
            tic = timer()
            running_loss = 0.0
            running_adamatch_loss = 0.0
            running_domain_loss = 0.0
            running_re_loss = 0.0 
            # this is where the unsupervised learning comes in, as such, we're not interested in labels
            for iter_id, (graph_source, graph_target) in enumerate(zip(graph_dl_source, graph_dl_target)):
                self.minemodel.train()

                ####### load source and target data
                source_data = graph_source["data"].to(self.device)
                source_label = graph_source["label"].to(self.device)
                source_domain = graph_source["domain"].to(self.device)
                source_edge_index = graph_source["edge_index"].to(self.device)

                target_data = graph_target["data"].to(self.device)
                target_domain = graph_target["domain"].to(self.device)
                target_edge_index = graph_target["edge_index"].to(self.device)
                
                # concatenate data (in case of low GPU power this could be done after classifying with the model)
                self.optimizer.zero_grad()

                # forward pass: calls the model once for both source and target and once for source only
                feats, domain_preds, recon_feats, label_preds  = self.minemodel([source_data, target_data], [source_edge_index, target_edge_index])
                logits_source_p = label_preds[0]

                ######## calculate reconstruction loss
                loss_re = criterion_re(torch.cat([source_data, target_data], dim = 0), torch.cat([recon_feats[0], recon_feats[1]], dim = 0))
                ###### Across privates domain loss
                loss_domain = criterion(torch.cat([domain_pred for domain_pred in domain_preds], dim = 0), torch.cat([source_domain, target_domain], dim = 0))

                ## calculate entropy loss
                output = torch.cat((label_preds[0], label_preds[1]))
                prob = F.softmax(output, dim=1)
                entropy_loss = entropy(torch.mean(prob, 0))            

                # from https://github.com/yizhe-ang/AdaMatch-PyTorch/blob/main/trainers/adamatch.py
                self._disable_batchnorm_tracking(self.minemodel.encoder)
                self._disable_batchnorm_tracking(self.minemodel.clf_label)

                source_label_preds = self.minemodel.clf_label(self.minemodel.encoder(source_data, source_edge_index))
                logits_source_pp = source_label_preds

                self._enable_batchnorm_tracking(self.minemodel.encoder)
                self._enable_batchnorm_tracking(self.minemodel.clf_label)

                # perform random logit interpolation
                lambd = torch.rand_like(logits_source_p).to(self.device)
                final_logits_source = (lambd * logits_source_p) + ((1-lambd) * logits_source_pp)

                # distribution allignment
                ## softmax for logits of source
                pseudolabels_source = F.softmax(final_logits_source, 1)

                ## softmax for logits of target
                pseudolabels_target = F.softmax(label_preds[1], 1)

                ## allign target label distribtion to source label distribution
                expectation_ratio = (1e-6 + torch.mean(pseudolabels_source)) / (1e-6 + torch.mean(pseudolabels_target))
                final_logits_target = F.normalize((pseudolabels_target * expectation_ratio), p=2, dim=1) # L2 normalization

                # perform relative confidence thresholding
                row_wise_max, _ = torch.max(pseudolabels_source, dim=1)
                final_sum = torch.mean(row_wise_max, 0)
                
                ## define relative confidence threshold
                c_tau = self.tau * final_sum

                max_values, _ = torch.max(final_logits_target, dim=1)
                mask = (max_values >= c_tau).float()
                # compute loss
                source_loss = self._compute_source_loss(logits_weak = final_logits_source, 
                                                        labels = source_label,
                                                        class_weight = class_weight,
                                                        )
                final_target_pseudolabels = torch.max(final_logits_target, 1)[1] # argmax
                target_loss = self._compute_target_loss(final_target_pseudolabels,
                                                        label_preds[1],
                                                        mask,
                                                        )

                ## compute target loss weight (mu)
                pi = torch.tensor(np.pi, dtype=torch.float).to(self.device)
                step = torch.tensor(current_step, dtype=torch.float).to(self.device)
                mu = 0.5 - torch.cos(torch.minimum(pi, (2*pi*step) / total_steps)) / 2

                ## get total loss
                adamatch_loss = source_loss + (mu * target_loss)

                ce_loss = center(feats[0], source_label)

                current_step += 1

                loss_total = loss_re + loss_domain + adamatch_loss - 0.3 * entropy_loss + 0.5 * ce_loss


                # backpropagate and update weights
                loss_total.backward()
                self.optimizer.step()
                # self.scaler.scale(loss_total).backward()
                # self.scaler.step(self.optimizer)
                # self.scaler.update()
                # metrics
                running_loss += loss_total.item()
                running_adamatch_loss += adamatch_loss.item()
                running_domain_loss += loss_domain.item()
                running_re_loss += loss_re.item()

            # get losses
            epoch_loss = running_loss / iters
            epoch_adamatch_loss = running_adamatch_loss / iters
            epoch_domain_loss = running_domain_loss / iters
            epoch_re_loss = running_re_loss / iters 
            if self.step_scheduler:
                self.scheduler.step()

            self.history["epoch_loss"].append(epoch_loss)
            self.history['epoch_adamatch_loss'].append(epoch_adamatch_loss)
            self.history['epoch_domain_loss'].append(epoch_domain_loss)
            self.history["epoch_re_loss"].append(epoch_re_loss)

            # self.evaluate on testing data (target domain)
            test_epoch_accuracy = self.evaluate_source(graph_dl_test)
            self.history['accuracy_test'].append(test_epoch_accuracy)

            # save checkpoint
            if epoch > epochs // 4 * 3:
                if test_epoch_accuracy > best_acc:
                    torch.save({'Dirac_weights': self.minemodel.state_dict(),
                                }, os.path.join(self.save_path, "Dirac_annotation.pt"))
                    best_acc = test_epoch_accuracy
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                
            pbar.set_postfix(
                {"Loss": epoch_loss, "Accuracy": test_epoch_accuracy}
                )
            
            if bad_epochs >= patience:
                print(f"reached {bad_epochs} bad epochs, stopping training with best test accuracy of {best_acc}!")
                break

        best = torch.load(os.path.join(self.save_path, "Dirac_annotation.pt"), map_location=torch.device('cpu'))
        self.minemodel.load_state_dict(best['Dirac_weights'])
        self.minemodel.to(self.device)
        return self.minemodel

    @torch.no_grad()
    def evaluate_source(
        self, 
        graph_dl, 
        return_lists_roc: bool = False,
        ):
        """
        Evaluates model on `graph_dl_test`.
        Arguments:
        ----------
        graph_dl_test: PyTorch DataLoader
            DataLoader with test data.
        return_lists_roc: bool
            If True returns also list of labels, a list of outputs and a list of predictions.
            Useful for some metrics.
        Returns:
        --------
        accuracy: float
            Accuracy achieved over `dataloader`.
        """

        # set network to evaluation mode
        self.minemodel.eval()
        with torch.no_grad():
            data = graph_dl["data"].to(self.device)
            label = graph_dl["label"].to(self.device)
            edge_index = graph_dl["edge_index"].to(self.device)
            feats = self.minemodel.encoder(data, edge_index)
            label_preds = self.minemodel.clf_label(feats)
            output = F.softmax(label_preds, dim = 1)
            # numpify
            feats_numpy = feats.detach().cpu().numpy()
            label_numpy = label.detach().cpu().numpy()
            output_numpy = output.detach().cpu().numpy() # probs (AUROC)
            pred = np.argmax(output_numpy, axis=1) # accuracy

        # metrics
        #auc = sklearn.metrics.roc_auc_score(labels_list, outputs_list, multi_class='ovr')
        accuracy = sklearn.metrics.accuracy_score(label_numpy, pred)

        if return_lists_roc:
            return accuracy, feats_numpy, output, pred
            
        return accuracy

    @torch.no_grad()
    def evaluate_target(
        self, 
        graph_dl, 
        return_lists_roc: bool = False,
        ):
        """
        Evaluates model on `graph_dl_test`.
        Arguments:
        ----------
        graph_dl_test: PyTorch DataLoader
            DataLoader with test data.
        return_lists_roc: bool
            If True returns also list of labels, a list of outputs and a list of predictions.
            Useful for some metrics.
        Returns:
        --------
        accuracy: float
            Accuracy achieved over `dataloader`.
        """

        # set network to evaluation mode
        self.minemodel.eval()

        with torch.no_grad():
            data = graph_dl["data"].to(self.device)
            edge_index = graph_dl["edge_index"].to(self.device)
            feats = self.minemodel.encoder(data, edge_index)
            label_preds = self.minemodel.clf_label(feats)
            prob = F.softmax(label_preds, dim = 1)
            conf, pred = prob.max(1)
            # numpify
            
            feats_numpy = feats.detach().cpu().numpy()
            output_numpy = label_preds.detach().cpu().numpy()
            prob_numpy = prob.detach().cpu().numpy() # probs (AUROC)
            pred_numpy = pred.detach().cpu().numpy() # accuracy
            conf_numpy = conf.detach().cpu().numpy()

            pred_numpy = pred_numpy.astype(int)
            mean_uncert = 1 - np.mean(conf_numpy)

        if return_lists_roc:
            return feats_numpy, output_numpy, prob_numpy, pred_numpy, conf_numpy, mean_uncert
        return pred


    @staticmethod
    def _disable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = False

        model.apply(fn)

    @staticmethod
    def _enable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = True

        model.apply(fn)


    @staticmethod
    def _compute_source_loss(
        logits_weak, 
        labels, 
        class_weight = None, 
        reduction = "mean",
        ):
        """
        Receives logits as input (dense layer outputs with no activation function)
        """
        if class_weight is not None:
            class_weight = class_weight.to(logits_weak.device)
        if class_weight is not None:
            weak_loss = F.cross_entropy(logits_weak, labels, weight = class_weight, reduction = reduction)
        else:
            weak_loss = F.cross_entropy(logits_weak, labels, reduction = reduction)
        return weak_loss


    @staticmethod
    def _compute_target_loss(
        pseudolabels, 
        logits_strong, 
        mask
        ):
        """
        Receives logits as input (dense layer outputs with no activation function).
        `pseudolabels` are treated as ground truth (standard SSL practice).
        """
        ce = nn.CrossEntropyLoss() # reduction="none"
        pseudolabels = pseudolabels.detach() # remove from backpropagation
        loss = ce(logits_strong, pseudolabels)
        
        return (loss * mask).mean()

    @staticmethod
    def _compute_center_loss(
        x,
        labels,
        num_classes: int,
        feat_dim: int = 2,
        ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, num_classes) + \
                  torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(num_classes, batch_size).t()
        distmat.addmm(1, -2, x, centers.t())
        classes = torch.arange(num_classes).long().to(device)
        labels = labels.unsqueeze(1).expand(batch_size, num_classes)
        mask = labels.eq(classes.expand(batch_size, num_classes))
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

    @staticmethod
    def _compute_domain_loss(
        domain_pred: torch.FloatTensor,
        dlabel: torch.FloatTensor,
        ):
        #######################################
        ### Compute adaptiive domain Loss
        #######################################
        loss_function = nn.CrossEntropyLoss()
        domain_pred = domain_pred.detach()
        dan_loss =  loss_function(domain_pred, dlabel)
        return dan_loss

    def plot_metrics(
        self,
        ):
        """
        Plots the training metrics (only usable after calling .train()).
        """

        # plot metrics for losses n stuff
        fig, axs = plt.subplots(1, 3, figsize=(18,5), dpi=300)

        epochs = len(self.history['epoch_adamatch_loss'])

        axs[0].plot(range(1, epochs+1), self.history['epoch_adamatch_loss'])
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Adamatch loss')

        axs[1].plot(range(1, epochs+1), self.history['epoch_domain_loss'])
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Loss')
        axs[1].set_title('Domain loss')

        axs[2].plot(range(1, epochs+1), self.history['accuracy_source'])
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel('Accuracy')
        axs[2].set_title('Accuracy on weakly augmented source')

        plt.show()
        plt.savefig(os.path.join(self.save_path, "metrics.pdf"), dpi=300)  

    def plot_cm_roc(
        self, 
        dataloader: dict, 
        n_classes: int,
        ):
        """
        Plots the confusion matrix and ROC curves of the model on `dataloader`.
        Arguments:
        ----------
        dataloader: PyTorch DataLoader
            DataLoader with test data.
        n_classes: int
            Number of classes.
        """

        cmap = sns.diverging_palette(220, 20, as_cmap=True)

        self.feat_model.eval()
        self.predict_model.eval()

        accuracy, labels_list, outputs_list, preds_list = self.test_epoch(dataloader, return_lists_roc=True)

        # plot confusion matrix
        cm = sklearn.metrics.confusion_matrix(labels_list, preds_list)
        group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
        group_percentages = ['({0:.2%})'.format(value) for value in cm.flatten()/np.sum(cm)]
        labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(n_classes,n_classes)
        #tn, fp, fn, tp = cm.ravel()

        plt.figure(figsize=(10,10), dpi=300)
        sns.heatmap(cm, annot=labels, cmap=cmap, fmt="")
        plt.title("Confusion matrix")
        plt.ylabel("Actual label")
        plt.xlabel("Predicted label")
        plt.show()
        plt.savefig(os.path.join(self.save_path, "confusion_matrix.pdf"), dpi=300) 

        # plot roc
        ## one hot encode data
        onehot = np.zeros((labels_list.size, labels_list.max()+1))
        onehot[np.arange(labels_list.size),labels_list] = 1
        onehot = onehot.astype('int')

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        ## get roc curve and auroc for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(onehot[:, i], outputs_list[:, i])
            roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

        ## get macro average auroc
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])

        plt.figure(figsize=(9,9), dpi=200)

        plt.plot([0, 1], [0, 1], color='black', linestyle='--')

        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f"AUC class {i} = {roc_auc[i]:.4f}")

        plt.plot(fpr["macro"], tpr["macro"], label=f"macro-average AUC = {roc_auc['macro']:.4f}", color='deeppink', linewidth=2)
            
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.xlabel('False Positives')
        plt.ylabel('True Positives')
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.legend(loc='lower right')
        plt.show()
        plt.savefig(os.path.join(self.save_path, "ROC.pdf"), dpi=300) 


