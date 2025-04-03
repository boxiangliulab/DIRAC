import os
import time
import random
from typing import Callable, Iterable, Union, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc 
import anndata

import torch
from torch_geometric.loader import DataLoader, ClusterData, ClusterLoader
from torch_geometric.utils import to_undirected
from torchvision import transforms
import torchvision

from .dataprep import GraphDS, GraphDataset, GraphDataset_unpaired
from .model import integrate_model, annotate_model
from .trainer import train_integrate, train_annotate
from .hyper import *


#########################################################
# Dirac's integration and annotation app
#########################################################

class integrate_app():
    def __init__(
        self,
        save_path: str = './Results/',
        subgraph: bool = True,
        use_gpu: bool = True,
        **kwargs,
        )-> None:
        super(integrate_app, self).__init__(**kwargs)
        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        self.subgraph = subgraph
        self.save_path = save_path

    def _get_data(
        self,
        dataset_list: list,
        domain_list: list,
        batch_list: list,
        edge_index,
        num_parts: int = 100,
        num_workers: int = 10,
        batch_size: int = 1,
        ):
        self.n_samples = len(dataset_list)
        ##### counts the number of domains
        if domain_list is None:
            self.num_domains = len(dataset_list)
        else:
            domains_max = []
            for i in range(len(domain_list)):
                domain_max = domain_list[i].max()
                domains_max.append(domain_max)
            self.num_domains = (int(np.max(domains_max)) + 1)
        print(f"Found {self.num_domains} unique domains.")
        ############## find correspondence between datasets & align multi-omics data in a common embedded space
        self.n_inputs_list = []
        for i in range(len(dataset_list)):
            n_inputs = dataset_list[i].shape[1]
            self.n_inputs_list.append(n_inputs)
            if i == 0:
                graph_ds = GraphDataset(
                            data = dataset_list[i],
                            domain = domain_list[i],
                            batch = batch_list[i],
                            edge_index = to_undirected(edge_index),
                            ) 
                graph_data = graph_ds.graph_data
            else:
                graph_data[f"data_{i}"] = torch.FloatTensor(dataset_list[i])
                graph_data[f"domain_{i}"] = torch.LongTensor(domain_list[i])
                graph_data[f"batch_{i}"] = torch.LongTensor(batch_list[i].copy())
        if self.subgraph:
            graph_dataset = ClusterData(graph_data, num_parts = num_parts, recursive = False)
            graph_dl = ClusterLoader(graph_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
        else:
            graph_dl = DataLoader([graph_data])
        samples = {
                    "graph_ds": graph_data,
                    "graph_dl": graph_dl,
                    "n_samples": self.n_samples,
                    "n_inputs_list": self.n_inputs_list,
                    "n_domains": self.num_domains
                    }
        return samples

    def _get_model(
        self,
        samples,
        n_hiddens: int = 128,
        n_outputs: int = 64,
        opt_GNN = "GCN",
        ):
        ##### Build a transfer model to conver atac data to rna shape 
        models = integrate_model(n_inputs_list = samples["n_inputs_list"], 
                            n_domains = samples["n_domains"],
                            n_hiddens = n_hiddens,
                            n_outputs = n_outputs,
                            opt_GNN = opt_GNN,
                            )

        return models

    def _train_dirac_integrate(
        self,
        samples,
        models,
        epochs: int = 500,
        optimizer_name: str = "adam",
        lr: float = 1e-3,
        tau: float = 0.9,
        wd: float = 5e-3,
        scheduler: bool = True,
        lamb: float = 5e-3, 
        scale_loss: float = 0.025,
        ):
        ######### load all dataloaders and dist arrays
        hyperparams = unsuper_hyperparams(lr = lr, tau = tau, wd = wd, scheduler = scheduler)
        un_dirac = train_integrate(
                    minemodel = models,
                    save_path = self.save_path,
                    device = self.device,
                )

        now = un_dirac._train(
                    samples = samples,
                    epochs = epochs,
                    hyperparams = hyperparams,
                    optimizer_name = optimizer_name,
                    lamb = lamb, 
                    scale_loss = scale_loss
            )

        data_z, combine_recon = un_dirac.evaluate(samples = samples)

        return data_z, combine_recon, now


class annotate_app(integrate_app):
    def _get_data(
        self,
        source_data,
        source_label,
        source_edge_index,
        target_data,
        target_edge_index,
        source_domain = None,
        target_domain = None,
        test_data = None,
        test_edge_index = None,
        weighted_classes = True,
        num_workers: int = 1,
        batch_size: int = 1,
        num_parts_source: int = 1,
        num_parts_target: int = 1,
        ):
        self.n_labels = len(np.unique(source_label))
        self.n_inputs = source_data.shape[1]
        # count the number of domains
        if (
            (source_domain is None)
            and (target_domain is None)
            and (target_data is not None)
        ):
            self.n_domains = 2
        elif (
            (source_domain is None)
            and (target_domain is None)
            and (target_data is None)
        ):
            self.n_domains = 1
        elif (source_domain is not None) and (target_domain is None):
            source_domain_max = source_domain.max()
            self.n_domains = int(source_domain_max)
        elif (source_domain is not None) and (target_domain is not None):
            source_domain_max = source_domain.max()
            target_domain_max = (
                0 if len(target_domain) == 0 else target_domain.max()
            )
            self.n_domains = (
                int(
                    np.max(
                        [
                            source_domain_max,
                            target_domain_max,
                        ]
                    )
                )
                + 1
            )
        else:
            msg = "domains supplied for only one set of data"
            raise ValueError(msg)
        print(f"Found {self.n_domains} unique domains.")
        ###### weight classes if applicable
        if weighted_classes:
            u_classes, class_counts = np.unique(source_label, return_counts=True)
            class_prop = class_counts / len(source_label)
            class_weight = 1.0 / class_prop
            class_weight = class_weight / class_weight.min()
            class_weight = torch.from_numpy(class_weight).float()
        else:
            class_weight = None
        ############## find correspondence between datasets & align multi-omics data in a common embedded space
        source_graph_ds = GraphDataset_unpaired(
                            data = source_data,
                            domain = source_domain,
                            edge_index = to_undirected(source_edge_index),
                            label = source_label ####### own label
                            ) 
        source_graph_dataset = ClusterData(source_graph_ds.graph_data, num_parts = num_parts_source, recursive = False)
        source_graph_dl = ClusterLoader(source_graph_dataset, batch_size = batch_size, shuffle=True, num_workers = num_workers)
        target_graph_ds = GraphDataset_unpaired(
                            data = target_data,
                            domain = target_domain,
                            edge_index = to_undirected(target_edge_index),
                            label = None ####### unlabel 
                            )
        target_graph_dataset = ClusterData(target_graph_ds.graph_data, num_parts = num_parts_target, recursive = False)
        target_graph_dl = ClusterLoader(target_graph_dataset, batch_size = batch_size, shuffle=True, num_workers = num_workers)

        if (test_data is not None) and (test_edge_index is not None):
            test_graph_ds = Data(data = torch.FloatTensor(test_data), 
                                 edge_index = test_edge_index)
        else:
            test_graph_ds = None
        samples = {
                    "source_graph_ds": source_graph_ds.graph_data,
                    "source_graph_dl": source_graph_dl,
                    "target_graph_ds": target_graph_ds.graph_data,
                    "target_graph_dl": target_graph_dl,
                    "test_graph_ds": test_graph_ds,
                    "class_weight": class_weight,
                    "n_labels": self.n_labels,
                    "n_inputs": self.n_inputs,
                    "n_domains": self.n_domains,
                    }  
        return samples

    def _get_model(
        self,
        samples,
        n_hiddens: int = 128,
        n_outputs: int = 64,
        opt_GNN: str = "GCN",
        s: int = 32,
        m: float = 0.10,
        easy_margin: bool = False,
        ):
        ##### Build a transfer model to conver atac data to rna shape 
        models = annotate_model(
        				n_inputs= samples["n_inputs"], 
                        n_domains = samples["n_domains"], 
                        n_labels = samples["n_labels"], 
                        n_hiddens = n_hiddens,
                        n_outputs = n_outputs, 
                        opt_GNN = opt_GNN,
                        s = s,
                        m = m,
                        easy_margin = easy_margin,
                        )
        self.n_outputs = n_outputs
        self.opt_GNN = opt_GNN
        self.n_hiddens = n_hiddens

        return models

    def _train_dirac_annotate(
        self,
        samples,
        models,
        epochs: int = 500,
        optimizer_name: str = "adam",
        lr: float = 1e-3,
        wd: float = 5e-3,
        scheduler: bool = True,
        n_epochs: int = 200,
        ):
        ######### load all dataloaders and dist arrays
        samples["n_outputs"] = self.n_outputs
        hyperparams = unsuper_hyperparams(lr = lr, wd = wd, scheduler = scheduler)
        semi_dirac = train_annotate(
                    minemodel = models,
                    save_path = self.save_path,
                    device = self.device,
                )
        semi_dirac._train(
                    samples = samples,
                    epochs = n_epochs,
                    hyperparams = hyperparams,
                    optimizer_name = optimizer_name,
                )
        _, source_feat, _, _ = semi_dirac.evaluate_source(graph_dl = samples["source_graph_ds"], return_lists_roc = True)
        target_feat, target_output, target_prob, target_pred, target_confs, target_mean_uncert = semi_dirac.evaluate_novel_target(graph_dl = samples["target_graph_dl"], return_lists_roc = True)
        if samples["test_graph_ds"] is not None:
            test_feat, test_output, test_prob, test_pred, test_confs, test_mean_uncert = semi_dirac.evaluate_target(graph_dl = samples["test_graph_ds"], return_lists_roc = True)
        else:
            test_feat = None 
            test_output = None
            test_prob = None
            test_pred = None
            test_confs = None
            test_mean_uncert = None
        results = {
                    "source_feat": source_feat,
                    "target_feat": target_feat,
                    "target_output": target_output,
                    "target_prob": target_prob,
                    "target_pred": target_pred,
                    "target_confs": target_confs,
                    "target_mean_uncert": target_mean_uncert,
                    "test_feat": test_feat,
                    "test_output": test_output,
                    "test_prob": test_prob,
                    "test_pred": test_pred,
                    "test_confs": test_confs,
                    "test_mean_uncert": test_mean_uncert,
                    } 

        return results

    def _train_dirac_novel(
        self,
        samples,
        minemodel,
        num_novel_class: int = 3,
        pre_epochs: int = 100,
        n_epochs: int = 200,
        num_parts: int = 30,
        resolution: float = 1, 
        s: int = 64,
        m: float = 0.1,
        weights: dict = {"alpha1": 1,"alpha2": 1,"alpha3": 1,"alpha4": 1,"alpha5": 1,"alpha6": 1,"alpha7": 1,"alpha8": 1}
        ):
        samples["n_outputs"] = self.n_outputs
        samples["opt_GNN"] = self.opt_GNN
        samples["n_hiddens"] = self.n_hiddens
        ######### Find Target Data for novel cell type
        unlabel_x = samples["target_graph_ds"].data
        
        print("Performing louvain...")
        adata = anndata.AnnData(unlabel_x.numpy())
        if adata.shape[1] > 100:
            sc.tl.pca(adata)
            sc.pp.neighbors(adata)
        else:
            sc.pp.neighbors(adata, use_rep = "X")

        sc.tl.louvain(adata, resolution = resolution, key_added='louvain')
        clusters = adata.obs["louvain"].values 
        clusters = clusters.astype(int)
        print("Louvain finished")
        ########## Training SpaGNNs_gpu for source domain 
        semi_dirac = train_annotate(
                    minemodel = minemodel,
                    save_path = self.save_path,
                    device = self.device,
                )
        pre_model = semi_dirac._train_supervised(samples = samples, graph_dl_source = samples["source_graph_dl"], epochs=pre_epochs, class_weight = samples["class_weight"])
        novel_label, entrs = semi_dirac._est_seeds(source_graph = samples["source_graph_ds"], target_graph = samples["target_graph_dl"], clusters = clusters, num_novel_class = num_novel_class)
        

        import time
        now = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
        adata.obs["novel_cell_type"] = pd.Categorical(novel_label) 
        adata.obs["entrs"] = entrs
        sc.tl.umap(adata)
        sc.pl.umap(adata, color=["louvain", "novel_cell_type", "entrs"], cmap="CMRmap_r", size=20)
        plt.savefig(os.path.join(self.save_path, f"UMAP_clusters_{now}.pdf"), bbox_inches='tight', dpi = 300)

        samples["target_graph_ds"].label = torch.tensor(novel_label)
        unlabeled_data = ClusterData(samples["target_graph_ds"], num_parts = num_parts, recursive = False)
        unlabeled_loader = ClusterLoader(unlabeled_data, batch_size=1, shuffle = True, num_workers=1)

        samples["target_graph_dl"] = unlabeled_loader 
        samples["n_novel_labels"] = num_novel_class + samples["n_labels"]
        if samples["class_weight"] is not None:
            samples["class_weight"] = torch.cat([samples["class_weight"], torch.ones(num_novel_class)], dim=0)

        ###### change models
        minemodel = annotate_model(
                        n_inputs= samples["n_inputs"], 
                        n_domains = samples["n_domains"], 
                        n_labels = samples["n_novel_labels"], 
                        n_hiddens = samples["n_hiddens"],
                        n_outputs = samples["n_outputs"], 
                        opt_GNN = samples["opt_GNN"]
                        )

        semi_dirac = train_annotate(
                    	minemodel = minemodel,
                    	save_path = self.save_path,
                    	device = self.device,
               		 	)
        hyperparams = unsuper_hyperparams()
        semi_dirac._train_novel(
                    	pre_model = pre_model,
                    	samples = samples,
                    	epochs = n_epochs,
                    	hyperparams = hyperparams,
                    	weights = weights,
                		)
        _, source_feat, _, _ = semi_dirac.evaluate_source(graph_dl = samples["source_graph_ds"], return_lists_roc = True)
        target_feat, target_output, target_prob, target_pred, target_confs, target_mean_uncert = semi_dirac.evaluate_novel_target(graph_dl = samples["target_graph_dl"], return_lists_roc = True)
        if samples["test_graph_ds"] is not None:
            test_feat, _, test_pred = semi_dirac.evaluate_target(graph_dl = samples["test_graph_ds"], return_lists_roc = True)
        else:
            test_feat = None 
            test_pred = None
        results = {
                    "source_feat": source_feat,
                    "target_feat": target_feat,
                    "target_output": target_output,
                    "target_prob": target_prob,
                    "target_pred": target_pred,
                    "target_confs": target_confs,
                    "target_mean_uncert": target_mean_uncert,
                    "test_feat": test_feat,
                    "test_pred": test_pred,
                    } 
        return results
