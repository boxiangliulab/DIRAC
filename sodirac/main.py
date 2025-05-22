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
from torch_geometric.data import InMemoryDataset, Data

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
        edge_index,
        batch = None,
        num_parts: int = 10,
        num_workers: int = 1,
        batch_size: int = 1,
    ):
        """Process multi-omics data and construct graph dataset.
        
        Args:
            dataset_list: List of omics data matrices (features x samples)
            domain_list: List of domain labels for each dataset
            edge_index: Graph connectivity in COO format (2 x num_edges)
            batch: Batch information (None, adata.obs['batch'] or np.array)
            num_parts: Number of partitions for subgraph sampling
            num_workers: Number of workers for data loading
            batch_size: Batch size for data loading
        
        Returns:
            Dictionary containing:
            - graph_ds: Processed graph data
            - graph_dl: Data loader for the graph
            - n_samples: Number of input datasets
            - n_inputs_list: List of feature dimensions for each dataset
            - n_domains: Number of unique domains
        """
        # Store number of input datasets (omics layers)
        self.n_samples = len(dataset_list)
        
        # Calculate number of unique domains
        if domain_list is None:
            # If no domain labels provided, treat each dataset as separate domain
            self.num_domains = len(dataset_list)
        else:
            # Find maximum domain index across all domain label arrays
            domains_max = [int(domain.max()) for domain in domain_list if domain is not None]
            # Number of domains is max index + 1 (assuming 0-based indexing)
            self.num_domains = max(domains_max) + 1 if domains_max else 1
        print(f"Found {self.num_domains} unique domains.")
        
        # Process batch information
        if batch is None:
            # Case 1: No batch information provided - create dummy batch labels (all 0)
            batch_size = len(dataset_list[0]) if dataset_list else 0
            batch_list = [np.zeros(batch_size, dtype=np.int64) for _ in range(self.n_samples)]
        elif hasattr(batch, 'values'):  
            # Case 2: Pandas Series input (e.g., adata.obs['batch'])
            # Convert to categorical codes (numerical representation)
            batch_values = batch.values
            categorical = pd.Categorical(batch_values)
            batch_list = [categorical.codes.astype(np.int64) for _ in range(self.n_samples)]
        elif isinstance(batch, (np.ndarray, list)):  
            # Case 3: Numpy array or Python list input
            batch_array = np.asarray(batch)
            if not np.issubdtype(batch_array.dtype, np.number):
                # Convert non-numeric batch labels to categorical codes
                categorical = pd.Categorical(batch_array)
                batch_list = [categorical.codes.astype(np.int64) for _ in range(self.n_samples)]
            else:
                # Use numerical batch labels directly
                batch_list = [batch_array.astype(np.int64) for _ in range(self.n_samples)]
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")
        
        # Validate batch dimensions match data
        for batch_arr in batch_list:
            if len(batch_arr) != len(dataset_list[0]):
                raise ValueError("Batch length does not match data length")
        
        # Initialize storage for graph data and feature dimensions
        self.n_inputs_list = []  # Will store feature dimensions for each dataset
        graph_data = {}  # Will store final graph data dictionary
        
        # Process each omics dataset
        for i, data in enumerate(dataset_list):
            # Store feature dimension for current dataset
            self.n_inputs_list.append(data.shape[1])
            
            if i == 0:
                # First dataset initializes the graph structure
                graph_ds = GraphDataset(
                    data=data,
                    domain=domain_list[i] if domain_list else None,
                    batch=batch_list[i],
                    edge_index=to_undirected(edge_index),  # Ensure undirected graph
                )
                graph_data = graph_ds.graph_data
            else:
                # Additional datasets are added as node features
                graph_data[f"data_{i}"] = torch.FloatTensor(data)
                graph_data[f"domain_{i}"] = torch.LongTensor(domain_list[i]) if domain_list else None
                graph_data[f"batch_{i}"] = torch.LongTensor(batch_list[i].copy())
        
        # Create appropriate data loader
        if self.subgraph:
            # For large graphs: use neighborhood sampling with ClusterData
            graph_dataset = ClusterData(graph_data, num_parts=num_parts, recursive=False)
            graph_dl = ClusterLoader(
                graph_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=num_workers
            )
        else:
            # For small graphs: use full-batch loading
            graph_dl = DataLoader([graph_data])
        
        # Return processed data and metadata
        return {
            "graph_ds": graph_data,  
            "graph_dl": graph_dl,     
            "n_samples": self.n_samples,  
            "n_inputs_list": self.n_inputs_list, 
            "n_domains": self.num_domains  
        }

    def _get_model(
        self,
        samples,
        n_hiddens: int = 128,
        n_outputs: int = 64,
        opt_GNN = "GAT",
        dropout_rate = 0.1,
        use_skip_connections = True,
        use_attention = True,
        n_attention_heads = 4,
        use_layer_scale = False,
        layer_scale_init = 1e-2,
        use_stochastic_depth = False,
        stochastic_depth_rate = 0.1,
        combine_method = 'concat',  # 'concat', 'sum', 'attention'
        ):
        ##### Build a transfer model to conver atac data to rna shape 
        models = integrate_model(n_inputs_list = samples["n_inputs_list"], 
                            n_domains = samples["n_domains"],
                            n_hiddens = n_hiddens,
                            n_outputs = n_outputs,
                            opt_GNN = opt_GNN,
                            dropout_rate = dropout_rate,
                            use_skip_connections = use_skip_connections,
                            use_attention = use_attention,
                            n_attention_heads = n_attention_heads,
                            use_layer_scale = use_layer_scale,
                            layer_scale_init = layer_scale_init,
                            use_stochastic_depth = use_stochastic_depth,
                            stochastic_depth_rate = stochastic_depth_rate,
                            combine_method = combine_method
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
        wd: float = 5e-2,
        scheduler: bool = True,
        lamb: float = 5e-4, 
        scale_loss: float = 0.025,
        ):
        ######### load all dataloaders and dist arrays
        hyperparams = unsuper_hyperparams(lr = lr, tau = tau, wd = wd, scheduler = scheduler)
        un_dirac = train_integrate(
                    minemodel = models,
                    save_path = self.save_path,
                    device = self.device,
                )

        un_dirac._train(
                    samples = samples,
                    epochs = epochs,
                    hyperparams = hyperparams,
                    optimizer_name = optimizer_name,
                    lamb = lamb, 
                    scale_loss = scale_loss
            )

        data_z, combine_recon = un_dirac.evaluate(samples = samples)

        return data_z, combine_recon

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
        weighted_classes = False,
        split_list = None,
        num_workers: int = 1,
        batch_size: int = 1,
        num_parts_source: int = 1,
        num_parts_target: int = 1,
    ):
        """Process and prepare graph data for domain adaptation training.
        
        Args:
            source_data: Features of source domain nodes
            source_label: Labels for source domain nodes
            source_edge_index: Edge connections for source graph
            target_data: Features of target domain nodes (optional)
            target_edge_index: Edge connections for target graph (optional)
            source_domain: Domain labels for source (optional)
            target_domain: Domain labels for target (optional)
            test_data: Test set features (optional)
            test_edge_index: Test set edges (optional)
            weighted_classes: Whether to apply class weighting
            num_workers: Workers for data loading
            batch_size: Batch size for training
            num_parts_source: Number of partitions for source graph
            num_parts_target: Number of partitions for target graph
            
        Returns:
            Dictionary containing processed datasets and metadata
        """
        
        # Calculate basic dataset properties       
        if not pd.api.types.is_numeric_dtype(source_label):
            categorical = pd.Categorical(source_label)
            source_label = np.asarray(categorical.codes, dtype=np.int64)
            self.pairs = dict(enumerate(categorical.categories))
        else:
            source_label = np.asarray(source_label, dtype=np.int64)
            self.pairs = None

        self.n_labels = len(np.unique(source_label))
        self.n_inputs = source_data.shape[1]
        
        # Handle domain label assignment
        # Default: source=0, target=1 when domains not specified
        if source_domain is None:
            source_domain = np.zeros(source_data.shape[0], dtype=np.int64)
        if target_domain is None and target_data is not None:
            target_domain = np.ones(target_data.shape[0], dtype=np.int64)
        
        # Determine number of unique domains
        if target_data is None:
            self.n_domains = 1  # Only source domain exists
        else:
            # Get maximum domain index from both domains
            source_max = int(source_domain.max())
            target_max = int(target_domain.max()) if target_domain is not None else 1
            self.n_domains = max(source_max, target_max) + 1  # +1 for zero-based indexing
            
        print(f"Identified {self.n_domains} unique domains.")
        
        # Calculate class weights for imbalanced datasets
        if weighted_classes:
            classes, counts = np.unique(source_label, return_counts=True)
            class_weights = (1.0 / (counts/counts.sum())) / (1.0 / (counts/counts.sum())).min()
            class_weight = torch.from_numpy(class_weights).float()
        else:
            class_weight = None
            
        # Prepare source domain dataset
        source_graph = GraphDataset_unpaired(
            data=source_data,
            domain=source_domain,
            edge_index=to_undirected(source_edge_index),
            label=source_label
        )
        source_clusters = ClusterData(
            source_graph.graph_data, 
            num_parts=num_parts_source, 
            recursive=False
        )
        source_loader = ClusterLoader(
            source_clusters, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
        
        # Prepare target domain dataset (if exists)
        target_graph = None
        target_loader = None
        if target_data is not None:
            target_graph = GraphDataset_unpaired(
                data=target_data,
                domain=target_domain,
                edge_index=to_undirected(target_edge_index),
                label=None  # Target domain is unlabeled
            )
            target_clusters = ClusterData(
                target_graph.graph_data,
                num_parts=num_parts_target,
                recursive=False
            )
            target_loader = ClusterLoader(
                target_clusters,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )
        
        # Prepare test dataset (if exists)
        test_graph = None
        if test_data is not None and test_edge_index is not None:
            test_graph = Data(
                data=torch.FloatTensor(test_data),
                edge_index=test_edge_index
            )
        
        return {
            "source_graph_ds": source_graph.graph_data,
            "source_graph_dl": source_loader,
            "target_graph_ds": target_graph.graph_data if target_graph else None,
            "target_graph_dl": target_loader,
            "test_graph_ds": test_graph,
            "class_weight": class_weight,
            "n_labels": self.n_labels,
            "n_inputs": self.n_inputs,
            "n_domains": self.n_domains,
            "split_list": split_list,
        }

    def _get_model(
        self,
        samples,
        n_hiddens: int = 128,
        n_outputs: int = 64,
        opt_GNN: str = "SAGE",
        s: int = 32,
        m: float = 0.10,
        easy_margin: bool = False,
        dropout_rate: float = 0.1,
        use_skip_connections: bool = False,
        use_attention: bool = True,
        n_attention_heads: int = 2,
        use_layer_scale: bool = False,
        layer_scale_init: float = 1e-2,
        use_stochastic_depth: bool = False,
        stochastic_depth_rate: float = 0.1,
        combine_method: str = 'concat',  # 'concat', 'sum', 'attention'
        ):
        ##### Build a transfer model to conver atac data to rna shape 
        # Handle multi-modal case
        if samples["split_list"] is not None:
            # Calculate input dimensions for each modality
            n_inputs = []
            for start, end in samples["split_list"]:
                n_inputs.append(end - start)
        else:
            # Single modality case
            n_inputs = samples["n_inputs"]
        
        models = annotate_model(
        				n_inputs= n_inputs, 
                        n_domains = samples["n_domains"], 
                        n_labels = samples["n_labels"], 
                        n_hiddens = n_hiddens,
                        n_outputs = n_outputs, 
                        opt_GNN = opt_GNN,
                        s = s,
                        m = m,
                        easy_margin = easy_margin,
                        dropout_rate = dropout_rate,
                        use_skip_connections = use_skip_connections,
                        use_attention = use_attention,
                        n_attention_heads = n_attention_heads,
                        use_layer_scale = use_layer_scale,
                        use_stochastic_depth = use_stochastic_depth,
                        stochastic_depth_rate = stochastic_depth_rate,
                        combine_method = combine_method,
                        input_split = samples["split_list"],
                        )
        self.n_outputs = n_outputs
        self.opt_GNN = opt_GNN
        self.n_hiddens = n_hiddens

        return models

    def _train_dirac_annotate(
            self,
            samples,
            models,
            n_epochs: int = 200,
            optimizer_name: str = "adam",
            lr: float = 1e-3,
            wd: float = 5e-3,
            scheduler: bool = True,
            filter_low_confidence: bool = True,
            confidence_threshold: float = 0.5,
        ):
        def _filter_predictions_by_confidence(preds, confs):
            return np.where(confs < confidence_threshold, "unassigned", preds)

        # Step 1: Initialize
        samples["n_outputs"] = self.n_outputs
        hyperparams = unsuper_hyperparams(lr=lr, wd=wd, scheduler=scheduler)

        semi_dirac = train_annotate(
            minemodel=models,
            save_path=self.save_path,
            device=self.device,
        )

        # Step 2: Train model
        semi_dirac._train(
            samples=samples,
            epochs=n_epochs,
            hyperparams=hyperparams,
            optimizer_name=optimizer_name,
        )

        # Step 3: Evaluate source
        _, source_feat, _, _ = semi_dirac.evaluate_source(
            graph_dl=samples["source_graph_ds"],
            return_lists_roc=True,
        )

        # Step 4: Evaluate target (novel)
        target_feat, target_output, target_prob, target_pred, target_confs, target_mean_uncert = semi_dirac.evaluate_novel_target(
            graph_dl=samples["target_graph_dl"],
            return_lists_roc=True,
        )
        target_pred_filtered = _filter_predictions_by_confidence(target_pred, target_confs) if filter_low_confidence else None

        # Step 5: Evaluate test set if available
        if samples.get("test_graph_ds") is not None:
            test_feat, test_output, test_prob, test_pred, test_confs, test_mean_uncert = semi_dirac.evaluate_target(
                graph_dl=samples["test_graph_ds"],
                return_lists_roc=True,
            )
            test_pred_filtered = _filter_predictions_by_confidence(test_pred, test_confs) if filter_low_confidence else None
        else:
            test_feat = test_output = test_prob = test_pred = test_confs = test_mean_uncert = test_pred_filtered = None

        if filter_low_confidence:
            pairs_filter = {str(k): v for k, v in self.pairs.items()}
            pairs_filter["unassigned"] = "unassigned"
        else:
            pairs_filter = None

        # Step 6: Package results
        results = {
            "source_feat": source_feat,
            "target_feat": target_feat,
            "target_output": target_output,
            "target_prob": target_prob,
            "target_pred": target_pred,
            "target_pred_filtered": target_pred_filtered,
            "target_confs": target_confs,
            "target_mean_uncert": target_mean_uncert,
            "test_feat": test_feat,
            "test_output": test_output,
            "test_prob": test_prob,
            "test_pred": test_pred,
            "test_pred_filtered": test_pred_filtered,
            "test_confs": test_confs,
            "test_mean_uncert": test_mean_uncert,
            "pairs": self.pairs,
            "pairs_filter": pairs_filter, 
            "low_confidence_threshold": confidence_threshold if filter_low_confidence else None,
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
