import os
import time
import random
from typing import Callable, Iterable, Union, List, Tuple, Dict, Any
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
    """High-level API for multi-omics graph **integration**.

    This class prepares data (optionally with subgraph sampling), builds an
    integration model, trains it in an unsupervised manner, and returns
    embeddings/reconstructions.
    """

    def __init__(
        self,
        save_path: str = './Results/',
        subgraph: bool = True,
        use_gpu: bool = True,
        **kwargs,
        )-> None:
        """Initialize the integration app.

        Parameters
        ----------
        save_path : str, default './Results/'
            Directory to write outputs (figures, checkpoints, etc.). Must be writable.
        subgraph : bool, default True
            If ``True``, use ``ClusterData``/``ClusterLoader`` for sampling. If ``False``,
            use a full-batch ``DataLoader`` for small graphs.
        use_gpu : bool, default True
            If ``True``, selects ``cuda`` when available; otherwise CPU.
        **kwargs : Any
            Ignored; forwarded to ``super``.

        Side Effects
        ------------
        Sets ``self.device``, ``self.subgraph``, and ``self.save_path``.
        """
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
        edge_index,
        domain_list = None,
        batch = None,
        num_parts: int = 10,
        num_workers: int = 1,
        batch_size: int = 1,
    ):
        """Process multi-omics node features and construct a graph dataset.
        
        Parameters
        ----------
        dataset_list : list of (ndarray | torch.Tensor)
            List of feature matrices, one per modality/layer. Each element must
            be shaped ``(n_nodes, n_features_i)`` **(rows = nodes, cols = features)**.
        edge_index : torch.LongTensor
            Graph connectivity in COO format with shape ``(2, E)``. Will be
            made undirected via ``to_undirected``.
        domain_list : list[np.ndarray] | None, optional
            Optional per-modality integer domain labels of length ``n_nodes``.
            If ``None``, each dataset is treated as its own domain (0..n-1).
        batch : None | pandas.Series | np.ndarray | list, optional
            Optional per-node batch labels of length ``n_nodes``. Non-numeric
            labels are categorical-encoded. If ``None``, a zero vector is used
            for each modality.
        num_parts : int, default 10
            Number of partitions for ``ClusterData`` when ``self.subgraph=True``.
        num_workers : int, default 1
            Number of workers for the loaders.
        batch_size : int, default 1
            Batch size for ``ClusterLoader`` when ``self.subgraph=True``.

        Returns
        -------
        dict
            A dictionary with the following keys:
            - ``graph_ds`` : dict
                Underlying graph data object/dict from ``GraphDataset`` with
                additional modality tensors (e.g., ``data_1``, ``domain_1``, ``batch_1``...).
            - ``graph_dl`` : ClusterLoader | DataLoader
                A ``ClusterLoader`` if ``self.subgraph=True``; otherwise a full-batch
                ``DataLoader`` with a single item.
            - ``n_samples`` : int
                Number of input datasets/modalities.
            - ``n_inputs_list`` : list[int]
                Feature dimensions for each dataset ``[n_features_0, n_features_1, ...]``.
            - ``n_domains`` : int
                Number of unique domains inferred from ``domain_list``.

        Raises
        ------
        ValueError
            If node counts differ across ``dataset_list``; if ``batch`` length
            mismatches data; or an unsupported ``batch`` type is provided.

        Notes
        -----
        Sets ``self.n_samples``, ``self.n_inputs_list``, and ``self.num_domains``.
        Prints the number of unique domains detected.
        """
        # Store number of input datasets (omics layers)
        self.n_samples = len(dataset_list)
        
        # Validate consistent number of nodes across datasets
        def _n_nodes(x):
            return x.shape[0] if hasattr(x, "shape") else len(x)
        n_nodes = _n_nodes(dataset_list[0])  
        
        for idx, data in enumerate(dataset_list):
            if _n_nodes(data) != n_nodes:
                raise ValueError(
                    f"All datasets must have the same number of rows (nodes). "
                    f"dataset_list[0] has {n_nodes}, but dataset_list[{idx}] has {_n_nodes(data)}."
                )
        # Calculate number of unique domains
        if domain_list is None:
            # If no domain labels provided, treat each dataset as separate domain
            domain_list = [np.full(n_nodes, i, dtype=np.int64) for i in range(self.n_samples)]
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
                    domain=domain_list[i],
                    batch=batch_list[i],
                    edge_index=to_undirected(edge_index),  # Ensure undirected graph
                )
                graph_data = graph_ds.graph_data
            else:
                # Additional datasets are added as node features
                graph_data[f"data_{i}"] = torch.FloatTensor(data)
                graph_data[f"domain_{i}"] = torch.LongTensor(domain_list[i])
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
        """Build the integration model with the provided hyperparameters.

        Parameters
        ----------
        samples : dict
            Output from ``_get_data``. Must contain ``n_inputs_list`` and ``n_domains``.
        n_hiddens : int, default 128
            Hidden dimension for GNN layers.
        n_outputs : int, default 64
            Output/embedding dimension per node.
        opt_GNN : str, default 'GAT'
            GNN backbone option consumed by ``integrate_model``.
        dropout_rate : float, default 0.1
            Dropout rate inside the model.
        use_skip_connections : bool, default True
            Whether to enable residual/skip connections (if supported).
        use_attention : bool, default True
            Whether to use attention (if supported by the chosen backbone).
        n_attention_heads : int, default 4
            Number of attention heads (if applicable).
        use_layer_scale : bool, default False
            If ``True``, enable layer scale with initialization ``layer_scale_init``.
        layer_scale_init : float, default 1e-2
            Initialization value for layer scaling.
        use_stochastic_depth : bool, default False
            Enable stochastic depth.
        stochastic_depth_rate : float, default 0.1
            Drop probability for stochastic depth.
        combine_method : {'concat','sum','attention'}, default 'concat'
            How to combine multi-modal features inside the model.

        Returns
        -------
        models : Any
            The model instance returned by ``integrate_model(...)``, ready for training.
        """
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
        """Train the integration model and evaluate embeddings/reconstructions.

        Parameters
        ----------
        samples : dict
            Output from ``_get_data`` with keys like ``graph_ds``, ``graph_dl``,
            ``n_inputs_list``, ``n_domains``.
        models : Any
            Model returned by ``_get_model`` / ``integrate_model``.
        epochs : int, default 500
            Training epochs.
        optimizer_name : str, default 'adam'
            Optimizer identifier consumed by the trainer.
        lr : float, default 1e-3
            Learning rate.
        tau : float, default 0.9
            Momentum/EMA or contrastive temperature parameter (per trainer definition).
        wd : float, default 5e-2
            Weight decay.
        scheduler : bool, default True
            Whether to use a learning-rate scheduler.
        lamb : float, default 5e-4
            Loss coefficient used by the trainer.
        scale_loss : float, default 0.025
            Additional loss scaling used by the trainer.

        Returns
        -------
        data_z : torch.Tensor
            Node embeddings; typically shaped ``(n_nodes, n_outputs)``.
        combine_recon : Any
            Reconstruction(s) as returned by ``train_integrate.evaluate``; may be a
            tensor or a structure of tensors.
        """
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
    """High-level API for **annotation / domain adaptation** on graphs.

    Prepares labeled source (and unlabeled target) graphs, builds an annotation
    model, supports semi-supervised training, optional novel-class discovery,
    and evaluation on source/target/test.
    """

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
        """Process labeled source and (optional) unlabeled target into loaders.
        
        Parameters
        ----------
        source_data : (ndarray | torch.Tensor)
            Source node features with shape ``(n_source_nodes, n_features)``.
        source_label : (array-like)
            Source labels; numeric or categorical. Non-numeric labels are encoded
            to 0-based integer codes. A mapping is stored in ``self.pairs``.
        source_edge_index : torch.LongTensor
            COO connectivity for the source graph, shape ``(2, E_source)``; made
            undirected.
        target_data : (ndarray | torch.Tensor) or None
            Optional target node features with shape ``(n_target_nodes, n_features)``.
        target_edge_index : torch.LongTensor or None
            Optional COO connectivity for target graph, shape ``(2, E_target)``; made
            undirected if provided.
        source_domain : array-like[int] or None, default None
            Optional per-node domain labels for source. Defaults to zeros.
        target_domain : array-like[int] or None, default None
            Optional per-node domain labels for target. Defaults to ones when
            ``target_data`` is provided.
        test_data : (ndarray | torch.Tensor) or None, default None
            Optional test node features ``(n_test_nodes, n_features)``.
        test_edge_index : torch.LongTensor or None, default None
            Required if ``test_data`` is provided.
        weighted_classes : bool, default False
            If ``True``, compute inverse-frequency class weights for source labels.
        split_list : list[tuple[int,int]] or None, default None
            Optional feature splits for multi-modal inputs, e.g., ``[(0,1000),(1000,1500)]``.
        num_workers : int, default 1
            DataLoader workers for source/target loaders.
        batch_size : int, default 1
            Batch size for ``ClusterLoader``.
        num_parts_source : int, default 1
            ``ClusterData`` partitions for source graph.
        num_parts_target : int, default 1
            ``ClusterData`` partitions for target graph.

        Returns
        -------
        dict
            Contains:
            - ``source_graph_ds`` : dict
                Graph data object/dict for source (from ``GraphDataset_unpaired``).
            - ``source_graph_dl`` : ClusterLoader
                Loader over source clusters.
            - ``target_graph_ds`` : dict | None
                Graph data for target or ``None`` if no target.
            - ``target_graph_dl`` : ClusterLoader | None
                Loader for target or ``None`` if no target.
            - ``test_graph_ds`` : torch_geometric.data.Data | None
                Test graph object if both ``test_data`` and ``test_edge_index`` provided.
            - ``class_weight`` : torch.FloatTensor | None
                Class weights when ``weighted_classes=True``.
            - ``n_labels`` : int
                Number of unique labels in source.
            - ``n_inputs`` : int
                Feature dimension.
            - ``n_domains`` : int
                Number of domains inferred from ``source_domain``/``target_domain``.
            - ``split_list`` : list[tuple[int,int]] | None
                Echo of the provided ``split_list``.

        Notes
        -----
        If ``source_label`` is categorical, ``self.pairs`` stores a mapping
        ``{code: original_label}``; otherwise ``self.pairs`` is ``None``.
        Sets ``self.n_labels``, ``self.n_inputs``, and ``self.n_domains``.
        Prints the number of unique domains.
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
        """Build the annotation model (classifier/domain-adaptation).

        Parameters
        ----------
        samples : dict
            Output from ``annotate_app._get_data``; must include ``n_domains``,
            ``n_labels``, and either ``n_inputs`` (int) or ``split_list`` for
            multi-modal cases.
        n_hiddens : int, default 128
            Hidden dimension.
        n_outputs : int, default 64
            Embedding dimension before the classification head.
        opt_GNN : str, default 'SAGE'
            GNN backbone identifier consumed by ``annotate_model``.
        s : int, default 32
            Scale parameter for margin-based head (if applicable).
        m : float, default 0.10
            Margin parameter for margin-based head.
        easy_margin : bool, default False
            Use easy margin variant if supported.
        dropout_rate : float, default 0.1
            Dropout rate.
        use_skip_connections : bool, default False
            Enable skip/residual connections (if supported).
        use_attention : bool, default True
            Enable attention (if supported).
        n_attention_heads : int, default 2
            Number of attention heads when applicable.
        use_layer_scale : bool, default False
            Enable layer scaling.
        layer_scale_init : float, default 1e-2
            Initial value for layer scale.
        use_stochastic_depth : bool, default False
            Enable stochastic depth.
        stochastic_depth_rate : float, default 0.1
            Drop probability for stochastic depth.
        combine_method : {'concat','sum','attention'}, default 'concat'
            Feature fusion strategy for multi-modal inputs.

        Returns
        -------
        models : Any
            Model instance returned by ``annotate_model(...)``.
        """
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
        """Train the annotation model (semi-supervised/domain adaptation) and evaluate.

        Parameters
        ----------
        samples : dict
            Output from ``_get_data``. Expected keys include ``source_graph_ds``,
            ``source_graph_dl``, optional ``target_graph_dl`` and ``test_graph_ds``,
            and possibly ``class_weight``.
        models : Any
            Model returned by ``_get_model`` / ``annotate_model``.
        n_epochs : int, default 200
            Number of training epochs.
        optimizer_name : str, default 'adam'
            Optimizer identifier.
        lr : float, default 1e-3
            Learning rate.
        wd : float, default 5e-3
            Weight decay.
        scheduler : bool, default True
            Whether to enable learning-rate scheduling.
        filter_low_confidence : bool, default True
            If ``True``, mark predictions with confidence < ``confidence_threshold``
            as ``"unassigned"`` in the returned ``target_pred_filtered`` / ``test_pred_filtered``.
        confidence_threshold : float, default 0.5
            Confidence threshold in [0, 1].

        Returns
        -------
        dict
            With keys (some may be ``None`` if target/test are absent):
            ``source_feat``, ``target_feat``, ``target_output``, ``target_prob``,
            ``target_pred``, ``target_pred_filtered``, ``target_confs``,
            ``target_mean_uncert``, ``test_feat``, ``test_output``, ``test_prob``,
            ``test_pred``, ``test_pred_filtered``, ``test_confs``, ``test_mean_uncert``,
            ``pairs``, ``pairs_filter``, and ``low_confidence_threshold``.
        """
        def _filter_predictions_by_confidence(preds, confs):
            """Return 'unassigned' where confidence is below the threshold."""
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
        """Discover novel target classes and retrain with expanded label space.

        Workflow
        --------
        1. Build a temporary ``AnnData`` from target features, run neighbors + Louvain
           clustering (via Scanpy), then compute UMAP and save a PDF.
        2. Supervised pretrain on source for ``pre_epochs``.
        3. Estimate novel-class seeds; relabel target; rebuild loaders.
        4. Expand classifier to ``n_labels + num_novel_class`` and train for ``n_epochs``.

        Parameters
        ----------
        samples : dict
            Output from ``_get_data``; must include keys
            ``source_graph_ds``, ``source_graph_dl``, ``target_graph_ds``,
            ``target_graph_dl``, ``class_weight`` (optional), ``n_labels``, and
            feature sizes ``n_inputs``.
        minemodel : Any
            Initial annotation model (from ``_get_model``).
        num_novel_class : int, default 3
            Number of novel classes to discover in target.
        pre_epochs : int, default 100
            Supervised pretraining epochs on source.
        n_epochs : int, default 200
            Training epochs for the novel-phase.
        num_parts : int, default 30
            Number of partitions for the (new) target ``ClusterData``.
        resolution : float, default 1
            Louvain resolution for clustering.
        s : int, default 64
            Scale parameter for the (re)built model head.
        m : float, default 0.1
            Margin parameter for the (re)built model head.
        weights : dict, default {"alpha1":1, ..., "alpha8":1}
            Loss weights dictionary consumed by ``_train_novel``.

        Returns
        -------
        dict
            With keys: ``source_feat``, ``target_feat``, ``target_output``,
            ``target_prob``, ``target_pred``, ``target_confs``,
            ``target_mean_uncert``, ``test_feat``, ``test_pred``. (``test_*`` may be
            ``None`` if a test set is not provided.)
        """
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
