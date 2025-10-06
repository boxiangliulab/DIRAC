#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/17/23 2:58 PM
# @Author  : Chang Xu
# @File    : dataprep.py
# @Email   : changxu@nus.edu.sg

import logging
from typing import Union, Callable, Any, Iterable, List, Optional, Dict

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, Data


class GraphDS(Dataset):
    """
    PyTorch Dataset for single-cell/spatial profiles with optional labels and domains.

    Parameters
    ----------
    counts : np.ndarray or sparse.csr_matrix
        Shape [cells, genes]. Expression/count matrix.
    labels : np.ndarray or sparse.csr_matrix, optional
        Shape [cells,]. Integer cell-type labels.
    domains : np.ndarray or sparse.csr_matrix, optional
        Shape [cells,]. Integer domain labels.
    transform : Callable, optional
        Callable applied to each sample dict.
    num_domains : int, optional
        Total number of domains for one-hot encoding of `domains`. Default: -1.

    Returns
    -------
    None

    Notes
    -----
    - Dense copies are created for input arrays when needed.
    - One-hot encodings are produced for labels/domains when provided.
    """

    def __init__(
        self,
        counts: Union[sparse.csr.csr_matrix, np.ndarray],
        labels: Union[sparse.csr.csr_matrix, np.ndarray] = None,
        domains: Union[sparse.csr.csr_matrix, np.ndarray] = None,
        transform: Callable = None,
        num_domains: int = -1,
    ) -> None:
        super(GraphDS, self).__init__()

        # type checks
        if type(counts) not in (np.ndarray, sparse.csr_matrix):
            msg = f"Counts is type {type(counts)}, must `np.ndarray` or `sparse.csr_matrix`"
            raise TypeError(msg)

        # densify counts if needed
        counts = counts.toarray() if sparse.issparse(counts) else counts
        self.counts = torch.FloatTensor(counts)

        self.labels = self._process_labels(labels)
        self.domains = self._process_domains(domains, num_domains)
        self.transform = transform
        self.indexes = torch.arange(self.counts.shape[0]).long()

    def _process_labels(
        self, labels: Optional[Union[np.ndarray, sparse.csr_matrix]]
    ) -> tuple:
        """
        Convert labels to torch tensors and one-hot encodings.

        Parameters
        ----------
        labels : np.ndarray or sparse.csr_matrix, optional
            Shape [cells,]. Integer labels.

        Returns
        -------
        (labels_tensor, one_hot) : Tuple[torch.LongTensor or None, torch.FloatTensor or None]
            Dense label tensor and one-hot tensor, or (None, None) if `labels` is None.

        Notes
        -----
        One-hot dimension equals the number of unique labels in the batch.
        """
        if labels is not None:
            if not isinstance(labels, (np.ndarray, sparse.csr_matrix)):
                raise TypeError(
                    f"Labels is type {type(labels)}, must be `np.ndarray` or `sparse.csr_matrix`"
                )
            labels = labels.toarray() if sparse.issparse(labels) else labels
            labels = torch.from_numpy(labels).long()
            labels_one_hot = torch.nn.functional.one_hot(
                labels, num_classes=len(torch.unique(labels))
            ).float()
            return labels, labels_one_hot
        return None, None

    def _process_domains(
        self, domains: Optional[Union[np.ndarray, sparse.csr_matrix]], num_domains: int
    ) -> tuple:
        """
        Convert domain labels to torch tensors and one-hot encodings.

        Parameters
        ----------
        domains : np.ndarray or sparse.csr_matrix, optional
            Shape [cells,]. Integer domain labels.
        num_domains : int
            Number of domain categories for one-hot encoding.

        Returns
        -------
        (domains_tensor, one_hot) : Tuple[torch.LongTensor or None, torch.FloatTensor or None]
            Dense domain tensor and one-hot tensor, or (None, None) if `domains` is None.
        """
        if domains is not None:
            if not isinstance(domains, (np.ndarray, sparse.csr_matrix)):
                raise TypeError(
                    f"Domains is type {type(domains)}, must be `np.ndarray` or `sparse.csr_matrix`"
                )
            domains = domains.toarray() if sparse.issparse(domains) else domains
            domains = torch.from_numpy(domains).long()
            domains_one_hot = torch.nn.functional.one_hot(domains, num_classes=num_domains).float()
            return domains, domains_one_hot
        return None, None

    def __len__(self) -> int:
        """
        Number of examples in the dataset.

        Returns
        -------
        n : int
            Dataset length.
        """
        return self.counts.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve a single sample with optional labels/domains.

        Parameters
        ----------
        idx : int
            Index in `range(len(self))`.

        Returns
        -------
        sample : dict
            {
              "input": torch.FloatTensor,           # feature vector
              "idx": torch.LongTensor,              # original index
              "output": torch.LongTensor,           # optional label
              "output_one_hot": torch.FloatTensor,  # optional label one-hot
              "domain": torch.LongTensor,           # optional domain label
              "domain_one_hot": torch.FloatTensor,  # optional domain one-hot
            }

        Notes
        -----
        Applies `self.transform(sample)` if a transform is provided.
        """
        if not isinstance(idx, int):
            raise TypeError(f"indices must be int, you passed {type(idx)}, {idx}")
        if idx < 0 or idx >= len(self):
            raise ValueError(f"idx {idx} is invalid for dataset with {len(self)} examples.")

        input_ = self.counts[idx, ...]
        sample: Dict[str, torch.Tensor] = {"input": input_, "idx": self.indexes[idx]}

        if self.labels is not None:
            sample["output"] = self.labels[0][idx]
            sample["output_one_hot"] = self.labels[1][idx]

        if self.domains is not None:
            sample["domain"] = self.domains[0][idx]
            sample["domain_one_hot"] = self.domains[1][idx]

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def balance_classes(
    y: np.ndarray,
    class_min: int = 256,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Balance class indices by undersampling majorities and oversampling minorities.

    Parameters
    ----------
    y : np.ndarray
        Shape [N,]. Class labels.
    class_min : int, default 256
        Minimum examples per class after balancing.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    balanced_idx : np.ndarray
        Balanced indices (with replacement for minority classes).

    Notes
    -----
    The smallest effective class count used is `max(min_count, class_min)`.
    """
    if not isinstance(y, np.ndarray):
        raise TypeError(f"y should be a numpy array, but got {type(y)}")

    if not isinstance(class_min, int) or class_min <= 0:
        raise ValueError(f"class_min should be a positive integer, but got {class_min}")

    if random_state is not None:
        np.random.seed(random_state)

    classes, counts = np.unique(y, return_counts=True)
    min_count = np.min(counts)
    min_count = max(min_count, class_min)

    balanced_idx: List[np.ndarray] = []
    for cls, count in zip(classes, counts):
        class_idx = np.where(y == cls)[0].astype(int)
        oversample = count < min_count
        if oversample:
            print(f"Class {cls} has {count} samples. Oversampling to {min_count} samples.")
        sampled_idx = np.random.choice(class_idx, size=min_count, replace=oversample)
        balanced_idx.append(sampled_idx)

    balanced_idx = np.concatenate(balanced_idx).astype(int)
    return balanced_idx


class GraphDataset(InMemoryDataset):
    """
    In-memory PyG dataset for a *paired* graph with features, batches, domains, and labels.

    Parameters
    ----------
    data : np.ndarray
        Shape [num_nodes, num_features]. Node features.
    batch : np.ndarray
        Shape [num_nodes]. Batch assignment per node.
    domain : np.ndarray
        Shape [num_nodes]. Domain labels per node.
    edge_index : torch.Tensor
        Shape [2, num_edges]. Edge index.
    label : np.ndarray, optional
        Shape [num_nodes]. Node labels. Default: None.
    transform : callable, optional
        A callable that takes and returns a `torch_geometric.data.Data` object.

    Attributes
    ----------
    graph_data : torch_geometric.data.Data
        Graph data object with fields:
        - data_0 (FloatTensor), batch_0 (LongTensor), domain_0 (LongTensor),
          edge_index (Tensor), idx (LongTensor), label (LongTensor or None),
          num_nodes (int).

    Notes
    -----
    This dataset contains a single graph (length = 1).
    """

    def __init__(
        self,
        data: np.ndarray,
        batch: np.ndarray,
        domain: np.ndarray,
        edge_index: torch.Tensor,
        label: np.ndarray = None,
        transform: Callable = None,
    ):
        self.root = "."  # customizable
        super(GraphDataset, self).__init__(self.root, transform)

        # type checks
        if not isinstance(data, np.ndarray):
            raise TypeError(f"data should be of type np.ndarray, but got {type(data)}")
        if not isinstance(batch, np.ndarray):
            raise TypeError(f"batch should be of type np.ndarray, but got {type(batch)}")
        if not isinstance(domain, np.ndarray):
            raise TypeError(f"domain should be of type np.ndarray, but got {type(domain)}")
        if not isinstance(edge_index, torch.Tensor):
            raise TypeError(f"edge_index should be of type torch.Tensor, but got {type(edge_index)}")
        if label is not None and not isinstance(label, np.ndarray):
            raise TypeError(f"label should be of type np.ndarray, but got {type(label)}")

        self.graph_data = Data(
            data_0=torch.FloatTensor(data.copy()),
            batch_0=torch.LongTensor(batch.copy()),
            edge_index=edge_index,
            idx=torch.LongTensor(np.arange(data.shape[0])),
            domain_0=torch.LongTensor(domain.copy()),
            label=None if label is None else torch.LongTensor(label),
            num_nodes=data.shape[0],
        )

    def __len__(self) -> int:
        """
        Number of graphs in the dataset.

        Returns
        -------
        n : int
            Always 1 for `InMemoryDataset` here.
        """
        return 1

    def __getitem__(self, idx: int) -> Data:
        """
        Retrieve the single stored graph.
        Parameters
        ----------
        idx : int
            Graph index (must be 0).
        Returns
        -------
        graph_data : torch_geometric.data.Data
            The stored graph object.
        Raises
        ------
        IndexError
            If `idx != 0`.
        """
        if idx != 0:
            raise IndexError("Index out of range. This dataset contains only one graph.")
        return self.graph_data


class GraphDataset_unpaired(InMemoryDataset):
    """
    In-memory PyG dataset for an *unpaired* graph with features, domains, and labels.

    Parameters
    ----------
    data : np.ndarray
        Shape [num_nodes, num_features]. Node features.
    domain : np.ndarray
        Shape [num_nodes]. Domain labels per node.
    edge_index : torch.Tensor
        Shape [2, num_edges]. Edge index.
    label : np.ndarray, optional
        Shape [num_nodes]. Node labels. Default: None.
    transform : callable, optional
        A callable that takes and returns a `torch_geometric.data.Data` object.

    Attributes
    ----------
    graph_data : torch_geometric.data.Data
        Graph data object with fields:
        - data (FloatTensor), domain (LongTensor),
          edge_index (Tensor), idx (LongTensor), label (LongTensor or None),
          num_nodes (int).

    Notes
    -----
    This dataset contains a single graph (length = 1).
    """

    def __init__(
        self,
        data: np.ndarray,
        domain: np.ndarray,
        edge_index: torch.Tensor,
        label: np.ndarray = None,
        transform: Callable = None,
    ):
        self.root = "."  # customizable
        super(GraphDataset_unpaired, self).__init__(self.root, transform)

        # type checks
        if not isinstance(data, np.ndarray):
            raise TypeError(f"data should be of type np.ndarray, but got {type(data)}")
        if not isinstance(domain, np.ndarray):
            raise TypeError(f"domain should be of type np.ndarray, but got {type(domain)}")
        if not isinstance(edge_index, torch.Tensor):
            raise TypeError(f"edge_index should be of type torch.Tensor, but got {type(edge_index)}")
        if label is not None and not isinstance(label, np.ndarray):
            raise TypeError(f"label should be of type np.ndarray, but got {type(label)}")

        self.graph_data = Data(
            data=torch.FloatTensor(data.copy()),
            edge_index=edge_index,
            idx=torch.LongTensor(np.arange(data.shape[0])),
            domain=torch.LongTensor(domain.copy()),
            label=None if label is None else torch.LongTensor(label),
            num_nodes=data.shape[0],
        )

    def __len__(self) -> int:
        """
        Number of graphs in the dataset.
        Returns
        -------
        n : int
            Always 1 for `InMemoryDataset` here.
        """
        return 1

    def __getitem__(self, idx: int) -> Data:
        """
        Retrieve the single stored graph.

        Parameters
        ----------
        idx : int
            Graph index (must be 0).

        Returns
        -------
        graph_data : torch_geometric.data.Data
            The stored graph object.

        Raises
        ------
        IndexError
            If `idx != 0`.
        """
        if idx != 0:
            raise IndexError("Index out of range. This dataset contains only one graph.")
        return self.graph_data
