#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/17/23 2:58 PM
# @Author  : Chang Xu
# @File    : dataprep.py
# @Email   : changxu@nus.edu.sg

import torch
import numpy as np
from scipy import sparse
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, Data
from typing import Union, Callable, Any, Iterable, List, Optional
import logging


class GraphDS(Dataset):
    def __init__(
        self,
        counts: Union[sparse.csr.csr_matrix, np.ndarray],
        labels: Union[sparse.csr.csr_matrix, np.ndarray] = None,
        domains: Union[sparse.csr.csr_matrix, np.ndarray] = None,
        transform: Callable = None,
        num_domains: int = -1,
    ) -> None:
        r"""
        Load spatial multi-omics profiles.

        Parameters
        ----------
        counts : np.ndarray or sparse.csr_matrix
            [Cells, Genes] expression count matrix.
        labels : np.ndarray or sparse.csr_matrix, optional
            [Cells,] integer cell type labels.
        domains : np.ndarray or sparse.csr_matrix, optional
            [Cells,] integer domain labels.
        transform : Callable, optional
            Transformation to apply to samples.
        num_domains : int, optional
            Total number of domains for the task.

        Returns
        -------
        None
        """
        super(GraphDS, self).__init__()

        # check types on input arrays
        if type(counts) not in (
            np.ndarray,
            sparse.csr_matrix,
        ):
            msg = f"Counts is type {type(counts)}, must `np.ndarray` or `sparse.csr_matrix`"
            raise TypeError(msg)
        # if the corresponding vectors are sparse, convert them to dense
        # we perform this operation on a samplewise-basis to avoid
        # storing the whole count matrix in dense format
        counts = counts.toarray() if sparse.issparse(counts) else counts
        
        self.counts = torch.FloatTensor(counts)

        self.labels = self._process_labels(labels)
        self.domains = self._process_domains(domains, num_domains)
        self.transform = transform
        self.indexes = torch.arange(self.counts.shape[0]).long() 


    def _process_labels(self, labels):
        if labels is not None:
            if not isinstance(labels, (np.ndarray, sparse.csr_matrix)):
                raise TypeError(f"Labels is type {type(labels)}, must be `np.ndarray` or `sparse.csr_matrix`")
            labels = labels.toarray() if sparse.issparse(labels) else labels
            labels = torch.from_numpy(labels).long()
            labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=len(torch.unique(labels))).float()
            return labels, labels_one_hot
        return None, None

    def _process_domains(self, domains, num_domains):
        if domains is not None:
            if not isinstance(domains, (np.ndarray, sparse.csr_matrix)):
                raise TypeError(f"Domains is type {type(domains)}, must be `np.ndarray` or `sparse.csr_matrix`")
            domains = domains.toarray() if sparse.issparse(domains) else domains
            domains = torch.from_numpy(domains).long()
            domains_one_hot = torch.nn.functional.one_hot(domains, num_classes=num_domains).float()
            return domains, domains_one_hot
        return None, None

    def __len__(
        self,
    ) -> int:
        """Return the number of examples in the data set."""
        return self.counts.shape[0]

    def __getitem__(
        self,
        idx: int,
    ) -> dict:
        """Get a single cell expression profile and corresponding label.

        Parameters
        ----------
        idx : int
            index value in `range(len(self))`.

        Returns
        -------
        sample : dict
            'input' - torch.FloatTensor, input vector
            'output' - torch.LongTensor, target label
        """
        if not isinstance(idx, int):
            raise TypeError(f"indices must be int, you passed {type(idx)}, {idx}")

        if idx < 0 or idx >= len(self):
            raise ValueError(f"idx {idx} is invalid for dataset with {len(self)} examples.")

        input_ = self.counts[idx, ...]
        sample = {"input": input_, "idx": self.indexes[idx]}

        if self.labels is not None:
            sample["output"] = self.labels[0][idx]
            sample["output_one_hot"] = self.labels[1][idx]

        if self.domains is not None:
            sample["domain"] = self.domains[0][idx]
            sample["domain_one_hot"] = self.domains[1][idx]

        # if a transformer was supplied, apply transformations
        # to the sample vector and label
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def balance_classes(
    y: np.ndarray,
    class_min: int = 256,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Perform class balancing by undersampling majority classes
    and oversampling minority classes, down to a minimum value.

    Parameters
    ----------
    y : np.ndarray
        Class assignment indices.
    class_min : int
        Minimum number of examples to use for a class.
        Below this value, minority classes will be oversampled
        with replacement.
    random_state : Optional[int]
        Seed for the random number generator for reproducibility.

    Returns
    -------
    balanced_idx : np.ndarray
        Indices for balanced classes. Some indices may be repeated.
    """
    # Check if y is a numpy array
    if not isinstance(y, np.ndarray):
        raise TypeError(f"y should be a numpy array, but got {type(y)}")

    # Check if class_min is a positive integer
    if not isinstance(class_min, int) or class_min <= 0:
        raise ValueError(f"class_min should be a positive integer, but got {class_min}")

    # Set random seed for reproducibility if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Determine the size of the smallest class
    classes, counts = np.unique(y, return_counts=True)
    min_count = np.min(counts)
    min_count = max(min_count, class_min)

    # Generate indices with equal representation of each class
    balanced_idx = []
    for cls, count in zip(classes, counts):
        class_idx = np.where(y == cls)[0].astype(int)
        oversample = count < min_count
        if oversample:
            # Log the oversampling information
            print(f"Class {cls} has {count} samples. Oversampling to {min_count} samples.")
        sampled_idx = np.random.choice(class_idx, size=min_count, replace=oversample)
        balanced_idx.append(sampled_idx)
    
    balanced_idx = np.concatenate(balanced_idx).astype(int)
    return balanced_idx



# class GraphDataset(InMemoryDataset):
#     def __init__(
#         self, 
#         data, 
#         batch,
#         domain, 
#         edge_index,
#         label = None,
#         transform = None,
#         ):
#         self.root = '.'
#         super(GraphDataset, self).__init__(self.root, transform)
#         self.graph_data = Data(data_0 = torch.FloatTensor(data), 
#                                batch_0 = torch.LongTensor(batch),
#                                edge_index = edge_index, 
#                                idx = torch.LongTensor(np.arange(data.shape[0])),
#                                domain_0 = torch.LongTensor(domain),
#                                label = None if label is None else torch.LongTensor(label),)
#     def __len__(self):
#         return 1

#     def __getitem__(self, idx):
#         return self.graph_data

# class GraphDataset_unpaired(InMemoryDataset):
#     def __init__(
#         self, 
#         data, 
#         domain, 
#         edge_index, 
#         label = None,
#         transform = None,
#         ):
#         self.root = '.'
#         super(GraphDataset_unpaired, self).__init__(self.root, transform)
#         self.graph_data = Data(data = torch.FloatTensor(data), 
#                                edge_index = edge_index, 
#                                idx = torch.LongTensor(np.arange(data.shape[0])),
#                                domain = torch.LongTensor(domain),
#                                label = None if label is None else torch.LongTensor(label),)
#     def __len__(self):
#         return 1

#     def __getitem__(self, idx):
#         return self.graph_data





class GraphDataset(InMemoryDataset):
    """
    Dataset class for loading graph data with associated labels and domains.

    Parameters
    ----------
    data : np.ndarray
        Feature matrix with shape [num_nodes, num_features].
    batch : np.ndarray
        Batch vector with shape [num_nodes], indicating the batch assignment of each node.
    domain : np.ndarray
        Domain labels with shape [num_nodes].
    edge_index : torch.Tensor
        Graph edge indices with shape [2, num_edges].
    label : np.ndarray, optional
        Node labels with shape [num_nodes]. Default is None.
    transform : callable, optional
        A function/transform that takes in an `torch_geometric.data.Data` object and returns a transformed version.

    Attributes
    ----------
    graph_data : torch_geometric.data.Data
        The graph data object containing features, edge indices, labels, and domains.
    """

    def __init__(
        self, 
        data: np.ndarray, 
        batch: np.ndarray,
        domain: np.ndarray, 
        edge_index: torch.Tensor,
        label: np.ndarray = None,
        transform: callable = None,
    ):
        self.root = '.'  # This can be customized as needed
        super(GraphDataset, self).__init__(self.root, transform)

        # Type checks and conversions
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

        self.graph_data = Data(data_0 = torch.FloatTensor(data.copy()), 
            				batch_0 = torch.LongTensor(batch.copy()),
            				edge_index = edge_index, 
            				idx = torch.LongTensor(np.arange(data.shape[0])),
            				domain_0 = torch.LongTensor(domain.copy()),
            				label = None if label is None else torch.LongTensor(label),
                            num_nodes = data.shape[0],
        					)

    def __len__(self):
        """
        Return the number of graphs in the dataset. For an InMemoryDataset, this is typically 1.
        """
        return 1

    def __getitem__(self, idx):
        """
        Retrieve the graph data.

        Parameters
        ----------
        idx : int
            Index of the graph to retrieve.

        Returns
        -------
        graph_data : torch_geometric.data.Data
            The graph data object.
        """
        if idx != 0:
            raise IndexError("Index out of range. This dataset contains only one graph.")
        return self.graph_data



class GraphDataset_unpaired(InMemoryDataset):
    """
    Dataset class for loading unpaired graph data with associated labels and domains.

    Parameters
    ----------
    data : np.ndarray
        Feature matrix with shape [num_nodes, num_features].
    domain : np.ndarray
        Domain labels with shape [num_nodes].
    edge_index : torch.Tensor
        Graph edge indices with shape [2, num_edges].
    label : np.ndarray, optional
        Node labels with shape [num_nodes]. Default is None.
    transform : callable, optional
        A function/transform that takes in an `torch_geometric.data.Data` object and returns a transformed version.

    Attributes
    ----------
    graph_data : torch_geometric.data.Data
        The graph data object containing features, edge indices, labels, and domains.
    """

    def __init__(
        self, 
        data: np.ndarray, 
        domain: np.ndarray, 
        edge_index: torch.Tensor, 
        label: np.ndarray = None,
        transform: callable = None,
    ):
        self.root = '.'  # This can be customized as needed
        super(GraphDataset_unpaired, self).__init__(self.root, transform)

        # Type checks and conversions
        if not isinstance(data, np.ndarray):
            raise TypeError(f"data should be of type np.ndarray, but got {type(data)}")
        if not isinstance(domain, np.ndarray):
            raise TypeError(f"domain should be of type np.ndarray, but got {type(domain)}")
        if not isinstance(edge_index, torch.Tensor):
            raise TypeError(f"edge_index should be of type torch.Tensor, but got {type(edge_index)}")
        if label is not None and not isinstance(label, np.ndarray):
            raise TypeError(f"label should be of type np.ndarray, but got {type(label)}")

        self.graph_data = Data(
            				data = torch.FloatTensor(data.copy()), 
            				edge_index = edge_index, 
            				idx = torch.LongTensor(np.arange(data.shape[0])),
            				domain = torch.LongTensor(domain.copy()),
            				label = None if label is None else torch.LongTensor(label),
                            num_nodes = data.shape[0],
        				)

    def __len__(self):
        """
        Return the number of graphs in the dataset. For an InMemoryDataset, this is typically 1.
        """
        return 1

    def __getitem__(self, idx):
        """
        Retrieve the graph data.

        Parameters
        ----------
        idx : int
            Index of the graph to retrieve.

        Returns
        -------
        graph_data : torch_geometric.data.Data
            The graph data object.
        """
        if idx != 0:
            raise IndexError("Index out of range. This dataset contains only one graph.")
        return self.graph_data








