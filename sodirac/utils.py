#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/17/23 2:58 PM
# @Author  : Chang Xu
# @File    : utils.py
# @Email   : changxu@nus.edu.sg

import os
import sys
import random
import logging
from builtins import range
from typing import Union, Callable, Any, Iterable, List, Optional, Tuple, Sequence, Dict

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn
import torch
import tqdm
from scipy import sparse, stats
from sklearn.metrics import pairwise_distances, calinski_harabasz_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph
from torch_geometric.nn import knn_graph, radius_graph

EdgeList = List[Tuple[int, int]]


def append_categorical_to_data(
    X: Union[np.ndarray, sparse.csr.csr_matrix],
    categorical: np.ndarray,
) -> Tuple[Union[np.ndarray, sparse.csr.csr_matrix], np.ndarray]:
    """
    Append a one-hot encoding of a categorical vector to each sample in `X`.

    Parameters
    ----------
    X : np.ndarray or sparse.csr_matrix
        Shape [cells, features]. Feature matrix.
    categorical : np.ndarray
        Shape [cells,]. Categorical labels per cell.

    Returns
    -------
    Xa : np.ndarray or sparse.csr_matrix
        Shape [cells, features + n_categories]. Matrix with one-hot appended.
    categories : np.ndarray
        Shape [n_categories,]. Category names in the order used for one-hot.

    Examples
    --------
    >>> X_aug, cats = append_categorical_to_data(X, adata.obs["batch"].values)

    Notes
    -----
    Uses `pd.Categorical(...).codes` to derive integer label indices, then a
    one-hot encoding (via `make_one_hot`) that is concatenated to `X`.
    """
    labels = pd.Categorical(categorical)
    idx = np.array(labels.codes)
    idx = torch.from_numpy(idx.astype("int32")).long()
    categories = np.array(labels.categories)

    one_hot_mat = make_one_hot(idx, C=len(categories)).numpy()
    assert X.shape[0] == one_hot_mat.shape[0], f"dims unequal at {X.shape[0]}, {one_hot_mat.shape[0]}"

    if sparse.issparse(X):
        X = sparse.hstack([X, one_hot_mat])
    else:
        X = np.concatenate([X, one_hot_mat], axis=1)
    return X, categories


def get_adata_asarray(
    adata: anndata.AnnData,
) -> Union[np.ndarray, sparse.csr.csr_matrix]:
    """
    Materialize `adata.X` as an array or CSR matrix (no view).

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with `.X` of shape [cells, genes].

    Returns
    -------
    X : np.ndarray or sparse.csr_matrix
        Concrete in-memory copy of `adata.X` with matching type.

    Notes
    -----
    Preserves the dense/sparse form of the original `.X`.
    """
    if sparse.issparse(adata.X):
        X = sparse.csr.csr_matrix(adata.X)
    else:
        X = np.array(adata.X)
    return X


def build_classification_matrix(
    X: Union[np.ndarray, sparse.csr.csr_matrix],
    model_genes: np.ndarray,
    sample_genes: np.ndarray,
    gene_batch_size: int = 512,
) -> Union[np.ndarray, sparse.csr.csr_matrix]:
    """
    Reindex a count matrix to the model's gene order, filling missing genes with zeros.

    Parameters
    ----------
    X : np.ndarray or sparse.csr_matrix
        Shape [cells, genes]. Count matrix for the sample.
    model_genes : np.ndarray
        Expected gene identifiers in model order.
    sample_genes : np.ndarray
        Gene identifiers (columns) for `X`.
    gene_batch_size : int, default 512
        Number of genes to copy per batch (speed vs. memory trade-off).

    Returns
    -------
    N : np.ndarray or sparse.csr_matrix
        Shape [cells, len(model_genes)]. Reindexed counts; zeros for absent genes.

    Notes
    -----
    If `model_genes` exactly matches `sample_genes`, returns `X` unchanged.
    Otherwise, constructs a new matrix with columns in model order and copies
    overlapping genes in batches to control memory usage.
    """
    if type(X) not in (np.ndarray, sparse.csr.csr_matrix):
        raise TypeError(f"X is type {type(X)}, must be `np.ndarray` or `sparse.csr_matrix`")

    n_cells = X.shape[0]
    if len(model_genes) == len(sample_genes) and np.all(model_genes == sample_genes):
        print("Gene names match exactly, returning input.")
        return X

    if isinstance(X, np.ndarray):
        N = np.zeros((n_cells, len(model_genes)))
    else:
        N = sparse.lil_matrix((n_cells, len(model_genes)))

    model_genes_indices = []
    sample_genes_indices = []
    common_genes = 0
    for i, g in tqdm.tqdm(enumerate(sample_genes), desc="mapping genes"):
        if np.sum(g == model_genes) > 0:
            model_genes_indices.append(int(np.where(g == model_genes)[0]))
            sample_genes_indices.append(i)
            common_genes += 1

    gene_idx = 0
    n_batches = int(np.ceil(len(model_genes_indices) / gene_batch_size))
    for _ in tqdm.tqdm(range(n_batches), desc="copying gene batches"):
        model_batch_idx = model_genes_indices[gene_idx: gene_idx + gene_batch_size]
        sample_batch_idx = sample_genes_indices[gene_idx: gene_idx + gene_batch_size]
        if len(model_batch_idx) == 0:
            break
        N[:, model_batch_idx] = X[:, sample_batch_idx]
        gene_idx += gene_batch_size

    if sparse.issparse(N):
        N = sparse.csr_matrix(N)

    print(f"Found {common_genes} common genes.")
    return N


def knn_smooth_pred_class(
    X: np.ndarray,
    pred_class: np.ndarray,
    grouping: Optional[np.ndarray] = None,
    k: int = 15,
) -> np.ndarray:
    """
    Smooth class labels by majority vote among k-nearest neighbors.

    Parameters
    ----------
    X : np.ndarray
        Shape [N, features]. Embedding used for neighbor search.
    pred_class : np.ndarray
        Shape [N,]. Class labels to be smoothed.
    grouping : np.ndarray, optional
        Shape [N,]. Group IDs restricting neighbors to within-group only. If
        None, all cells are considered a single group.
    k : int, default 15
        Number of neighbors to use.

    Returns
    -------
    smooth_pred_class : np.ndarray
        Shape [N,]. Smoothed class labels.

    Notes
    -----
    For each group (or globally), builds a kNN graph and assigns to each cell
    the majority class among its k neighbors (including or excluding itself
    depending on scikit-learn defaults used here).
    """
    if grouping is None:
        grouping = np.zeros(X.shape[0])

    smooth_pred_class = np.zeros_like(pred_class)
    for group in np.unique(grouping):
        group_idx = np.where(grouping == group)[0].astype("int")
        X_group = X[grouping == group, :]
        k_use = min(k, X_group.shape[0])

        nns = NearestNeighbors(n_neighbors=k_use).fit(X_group)
        _, idx = nns.kneighbors(X_group)

        for i in range(X_group.shape[0]):
            classes = pred_class[group_idx[idx[i, :]]]
            uniq_classes, counts = np.unique(classes, return_counts=True)
            maj_class = uniq_classes[int(np.argmax(counts))]
            smooth_pred_class[group_idx[i]] = maj_class
    return smooth_pred_class


def knn_smooth_pred_class_prob(
    X: np.ndarray,
    pred_probs: np.ndarray,
    names: np.ndarray,
    grouping: Optional[np.ndarray] = None,
    k: Union[Callable[[int], int], int] = 15,
    dm: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Smooth class probabilities by kNN regression with RBF distance weights.

    Parameters
    ----------
    X : np.ndarray
        Shape [N, features]. Embedding used for neighbor search.
    pred_probs : np.ndarray
        Shape [N, C]. Class prediction probabilities per cell.
    names : np.ndarray
        Shape [C,]. Class names corresponding to columns of `pred_probs`.
    grouping : np.ndarray, optional
        Shape [N,]. Group IDs restricting neighbors to within-group only. If
        None, all cells are considered a single group.
    k : Callable[[int], int] or int, default 15
        If callable, receives the group size and returns k for that group;
        otherwise a fixed k is used.
    dm : np.ndarray, optional
        Shape [N, N]. Precomputed distance matrix to set the RBF kernel
        parameter efficiently.
    **kwargs : Any
        Additional kwargs forwarded to `KNeighborsRegressor`.

    Returns
    -------
    smooth_pred_class : np.ndarray
        Shape [N,]. Class labels from argmax of smoothed probabilities.

    Examples
    --------
    >>> smooth = knn_smooth_pred_class_prob(X, probs, class_names, grouping=clusters, k=15)

    Notes
    -----
    Uses `RBFWeight` to set kernel width from median pairwise distance, then
    applies weighted kNN regression to smooth class probabilities within each
    group. Class labels are taken as argmax of the smoothed probabilities.
    """
    if grouping is None:
        grouping = np.zeros(X.shape[0])

    smooth_pred_probs = np.zeros_like(pred_probs)
    smooth_pred_class = np.zeros(pred_probs.shape[0], dtype="object")
    for group in np.unique(grouping):
        group_idx = np.where(grouping == group)[0].astype("int")
        X_group = X[grouping == group, :]
        y_group = pred_probs[grouping == group, :]

        k_use = k(X_group.shape[0]) if callable(k) else k
        k_use = min(k_use, X_group.shape[0])

        rbf = RBFWeight()
        rbf.set_alpha(X=X_group, n_max=None, dm=dm)

        if "dm" in kwargs:
            del kwargs["dm"]

        nns = KNeighborsRegressor(
            n_neighbors=k_use,
            weights=rbf,
            **kwargs,
        ).fit(X_group, y_group)
        smoothed_probs = nns.predict(X_group)

        smooth_pred_probs[group_idx, :] = smoothed_probs
        g_classes = names[np.argmax(smoothed_probs, axis=1)]
        smooth_pred_class[group_idx] = g_classes

    return smooth_pred_class


def argmax_pred_class(
    grouping: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    """
    Assign groupwise majority class to all elements in each group.

    Parameters
    ----------
    grouping : np.ndarray
        Shape [N,]. Group IDs for each element.
    prediction : np.ndarray
        Shape [N,]. Predicted class for each element.

    Returns
    -------
    assigned_classes : np.ndarray
        Shape [N,]. Majority class per group applied to all elements.

    Examples
    --------
    >>> grouping = np.array([0,0,0,1,1,1,2,2,2,2])
    >>> prediction = np.array(['A','A','A','B','A','B','C','A','B','C'])
    >>> argmax_pred_class(grouping, prediction)
    array(['A','A','A','B','B','B','C','C','C','C'], dtype=object)

    Notes
    -----
    Useful when leveraging cluster assignments from another method to
    simplify cell-level labels to cluster-level majorities.
    """
    assert grouping.shape[0] == prediction.shape[0], "`grouping` and `prediction` must be the same length"
    groups = sorted(list(set(grouping.tolist())))

    assigned_classes = np.zeros(grouping.shape[0], dtype="object")
    for group in groups:
        classes, counts = np.unique(prediction[grouping == group], return_counts=True)
        majority_class = classes[np.argmax(counts)]
        assigned_classes[grouping == group] = majority_class
    return assigned_classes


def compute_entropy_of_mixing(
    X: np.ndarray,
    y: np.ndarray,
    n_neighbors: int,
    n_iters: Optional[int] = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Compute entropy of group mixing in local neighborhoods.

    Parameters
    ----------
    X : np.ndarray
        Shape [N, P]. Feature matrix used for neighbor search.
    y : np.ndarray
        Shape [N,]. Discrete group labels.
    n_neighbors : int
        Number of neighbors drawn when computing local distributions.
    n_iters : int, optional
        Number of random query points to evaluate. If None, uses all points.
    **kwargs : Any
        Additional keyword arguments forwarded to `NearestNeighbors`.

    Returns
    -------
    entropy_of_mixing : np.ndarray
        Shape [n_iters,]. Entropy values per query point (in nats).

    Notes
    -----
    For each query point, counts group membership among its k-nearest neighbors
    and computes entropy of the resulting probability vector.
    """
    n_neighbors = min(n_neighbors, X.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", **kwargs).fit(X)
    nn_idx = nn.kneighbors(return_distance=False)

    if n_iters is not None:
        n_iters = min(n_iters, X.shape[0])
    if (n_iters is None) or (n_iters == X.shape[0]):
        query_points = np.arange(X.shape[0])
    else:
        assert n_iters < X.shape[0]
        query_points = np.random.choice(X.shape[0], size=n_iters, replace=False)

    entropy_of_mixing = np.zeros(len(query_points))
    for i, ridx in enumerate(query_points):
        nn_y = y[nn_idx[ridx, :]]
        uniques = np.unique(y)
        nn_y_p = np.zeros(len(uniques))
        for j, v in enumerate(uniques):
            nn_y_p[j] = np.sum(nn_y == v)
        nn_y_p = nn_y_p / nn_y_p.sum()
        H = stats.entropy(nn_y_p)
        entropy_of_mixing[i] = H
    return entropy_of_mixing


def pp_adatas(
    adata_sc: anndata.AnnData,
    adata_sp: anndata.AnnData,
    genes: Optional[Iterable[str]] = None,
    gene_to_lowercase: bool = True,
) -> None:
    """
    Preprocess single-cell and spatial AnnData to align genes and compute density priors.

    Parameters
    ----------
    adata_sc : anndata.AnnData
        Single-cell AnnData.
    adata_sp : anndata.AnnData
        Spatial expression AnnData.
    genes : Iterable[str], optional
        Marker genes to use. If None, all genes from `adata_sc` are considered.
    gene_to_lowercase : bool, default True
        If True, lowercases all gene names to align case between modalities.

    Returns
    -------
    None

    Notes
    -----
    - Filters out all-zero genes in both datasets.
    - Stores shared training genes in `.uns["training_genes"]`.
    - Stores overlapping genes in `.uns["overlap_genes"]`.
    - Computes uniform and RNA-count-based density priors in `adata_sp.obs`.
    """
    sc.pp.filter_genes(adata_sc, min_cells=1)
    sc.pp.filter_genes(adata_sp, min_cells=1)

    if genes is None:
        genes = adata_sc.var.index

    if gene_to_lowercase:
        adata_sc.var.index = [g.lower() for g in adata_sc.var.index]
        adata_sp.var.index = [g.lower() for g in adata_sp.var.index]
        genes = [g.lower() for g in genes]

    adata_sc.var_names_make_unique()
    adata_sp.var_names_make_unique()

    genes = list(set(genes) & set(adata_sc.var.index) & set(adata_sp.var.index))

    adata_sc.uns["training_genes"] = genes
    adata_sp.uns["training_genes"] = genes
    logging.info("%d training genes saved to `.uns['training_genes']`.", len(genes))

    overlap_genes = list(set(adata_sc.var.index) & set(adata_sp.var.index))
    adata_sc.uns["overlap_genes"] = overlap_genes
    adata_sp.uns["overlap_genes"] = overlap_genes
    logging.info("%d overlap genes saved to `.uns['overlap_genes']`.", len(overlap_genes))

    adata_sp.obs["uniform_density"] = np.ones(adata_sp.X.shape[0]) / adata_sp.X.shape[0]
    logging.info("Uniform density prior written to `adata_sp.obs['uniform_density']`.")

    rna_count_per_spot = np.array(adata_sp.X.sum(axis=1)).squeeze()
    adata_sp.obs["rna_count_based_density"] = rna_count_per_spot / np.sum(rna_count_per_spot)
    logging.info("RNA count-based density prior written to `adata_sp.obs['rna_count_based_density']`.")


class RBFWeight(object):
    """
    Radial basis function (Gaussian) weight generator for distances.

    Parameters
    ----------
    alpha : float, optional
        RBF parameter (1 / (2 * sigma^2)). If not set, must call `set_alpha`.

    Notes
    -----
    Weights follow: w(r) = exp(- (alpha * r)^2 ).
    """

    def __init__(self, alpha: Optional[float] = None) -> None:
        self.alpha = alpha

    def set_alpha(
        self,
        X: np.ndarray,
        n_max: Optional[int] = None,
        dm: Optional[np.ndarray] = None,
    ) -> None:
        """
        Estimate `alpha` from the median pairwise distance.

        Parameters
        ----------
        X : np.ndarray
            Shape [N, P]. Observations.
        n_max : int, optional
            Max observations to subsample for the median distance computation.
        dm : np.ndarray, optional
            Shape [N, N]. Precomputed distance matrix; if provided, speeds up estimation.

        Returns
        -------
        None

        References
        ----------
        Gretton et al., "A Kernel Two-Sample Test", JMLR 13(Mar):723–773, 2012.
        """
        if n_max is None:
            n_max = X.shape[0]

        if dm is None:
            if X.shape[0] > n_max:
                ridx = np.random.choice(X.shape[0], size=n_max, replace=False)
                X_p = X[ridx, :]
            else:
                X_p = X
            dm = euclidean_distances(X_p)

        upper = dm[np.triu_indices_from(dm, k=1)]
        sigma = np.median(upper, overwrite_input=True)
        self.alpha = 1.0 / (2 * (sigma ** 2))

    def __call__(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute RBF weights for given distances.

        Parameters
        ----------
        distances : np.ndarray
            Shape [N,]. Distances to weight.

        Returns
        -------
        weights : np.ndarray
            Shape [N,]. RBF weights.

        Notes
        -----
        Requires `self.alpha` to be set via constructor or `set_alpha`.
        """
        if self.alpha is None:
            msg = "must set `alpha` attribute before computing weights.\n"
            msg += "use `.set_alpha()` method to estimate from data."
            raise ValueError(msg)

        weights = np.exp(-((self.alpha * distances) ** 2))
        return weights


def adata_to_cluster_expression(
    adata: anndata.AnnData,
    cluster_label: str,
    scale: bool = True,
    add_density: bool = True,
) -> anndata.AnnData:
    """
    Aggregate single-cell expression to cluster-level expression by `cluster_label`.

    Parameters
    ----------
    adata : anndata.AnnData
        Single-cell AnnData.
    cluster_label : str
        Column in `adata.obs` defining clusters.
    scale : bool, default True
        If True, sums counts per cluster (proportional to cluster size).
        If False, takes mean per cluster.
    add_density : bool, default True
        If True, adds normalized cluster sizes to `.obs['cluster_density']`.

    Returns
    -------
    aggregated : anndata.AnnData
        AnnData with one observation per cluster and the same variables as input.

    Notes
    -----
    Only `cluster_label` is preserved in `.obs` of the returned AnnData (plus
    `cluster_density` if requested).
    """
    try:
        value_counts = adata.obs[cluster_label].value_counts(normalize=True)
    except KeyError as e:
        raise ValueError("Provided label must belong to adata.obs.") from e

    unique_labels = value_counts.index
    new_obs = pd.DataFrame({cluster_label: unique_labels})
    adata_ret = sc.AnnData(obs=new_obs, var=adata.var, uns=adata.uns)

    X_new = np.empty((len(unique_labels), adata.shape[1]))
    for index, l in enumerate(unique_labels):
        if not scale:
            X_new[index] = adata[adata.obs[cluster_label] == l].X.mean(axis=0)
        else:
            X_new[index] = adata[adata.obs[cluster_label] == l].X.sum(axis=0)
    adata_ret.X = X_new

    if add_density:
        adata_ret.obs["cluster_density"] = adata_ret.obs[cluster_label].map(lambda i: value_counts[i])

    return adata_ret


def _edge_list_to_tensor(edge_list: Sequence[Sequence[int]]) -> torch.LongTensor:
    """
    Convert a list of directed edges to a tensor of shape [2, E].

    Parameters
    ----------
    edge_list : sequence of (i, j)
        List-like edge pairs.

    Returns
    -------
    edge_index : torch.LongTensor
        Shape [2, E]. Empty [2, 0] if input is empty.

    Notes
    -----
    Validates shape and dtype; does not deduplicate edges.
    """
    if edge_list is None or len(edge_list) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    arr = np.asarray(edge_list, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("edge_list must be of shape [E, 2].")
    return torch.from_numpy(arr).long().t()


def _to_tuple_list(edges: Union[EdgeList, torch.Tensor, np.ndarray]) -> EdgeList:
    """
    Normalize various edge representations to a list of (i, j) tuples.

    Parameters
    ----------
    edges : list/array/tensor
        List of pairs [[i, j], ...] / [(i, j), ...] or a 2D tensor/array
        of shape [2, E] or [E, 2].

    Returns
    -------
    edge_list : List[Tuple[int, int]]
        List of directed edge tuples.

    Notes
    -----
    Accepts torch or numpy arrays in either [2, E] or [E, 2] layout.
    """
    if isinstance(edges, torch.Tensor):
        if edges.ndim != 2:
            raise ValueError("edge tensor must be 2D.")
        if edges.shape[0] == 2:
            arr = edges.t().detach().cpu().numpy()
        elif edges.shape[1] == 2:
            arr = edges.detach().cpu().numpy()
        else:
            raise ValueError("edge tensor must be [2, E] or [E, 2].")
        return [(int(i), int(j)) for i, j in arr]

    if isinstance(edges, np.ndarray):
        if edges.ndim != 2:
            raise ValueError("edge array must be 2D.")
        if edges.shape[0] == 2:
            arr = edges.T
        elif edges.shape[1] == 2:
            arr = edges
        else:
            raise ValueError("edge array must be [2, E] or [E, 2].")
        return [(int(i), int(j)) for i, j in arr]

    return [(int(e[0]), int(e[1])) for e in edges]


def get_multi_edge_index(
    pos: np.ndarray,
    regions: np.ndarray,
    graph_methods: str = "knn",
    n_neighbors: Optional[int] = None,
    n_radius: Optional[float] = None,
    verbose: bool = True,
) -> torch.LongTensor:
    """
    Build intra-region graphs (no cross-region edges) and merge them.

    Parameters
    ----------
    pos : np.ndarray
        Shape [N, d]. Coordinates.
    regions : np.ndarray
        Shape [N,]. Region label for each node.
    graph_methods : {"knn", "radius"}, default "knn"
        Graph construction method.
    n_neighbors : int, optional
        Required if `graph_methods == "knn"`. Number of neighbors (> 0).
    n_radius : float, optional
        Required if `graph_methods == "radius"`. Neighborhood radius (> 0).
    verbose : bool, default True
        If True, prints average directed neighbors per node.

    Returns
    -------
    edge_index : torch.LongTensor
        Shape [2, E]. Directed edges (i, j). Empty [2, 0] if none.

    Notes
    -----
    Uses PyG `knn_graph` or `radius_graph` per region and remaps indices to
    global node IDs before concatenation.
    """
    if pos.shape[0] != regions.shape[0]:
        raise ValueError("pos and regions must have the same length")

    if graph_methods not in ["knn", "radius"]:
        raise ValueError("graph_methods must be either 'knn' or 'radius'")

    if graph_methods == "knn" and (n_neighbors is None or n_neighbors <= 0):
        raise ValueError("n_neighbors must be a positive integer for knn method")

    if graph_methods == "radius" and (n_radius is None or n_radius <= 0):
        raise ValueError("n_radius must be a positive value for radius method")

    edge_list = []
    regions_unique = np.unique(regions)
    for reg in regions_unique:
        locs = np.where(regions == reg)[0]
        pos_region = pos[locs, :]
        if graph_methods == "knn":
            edge_index = knn_graph(
                torch.Tensor(pos_region),
                k=n_neighbors,
                batch=torch.LongTensor(np.zeros(pos_region.shape[0])),
                loop=True,
            )
        elif graph_methods == "radius":
            edge_index = radius_graph(
                torch.Tensor(pos_region),
                r=n_radius,
                batch=torch.LongTensor(np.zeros(pos_region.shape[0])),
                loop=True,
            )
        for (i, j) in zip(edge_index[1].numpy(), edge_index[0].numpy()):
            edge_list.append([locs[i], locs[j]])

    if verbose:
        N = pos.shape[0]
        E = len(edge_list)
        avg = (E / N) if N > 0 else 0.0
        print(f"Average neighbors per node (directed): {avg:.2f} (edges={E}, nodes={N})")

    if len(edge_list) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.LongTensor(edge_list).T


def get_single_edge_index(
    pos: np.ndarray,
    graph_methods: str = "knn",
    n_neighbors: Optional[int] = None,
    n_radius: Optional[float] = None,
    verbose: bool = True,
) -> torch.LongTensor:
    """
    Build a graph on a single region or the whole set.

    Parameters
    ----------
    pos : np.ndarray
        Shape [N, d]. Coordinates.
    graph_methods : {"knn", "radius"}, default "knn"
        Graph construction method.
    n_neighbors : int, optional
        Required if `graph_methods == "knn"`. Number of neighbors (> 0).
    n_radius : float, optional
        Required if `graph_methods == "radius"`. Neighborhood radius (> 0).
    verbose : bool, default True
        If True, prints average directed neighbors per node.

    Returns
    -------
    edge_index : torch.LongTensor
        Shape [2, E]. Directed edges (i, j). Empty [2, 0] if none.
    """
    if graph_methods not in ["knn", "radius"]:
        raise ValueError("graph_methods must be either 'knn' or 'radius'")

    if graph_methods == "knn" and (n_neighbors is None or n_neighbors <= 0):
        raise ValueError("n_neighbors must be a positive integer for knn method")

    if graph_methods == "radius" and (n_radius is None or n_radius <= 0):
        raise ValueError("n_radius must be a positive value for radius method")

    edge_list = []
    if graph_methods == "knn":
        edge_index = knn_graph(
            torch.Tensor(pos),
            k=n_neighbors,
            batch=torch.LongTensor(np.zeros(pos.shape[0])),
            loop=False,
        )
    elif graph_methods == "radius":
        edge_index = radius_graph(
            torch.Tensor(pos),
            r=n_radius,
            batch=torch.LongTensor(np.zeros(pos.shape[0])),
            loop=False,
        )
    for (i, j) in zip(edge_index[1].numpy(), edge_index[0].numpy()):
        edge_list.append([i, j])

    if verbose:
        N = pos.shape[0]
        E = len(edge_list)
        avg = (E / N) if N > 0 else 0.0
        print(f"Average neighbors per node (directed): {avg:.2f} (edges={E}, nodes={N})")

    if len(edge_list) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.LongTensor(edge_list).T


def get_expr_edge_index(
    expr: np.ndarray,
    n_neighbors: int = 20,
    mode: str = "connectivity",
    metric: str = "correlation",
    include_self: bool = False,
) -> List[Tuple[int, int]]:
    """
    Build a kNN graph from a feature/expression matrix using scikit-learn.

    Parameters
    ----------
    expr : np.ndarray
        Shape [N, P]. Feature matrix.
    n_neighbors : int, default 20
        Number of neighbors.
    mode : {"connectivity", "distance"}, default "connectivity"
        Graph construction mode.
    metric : str, default "correlation"
        Distance/affinity metric passed to `kneighbors_graph`.
    include_self : bool, default False
        Whether to include self-edges.

    Returns
    -------
    edges : list of tuple
        Directed edges (row -> col) as a list of (i, j) pairs.

    Notes
    -----
    Returns COO order from the sparse adjacency.
    """
    adj = kneighbors_graph(
        expr,
        n_neighbors,
        mode=mode,
        metric=metric,
        include_self=include_self,
    )
    edge_list = list(zip(adj.tocoo().row, adj.tocoo().col))
    return edge_list


def edge_lists_intersection(edges1: Union[EdgeList, torch.Tensor], edges2: Union[EdgeList, torch.Tensor]) -> List[Tuple[int, int]]:
    """
    Compute the direction-sensitive intersection of two edge sets.

    Parameters
    ----------
    edges1 : list of (i, j) or torch.LongTensor
        First edge set.
    edges2 : list of (i, j) or torch.LongTensor
        Second edge set.

    Returns
    -------
    edges : list of tuple
        Intersection as a list of directed edges.

    Notes
    -----
    Converts inputs to tuple lists, then intersects as Python sets.
    """
    def to_tuple_list(edges):
        if isinstance(edges, torch.Tensor):
            if edges.ndim != 2 or edges.shape[0] != 2:
                raise ValueError("Edge tensor must have shape [2, E].")
            arr = edges.t().cpu().numpy()
            return [tuple(map(int, e)) for e in arr]
        return [tuple(e) for e in edges]

    set1 = set(to_tuple_list(edges1))
    set2 = set(to_tuple_list(edges2))
    return list(set1 & set2)


def get_consensus_edges(
    spatial: np.ndarray,
    *omics: np.ndarray,
    target_neighbors: int = 8,
    max_iter: int = 20,
) -> torch.LongTensor:
    """
    Intersect spatial and feature kNN graphs to target a desired average degree.

    Parameters
    ----------
    spatial : np.ndarray
        Shape [N, d]. Coordinates.
    *omics : np.ndarray
        One or more matrices, each shape [N, p_k], concatenated for feature kNN.
    target_neighbors : int, default 8
        Desired average number of neighbors in the intersection (0 < target < N).
    max_iter : int, default 20
        Maximum number of binary-search iterations.

    Returns
    -------
    edge_index : torch.LongTensor
        Shape [2, E]. Intersection edges (i, j). Empty [2, 0] if none.

    Notes
    -----
    Binary-searches the neighbor count used for both spatial and feature graphs
    until the intersection's average degree approaches `target_neighbors`.
    """
    n_cells = spatial.shape[0]
    assert all(omic.shape[0] == n_cells for omic in omics), "Omics data dimension mismatch"
    assert 0 < target_neighbors < n_cells, "Invalid target_neighbors"

    low, high = 4, min(80, n_cells - 1)
    combined_omics = np.hstack(omics)

    for _ in range(max_iter):
        n_neighbors = (low + high) // 2
        edge_index_spatial = get_single_edge_index(spatial, n_neighbors=n_neighbors)
        edge_index_feat = get_expr_edge_index(combined_omics, n_neighbors=n_neighbors)

        edge_index_intersect = edge_lists_intersection(edge_index_spatial, edge_index_feat)
        avg_neighbors = len(edge_index_intersect) / n_cells

        if abs(avg_neighbors - target_neighbors) < 0.5:
            break
        elif avg_neighbors < target_neighbors:
            low = n_neighbors + 1
        else:
            high = n_neighbors - 1
        if low >= high:
            break

    edge_index = torch.LongTensor(edge_index_intersect).T if len(edge_index_intersect) > 0 \
        else torch.empty((2, 0), dtype=torch.long)
    return edge_index


def tfidf(X: Union[np.ndarray, sparse.csr_matrix]) -> Union[np.ndarray, sparse.csr_matrix]:
    """
    Apply TF-IDF normalization (Seurat v3-style).

    Parameters
    ----------
    X : np.ndarray or sparse.csr_matrix
        Input matrix of shape [cells, features].

    Returns
    -------
    X_tfidf : np.ndarray or sparse.csr_matrix
        TF-IDF normalized matrix with the same sparsity type as input.

    Notes
    -----
    idf = n_cells / feature_sum; tf is row-normalized counts.
    """
    idf = X.shape[0] / X.sum(axis=0)
    if sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def lsi(
    adata: anndata.AnnData,
    n_comps: int = 20,
    use_highly_variable: Optional[bool] = None,
    **kwargs: Any,
) -> anndata.AnnData:
    """
    Compute LSI embeddings following the Seurat v3 approach.

    Parameters
    ----------
    adata : anndata.AnnData
        Input AnnData.
    n_comps : int, default 20
        Number of LSI dimensions.
    use_highly_variable : bool, optional
        If None, uses HVGs if available; otherwise all genes.
    **kwargs : Any
        Additional kwargs forwarded to `sklearn.utils.extmath.randomized_svd`.

    Returns
    -------
    adata : anndata.AnnData
        Input AnnData with `adata.obsm["X_lsi"]` added.

    Notes
    -----
    Applies TF-IDF, L1 normalization, log1p scaling, randomized SVD, and per-cell
    z-scoring (mean 0, std 1).
    """
    if "random_state" not in kwargs:
        kwargs["random_state"] = 0
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var

    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata

    X = tfidf(adata_use.X)
    X_norm = normalize(X, norm="l1")
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_comps, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi
    return adata


def _optimize_cluster(
    adata: anndata.AnnData,
    resolution: List[float] = list(np.arange(0.01, 2.5, 0.01)),
) -> float:
    """
    Optimize Leiden resolution by maximizing the Calinski–Harabasz score.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData whose `.X` (samples × features) is used for scoring.
        This function overwrites `adata.obs["leiden"]` during the search.
    resolution : list of float, default np.arange(0.01, 2.5, 0.01)
        Candidate resolutions.

    Returns
    -------
    res : float
        Best resolution (prints it as well).

    Notes
    -----
    Runs `sc.tl.leiden` for each resolution and computes CH score on `.X`.
    """
    scores = []
    for r in resolution:
        sc.tl.leiden(adata, resolution=r, flavor="igraph", n_iterations=2, directed=False)
        s = calinski_harabasz_score(adata.X, adata.obs["leiden"])
        scores.append(s)
    cl_opt_df = pd.DataFrame({"resolution": resolution, "score": scores})
    best_idx = int(np.argmax(cl_opt_df["score"]))
    res = float(cl_opt_df.iloc[best_idx, 0])
    print("Best resolution: ", res)
    return res


def _priori_cluster(
    adata: anndata.AnnData,
    eval_cluster_n: int = 7,
    res_min: float = 0.01,
    res_max: float = 2.5,
    res_step: float = 0.01,
) -> float:
    """
    Find a Leiden resolution that yields a target number of clusters.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData to be clustered. Overwrites `adata.obs["leiden"]`.
    eval_cluster_n : int, default 7
        Desired number of Leiden clusters.
    res_min : float, default 0.01
        Minimum resolution to try (inclusive).
    res_max : float, default 2.5
        Maximum resolution to try (inclusive).
    res_step : float, default 0.01
        Step size between consecutive resolutions.

    Returns
    -------
    res : float
        Resolution that first (from high to low) yields exactly `eval_cluster_n`,
        or the last tried value if none match.

    Notes
    -----
    Search order is descending to mirror original behavior.
    """
    if res_step <= 0:
        raise ValueError("res_step must be > 0")
    if res_max < res_min:
        raise ValueError("res_max must be >= res_min")

    resolutions = np.arange(res_min, res_max, res_step)
    if resolutions.size == 0:
        raise ValueError("Empty resolution grid. Check res_min/res_max/res_step.")

    for res in sorted(resolutions.tolist(), reverse=True):
        sc.tl.leiden(adata, resolution=res, flavor="igraph", n_iterations=2, directed=False)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        if count_unique_leiden == int(eval_cluster_n):
            break

    print("Best resolution: ", res)
    return res


def mclust_R(
    adata: anndata.AnnData,
    num_cluster: int,
    modelNames: str = 'EEE',
    used_obsm: str = "DIRAC_embed",
    random_seed: int = 2020,
    key_added: str = "DIRAC",
) -> None:
    """
    Cluster embeddings using R's `mclust` via `rpy2`.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with embeddings in `adata.obsm[used_obsm]` of shape [N, d].
    num_cluster : int
        Number of clusters for `Mclust`.
    modelNames : str, default 'EEE'
        Covariance model string (see mclust docs).
    used_obsm : str, default 'DIRAC_embed'
        Key in `adata.obsm` to cluster.
    random_seed : int, default 2020
        Random seed for NumPy and R.
    key_added : str, default 'DIRAC'
        Column name to store cluster labels in `adata.obs`.

    Returns
    -------
    None

    Notes
    -----
    Requires `rpy2` and R package `mclust`. Writes integer labels to
    `adata.obs[key_added]` as categorical.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])
    adata.obs[key_added] = mclust_res
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')


def seed_torch(seed: int = 1029) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch (CPU & CUDA) for reproducibility.

    Parameters
    ----------
    seed : int, default 1029
        Seed value used across Python's `random`, NumPy, and PyTorch.

    Returns
    -------
    None

    Notes
    -----
    Sets:
      - os.environ['PYTHONHASHSEED']
      - torch.manual_seed / cuda.manual_seed / cuda.manual_seed_all
      - torch.backends.cudnn.benchmark = False
      - torch.backends.cudnn.deterministic = True
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def combine_multimodal_adatas(
    adatas: Dict[str, anndata.AnnData],
    *,
    prefixes: Optional[Dict[str, str]] = None,
    align_obs: bool = True,
    preserve_obsm: Iterable[str] = ("spatial",),
    dtype: np.dtype = np.float32,
) -> anndata.AnnData:
    """
    Concatenate features across modalities for the same set of cells.

    Parameters
    ----------
    adatas : Dict[str, AnnData]
        Mapping from modality name (e.g., "RNA", "ATAC", "ADT") to AnnData.
        Dict insertion order determines feature block order.
    prefixes : Dict[str, str], optional
        Per-modality prefixes for feature names (default: f"{mod.upper()}_").
    align_obs : bool, default True
        If True, reindex each AnnData to match the first modality's cells.
        If False, raises on mismatch.
    preserve_obsm : Iterable[str], default ("spatial",)
        Keys in `.obsm` to copy from the first modality if present.
    dtype : np.dtype, default np.float32
        Output matrix dtype.

    Returns
    -------
    combined : anndata.AnnData
        AnnData with dense `X` (features concatenated), `.var` describing
        feature types and original names, `.obs` from the reference modality,
        and `.uns` with combination metadata.

    Notes
    -----
    Converts all blocks to dense arrays before horizontal concatenation.
    """
    if not adatas:
        raise ValueError("`adatas` is empty. Provide at least one AnnData.")

    modalities = list(adatas.keys())
    ref_key = modalities[0]
    ref = adatas[ref_key]
    ref_index = ref.obs.index

    for k, ad in list(adatas.items()):
        if align_obs:
            if not ref_index.equals(ad.obs.index):
                try:
                    adatas[k] = ad[ref_index].copy()
                except KeyError:
                    raise ValueError(
                        f"Cell sets differ between '{ref_key}' and '{k}'. Cannot align."
                    ) from None
        else:
            if not ref_index.equals(ad.obs.index):
                raise ValueError(
                    f"Observation names between '{ref_key}' and '{k}' do not match in the same order."
                )

    dense_blocks = []
    var_index = []
    feature_types = []
    original_features = []

    for mod in modalities:
        ad = adatas[mod]
        X = ad.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=dtype, order="C")
        dense_blocks.append(X)

        pref = (prefixes or {}).get(mod, f"{mod.upper()}_")
        names = [f"{pref}{n}" for n in ad.var_names.astype(str)]
        var_index.extend(names)
        feature_types.extend([mod] * ad.n_vars)
        original_features.extend(ad.var_names.astype(str).tolist())

    combined_X = np.hstack(dense_blocks).astype(dtype, copy=False)

    combined_var = pd.DataFrame(
        {
            "feature_types": feature_types,
            "original_feature": original_features,
        },
        index=pd.Index(var_index, name="feature"),
    )

    combined = anndata.AnnData(
        X=combined_X,
        obs=adatas[ref_key].obs.copy(),
        var=combined_var,
        uns={
            "modality_combine": "+".join(modalities),
            "modalities": {m: adatas[m].n_vars for m in modalities},
        },
    )

    for key in preserve_obsm:
        if key in ref.obsm:
            combined.obsm[key] = ref.obsm[key].copy()

    return combined


def ctg(
    adata_sc: anndata.AnnData,
    cluster_label: str,
    n_genes: int = 150,
    *,
    min_cells: int = 3,
    method: str = "wilcoxon",
    use_raw: bool = False,
) -> List[str]:
    """
    Select top marker genes per cluster using Scanpy's `rank_genes_groups`.

    Parameters
    ----------
    adata_sc : anndata.AnnData
        Single-cell AnnData with expression matrix and metadata.
    cluster_label : str
        Column in `adata_sc.obs` with cluster identities.
    n_genes : int, default 150
        Number of top-ranked genes to collect per cluster before de-duplication.
    min_cells : int, keyword-only, default 3
        Minimum cells a gene must be expressed in prior to ranking.
    method : {"wilcoxon", "t-test", "logreg"}, keyword-only, default "wilcoxon"
        Statistical test for differential expression.
    use_raw : bool, keyword-only, default False
        Whether to use `adata.raw` for testing.

    Returns
    -------
    markers : List[str]
        Unique list of top marker genes across clusters.

    Notes
    -----
    Works on a copy of the input AnnData to avoid in-place modifications.
    Applies minimal preprocessing (filter, normalize_total, log1p) before DE.
    """
    if cluster_label not in adata_sc.obs.columns:
        raise ValueError(
            f"`cluster_label` '{cluster_label}' not found in adata_sc.obs. "
            f"Available columns: {list(adata_sc.obs.columns)}"
        )
    if n_genes <= 0:
        raise ValueError("`n_genes` must be a positive integer.")
    if method not in {"wilcoxon", "t-test", "logreg"}:
        raise ValueError("`method` must be one of: 'wilcoxon', 't-test', 'logreg'.")

    adata = adata_sc.copy()

    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_label,
        method=method,
        use_raw=use_raw,
    )

    names = adata.uns["rank_genes_groups"]["names"]
    markers_df = pd.DataFrame(names)

    top_n = min(n_genes, markers_df.shape[0])
    top_df = markers_df.iloc[:top_n, :]

    unique_markers = list(np.unique(top_df.to_numpy().ravel()))
    return unique_markers
