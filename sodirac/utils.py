#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/17/23 2:58 PM
# @Author  : Chang Xu
# @File    : utils.py
# @Email   : changxu@nus.edu.sg

import os
import torch
import numpy as np
from scipy import sparse
import pandas as pd
import tqdm
import sys
from scipy import stats
from builtins import range
import scanpy as sc
import sklearn
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.metrics.pairwise import euclidean_distances
from typing import Union, Callable, Any, Iterable, List, Optional, Tuple, Sequence, Dict
from sklearn.metrics import pairwise_distances, calinski_harabasz_score
from torch_geometric.nn import knn_graph, radius_graph
import anndata
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph 
import random

EdgeList = List[Tuple[int, int]]


def append_categorical_to_data(
    X: Union[np.ndarray, sparse.csr.csr_matrix],
    categorical: np.ndarray,
	) -> (Union[np.ndarray, sparse.csr.csr_matrix], np.ndarray):
    """Convert `categorical` to a one-hot vector and append
    this vector to each sample in `X`.

    Parameters
    ----------
    X : np.ndarray, sparse.csr.csr_matrix
        [Cells, Features]
    categorical : np.ndarray
        [Cells,]

    Returns
    -------
    Xa : np.ndarray
        [Cells, Features + N_Categories]
    categories : np.ndarray
        [N_Categories,] str category descriptors.
    """
    # `pd.Categorical(xyz).codes` are int values for each unique
    # level in the vector `xyz`
    labels = pd.Categorical(categorical)
    idx = np.array(labels.codes)
    idx = torch.from_numpy(idx.astype("int32")).long()
    categories = np.array(labels.categories)

    one_hot_mat = make_one_hot(
        idx,
        C=len(categories),
    )
    one_hot_mat = one_hot_mat.numpy()
    assert X.shape[0] == one_hot_mat.shape[0], "dims unequal at %d, %d" % (
        X.shape[0],
        one_hot_mat.shape[0],
    )
    # append one hot vector to the [Cells, Features] matrix
    if sparse.issparse(X):
        X = sparse.hstack([X, one_hot_mat])
    else:
        X = np.concatenate([X, one_hot_mat], axis=1)
    return X, categories

def get_adata_asarray(
    adata: anndata.AnnData,
	) -> Union[np.ndarray, sparse.csr.csr_matrix]:
    """Get the gene expression matrix `.X` of an
    AnnData object as an array rather than a view.

    Parameters
    ----------
    adata : anndata.AnnData
        [Cells, Genes] AnnData experiment.

    Returns
    -------
    X : np.ndarray, sparse.csr.csr_matrix
        [Cells, Genes] `.X` attribute as an array
        in memory.

    Notes
    -----
    Returned `X` will match the type of `adata.X` view.
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
    Build a matrix for classification using only genes that overlap
    between the current sample and the pre-trained model.

    Parameters
    ----------
    X : np.ndarray, sparse.csr_matrix
        [Cells, Genes] count matrix.
    model_genes : np.ndarray
        gene identifiers in the order expected by the model.
    sample_genes : np.ndarray
        gene identifiers for the current sample.
    gene_batch_size : int
        number of genes to copy between arrays per batch.
        controls a speed vs. memory trade-off.

    Returns
    -------
    N : np.ndarray, sparse.csr_matrix
        [Cells, len(model_genes)] count matrix.
        Values where a model gene was not present in the sample are left
        as zeros. `type(N)` will match `type(X)`.
    """
    # check types
    if type(X) not in (np.ndarray, sparse.csr.csr_matrix):
        msg = f"X is type {type(X)}, must `np.ndarray` or `sparse.csr_matrix`"
        raise TypeError(msg)
    n_cells = X.shape[0]
    # check if gene names already match exactly
    if len(model_genes) == len(sample_genes):
        if np.all(model_genes == sample_genes):
            print("Gene names match exactly, returning input.")
            return X

    # instantiate a new [Cells, model_genes] matrix where columns
    # retain the order used during training
    if type(X) == np.ndarray:
        N = np.zeros((n_cells, len(model_genes)))
    else:
        # use sparse matrices if the input is sparse
        N = sparse.lil_matrix(
            (
                n_cells,
                len(model_genes),
            )
        )

    # map gene indices from the model to the sample genes
    model_genes_indices = []
    sample_genes_indices = []
    common_genes = 0
    for i, g in tqdm.tqdm(enumerate(sample_genes), desc="mapping genes"):
        if np.sum(g == model_genes) > 0:
            model_genes_indices.append(int(np.where(g == model_genes)[0]))
            sample_genes_indices.append(
                i,
            )
            common_genes += 1

    # copy the data in batches to the new array to avoid memory overflows
    gene_idx = 0
    n_batches = int(np.ceil(N.shape[1] / gene_batch_size))
    for b in tqdm.tqdm(range(n_batches), desc="copying gene batches"):
        model_batch_idx = model_genes_indices[gene_idx : gene_idx + gene_batch_size]
        sample_batch_idx = sample_genes_indices[gene_idx : gene_idx + gene_batch_size]
        N[:, model_batch_idx] = X[:, sample_batch_idx]
        gene_idx += gene_batch_size

    if sparse.issparse(N):
        # convert to `csr` from `csc`
        N = sparse.csr_matrix(N)
    print("Found %d common genes." % common_genes)
    return N


def knn_smooth_pred_class(
    X: np.ndarray,
    pred_class: np.ndarray,
    grouping: np.ndarray = None,
    k: int = 15,
	) -> np.ndarray:
    """
    Smooths class predictions by taking the modal class from each cell's
    nearest neighbors.

    Parameters
    ----------
    X : np.ndarray
        [N, Features] embedding space for calculation of nearest neighbors.
    pred_class : np.ndarray
        [N,] array of unique class labels.
    groupings : np.ndarray
        [N,] unique grouping labels for i.e. clusters.
        if provided, only considers nearest neighbors *within the cluster*.
    k : int
        number of nearest neighbors to use for smoothing.

    Returns
    -------
    smooth_pred_class : np.ndarray
        [N,] unique class labels, smoothed by kNN.

    Examples
    --------
    >>> smooth_pred_class = knn_smooth_pred_class(
    ...     X = X,
    ...     pred_class = raw_predicted_classes,
    ...     grouping = louvain_cluster_groups,
    ...     k = 15,)

    Notes
    -----
    scNym classifiers do not incorporate neighborhood information.
    By using a simple kNN smoothing heuristic, we can leverage neighborhood
    information to improve classification performance, smoothing out cells
    that have an outlier prediction relative to their local neighborhood.
    """
    if grouping is None:
        # do not use a grouping to restrict local neighborhood
        # associations, create a universal pseudogroup `0`.
        grouping = np.zeros(X.shape[0])

    smooth_pred_class = np.zeros_like(pred_class)
    for group in np.unique(grouping):
        # identify only cells in the relevant group
        group_idx = np.where(grouping == group)[0].astype("int")
        X_group = X[grouping == group, :]
        # if there are < k cells in the group, change `k` to the
        # group size
        if X_group.shape[0] < k:
            k_use = X_group.shape[0]
        else:
            k_use = k
        # compute a nearest neighbor graph and identify kNN
        nns = NearestNeighbors(
            n_neighbors=k_use,
        ).fit(X_group)
        dist, idx = nns.kneighbors(X_group)

        # for each cell in the group, assign a class as
        # the majority class of the kNN
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
    grouping: np.ndarray = None,
    k: Union[Callable, int] = 15,
    dm: np.ndarray = None,
    **kwargs,
	) -> np.ndarray:
    """
    Smooths class predictions by taking the modal class from each cell's
    nearest neighbors.

    Parameters
    ----------
    X : np.ndarray
        [N, Features] embedding space for calculation of nearest neighbors.
    pred_probs : np.ndarray
        [N, C] array of class prediction probabilities.
    names : np.ndarray,
        [C,] names of predicted classes in `pred_probs`.
    groupings : np.ndarray
        [N,] unique grouping labels for i.e. clusters.
        if provided, only considers nearest neighbors *within the cluster*.
    k : int
        number of nearest neighbors to use for smoothing.
    dm : np.ndarray, optional
        [N, N] distance matrix for setting the RBF kernel parameter.
        speeds computation if pre-computed.

    Returns
    -------
    smooth_pred_class : np.ndarray
        [N,] unique class labels, smoothed by kNN.

    Examples
    --------
    >>> smooth_pred_class = knn_smooth_pred_class_prob(
    ...     X = X,
    ...     pred_probs = predicted_class_probs,
    ...     grouping = louvain_cluster_groups,
    ...     k = 15,)

    Notes
    -----
    scNym classifiers do not incorporate neighborhood information.
    By using a simple kNN smoothing heuristic, we can leverage neighborhood
    information to improve classification performance, smoothing out cells
    that have an outlier prediction relative to their local neighborhood.
    """
    if grouping is None:
        # do not use a grouping to restrict local neighborhood
        # associations, create a universal pseudogroup `0`.
        grouping = np.zeros(X.shape[0])

    smooth_pred_probs = np.zeros_like(pred_probs)
    smooth_pred_class = np.zeros(pred_probs.shape[0], dtype="object")
    for group in np.unique(grouping):
        # identify only cells in the relevant group
        group_idx = np.where(grouping == group)[0].astype("int")
        X_group = X[grouping == group, :]
        y_group = pred_probs[grouping == group, :]
        # if k is a Callable, use it to define k for this group
        if callable(k):
            k_use = k(X_group.shape[0])
        else:
            k_use = k

        # if there are < k cells in the group, change `k` to the
        # group size
        if X_group.shape[0] < k_use:
            k_use = X_group.shape[0]

        # set up weights using a radial basis function kernel
        rbf = RBFWeight()
        rbf.set_alpha(
            X=X_group,
            n_max=None,
            dm=dm,
        )

        if "dm" in kwargs:
            del kwargs["dm"]
        # fit a nearest neighbor regressor
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
	):
    """Assign class to elements in groups based on the
    most common predicted class for that group.

    Parameters
    ----------
    grouping : np.ndarray
        [N,] partition values defining groups to be classified.
    prediction : np.ndarray
        [N,] predicted values for each element in `grouping`.

    Returns
    -------
    assigned_classes : np.ndarray
        [N,] class labels based on the most common class assigned
        to elements in the group partition.

    Examples
    --------
    >>> grouping = np.array([0,0,0,1,1,1,2,2,2,2])
    >>> prediction = np.array(['A','A','A','B','A','B','C','A','B','C'])
    >>> argmax_pred_class(grouping, prediction)
    np.ndarray(['A','A','A','B','B','B','C','C','C','C',])

    Notes
    -----
    scNym classifiers do not incorporate neighborhood information.
    This simple heuristic leverages cluster information obtained by
    an orthogonal method and assigns all cells in a given cluster
    the majority class label within that cluster.
    """
    assert (
        grouping.shape[0] == prediction.shape[0]
    ), "`grouping` and `prediction` must be the same length"
    groups = sorted(list(set(grouping.tolist())))

    assigned_classes = np.zeros(grouping.shape[0], dtype="object")

    for i, group in enumerate(groups):
        classes, counts = np.unique(prediction[grouping == group], return_counts=True)
        majority_class = classes[np.argmax(counts)]
        assigned_classes[grouping == group] = majority_class
    return assigned_classes   

def compute_entropy_of_mixing(
    X: np.ndarray,
    y: np.ndarray,
    n_neighbors: int,
    n_iters: int = None,
    **kwargs,
	) -> np.ndarray:
    """Compute the entropy of mixing among groups given
    a distance matrix.

    Parameters
    ----------
    X : np.ndarray
        [N, P] feature matrix.
    y : np.ndarray
        [N,] group labels.
    n_neighbors : int
        number of nearest neighbors to draw for each iteration
        of the entropy computation.
    n_iters : int
        number of iterations to perform.
        if `n_iters is None`, uses every point.

    Returns
    -------
    entropy_of_mixing : np.ndarray
        [n_iters,] entropy values for each iteration.

    Notes
    -----
    The entropy of batch mixing is computed by sampling `n_per_sample`
    cells from a local neighborhood in the nearest neighbor graph
    and contructing a probability vector based on their group membership.
    The entropy of this probability vector is computed as a metric of
    intermixing between groups.

    If groups are more mixed, the probability vector will have higher
    entropy, and vice-versa.
    """
    # build nearest neighbor graph
    n_neighbors = min(n_neighbors, X.shape[0])
    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
        **kwargs,
    )
    nn.fit(X)
    nn_idx = nn.kneighbors(return_distance=False)

    # define query points
    if n_iters is not None:
        # don't duplicate points when sampling
        n_iters = min(n_iters, X.shape[0])

    if (n_iters is None) or (n_iters == X.shape[0]):
        # sample all points
        query_points = np.arange(X.shape[0])
    else:
        # subset random query points for entropy
        # computation
        assert n_iters < X.shape[0]
        query_points = np.random.choice(
            X.shape[0],
            size=n_iters,
            replace=False,
        )

    entropy_of_mixing = np.zeros(len(query_points))
    for i, ridx in enumerate(query_points):
        # get the nearest neighbors of a point
        nn_y = y[nn_idx[ridx, :]]

        nn_y_p = np.zeros(len(np.unique(y)))
        for j, v in enumerate(np.unique(y)):
            nn_y_p[j] = sum(nn_y == v)
        nn_y_p = nn_y_p / nn_y_p.sum()

        # use base 2 to return values in bits rather
        # than the default nats
        H = stats.entropy(nn_y_p)
        entropy_of_mixing[i] = H
    return entropy_of_mixing


def pp_adatas(
    adata_sc, 
    adata_sp, 
    genes = None, 
    gene_to_lowercase = True
    ):
    """
    Pre-process AnnDatas so that they can be mapped. Specifically:
    - Remove genes that all entries are zero
    - Find the intersection between adata_sc, adata_sp and given marker gene list, save the intersected markers in two adatas
    - Calculate density priors and save it with adata_sp
    Args:
        adata_sc (AnnData): single cell data
        adata_sp (AnnData): spatial expression data
        genes (List): Optional. List of genes to use. If `None`, all genes are used.
    
    Returns:
        update adata_sc by creating `uns` `training_genes` `overlap_genes` fields 
        update adata_sp by creating `uns` `training_genes` `overlap_genes` fields and creating `obs` `rna_count_based_density` & `uniform_density` field
    """

    # remove all-zero-valued genes
    sc.pp.filter_genes(adata_sc, min_cells=1)
    sc.pp.filter_genes(adata_sp, min_cells=1)

    if genes is None:
        # Use all genes
        genes = adata_sc.var.index
               
    # put all var index to lower case to align
    if gene_to_lowercase:
        adata_sc.var.index = [g.lower() for g in adata_sc.var.index]
        adata_sp.var.index = [g.lower() for g in adata_sp.var.index]
        genes = list(g.lower() for g in genes)

    adata_sc.var_names_make_unique()
    adata_sp.var_names_make_unique()
    
    # Refine `marker_genes` so that they are shared by both adatas
    genes = list(set(genes) & set(adata_sc.var.index) & set(adata_sp.var.index))
    # logging.info(f"{len(genes)} shared marker genes.")

    adata_sc.uns["training_genes"] = genes
    adata_sp.uns["training_genes"] = genes
    logging.info(
        "{} training genes are saved in `uns``training_genes` of both single cell and spatial Anndatas.".format(
            len(genes)
        )
    )

    # Find overlap genes between two AnnDatas
    overlap_genes = list(set(adata_sc.var.index) & set(adata_sp.var.index))
    # logging.info(f"{len(overlap_genes)} shared genes.")

    adata_sc.uns["overlap_genes"] = overlap_genes
    adata_sp.uns["overlap_genes"] = overlap_genes
    logging.info(
        "{} overlapped genes are saved in `uns``overlap_genes` of both single cell and spatial Anndatas.".format(
            len(overlap_genes)
        )
    )

    # Calculate uniform density prior as 1/number_of_spots
    adata_sp.obs["uniform_density"] = np.ones(adata_sp.X.shape[0]) / adata_sp.X.shape[0]
    logging.info(
        f"uniform based density prior is calculated and saved in `obs``uniform_density` of the spatial Anndata."
    )

    # Calculate rna_count_based density prior as % of rna molecule count
    rna_count_per_spot = np.array(adata_sp.X.sum(axis=1)).squeeze()
    adata_sp.obs["rna_count_based_density"] = rna_count_per_spot / np.sum(rna_count_per_spot)
    logging.info(
        f"rna count based density prior is calculated and saved in `obs``rna_count_based_density` of the spatial Anndata."
    )

class RBFWeight(object):
    def __init__(
        self,
        alpha: float = None,
    ) -> None:
        """Generate a set of weights based on distances to a point
        with a radial basis function kernel.

        Parameters
        ----------
        alpha : float
            radial basis function parameter. inverse of sigma
            for a standard Gaussian pdf.

        Returns
        -------
        None.
        """
        self.alpha = alpha
        return

    def set_alpha(
        self,
        X: np.ndarray,
        n_max: int = None,
        dm: np.ndarray = None,
    ) -> None:
        """Set the alpha parameter of a Gaussian RBF kernel
        as the median distance between points in an array of
        observations.

        Parameters
        ----------
        X : np.ndarray
            [N, P] matrix of observations and features.
        n_max : int
            maximum number of observations to use for median
            distance computation.
        dm : np.ndarray, optional
            [N, N] distance matrix for setting the RBF kernel parameter.
            speeds computation if pre-computed.

        Returns
        -------
        None. Sets `self.alpha`.

        References
        ----------
        A Kernel Two-Sample Test
        Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch,
        Bernhard Schölkopf, Alexander Smola.
        JMLR, 13(Mar):723−773, 2012.
        http://jmlr.csail.mit.edu/papers/v13/gretton12a.html
        """
        if n_max is None:
            n_max = X.shape[0]

        if dm is None:
            # compute a distance matrix from observations
            if X.shape[0] > n_max:
                ridx = np.random.choice(
                    X.shape[0],
                    size=n_max,
                    replace=False,
                )
                X_p = X[ridx, :]
            else:
                X_p = X

            dm = euclidean_distances(
                X_p,
            )

        upper = dm[np.triu_indices_from(dm, k=1)]

        # overwrite_input = True saves memory by overwriting
        # the upper indices in the distance matrix array during
        # median computation
        sigma = np.median(
            upper,
            overwrite_input=True,
        )
        self.alpha = 1.0 / (2 * (sigma ** 2))
        return

    def __call__(
        self,
        distances: np.ndarray,
    ) -> np.ndarray:
        """Generate a set of weights based on distances to a point
        with a radial basis function kernel.

        Parameters
        ----------
        distances : np.ndarray
            [N,] distances used to generate weights.

        Returns
        -------
        weights : np.ndarray
            [N,] weights from the radial basis function kernel.

        Notes
        -----
        We weight distances with a Gaussian RBF.

        .. math::

            f(r) = \exp -(\alpha r)^2

        """
        # check that alpha parameter is set
        if self.alpha is None:
            msg = "must set `alpha` attribute before computing weights.\n"
            msg += "use `.set_alpha() method to estimate from data."
            raise ValueError(msg)

        # generate weights with an RBF kernel
        weights = np.exp(-((self.alpha * distances) ** 2))
        return weights


def adata_to_cluster_expression(
    adata, 
    cluster_label, 
    scale=True, 
    add_density=True
    ):
    """
    Convert an AnnData to a new AnnData with cluster expressions. Clusters are based on `cluster_label` in `adata.obs`.  
    The returned AnnData has an observation for each cluster, with the cluster-level expression equals to the average expression for that cluster.
    All annotations in `adata.obs` except `cluster_label` are discarded in the returned AnnData.
    
    Args:
        adata (AnnData): single cell data
        cluster_label (String): field in `adata.obs` used for aggregating values
        scale (bool): Optional. Whether weight input single cell by # of cells in cluster. Default is True.
        add_density (bool): Optional. If True, the normalized number of cells in each cluster is added to the returned AnnData as obs.cluster_density. Default is True.
    Returns:
        AnnData: aggregated single cell data
    """
    try:
        value_counts = adata.obs[cluster_label].value_counts(normalize=True)
    except KeyError as e:
        raise ValueError("Provided label must belong to adata.obs.")
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
        adata_ret.obs["cluster_density"] = adata_ret.obs[cluster_label].map(
            lambda i: value_counts[i]
        )

    return adata_ret


def _edge_list_to_tensor(edge_list: Sequence[Sequence[int]]) -> torch.LongTensor:
    """
    Convert [[i, j], ...] or [(i, j), ...] to a torch.LongTensor of shape [2, E].
    Returns an empty [2, 0] tensor if the list is empty.
    """
    if edge_list is None or len(edge_list) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    arr = np.asarray(edge_list, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("edge_list must be of shape [E, 2].")
    return torch.from_numpy(arr).long().t()  # [2, E]


def _to_tuple_list(
    edges: Union[EdgeList, torch.Tensor, np.ndarray]
) -> EdgeList:
    """
    Normalize different edge representations to a List[Tuple[int, int]].

    Accepts:
      - list of pairs [[i, j], ...] / [(i, j), ...]
      - torch.Tensor of shape [2, E] or [E, 2]
      - np.ndarray of shape [2, E] or [E, 2]
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
        if edges.shape[0] == 2:      # [2, E]
            arr = edges.T
        elif edges.shape[1] == 2:    # [E, 2]
            arr = edges
        else:
            raise ValueError("edge array must be [2, E] or [E, 2].")
        return [(int(i), int(j)) for i, j in arr]

    # assume list-like of pairs
    return [(int(e[0]), int(e[1])) for e in edges]


# ---------- main APIs ----------
def get_multi_edge_index(
    pos, 
    regions, 
    graph_methods = "knn",
    n_neighbors = None,
    n_radius = None,
    verbose: bool = True,
    ):
    """
    Build intra-region graphs (no cross-region edges) and merge them.

    Inputs
    ------
    pos : np.ndarray, shape [N, d]
        Coordinate matrix.
    regions : np.ndarray, shape [N]
        Region label for each node (no edges across regions).
    graph_methods : {"knn", "radius"}, default "knn"
        Graph construction method.
    n_neighbors : int, required if graph_methods == "knn"
        Number of neighbors (must be > 0).
    n_radius : float, required if graph_methods == "radius"
        Neighborhood radius (must be > 0).

    Output
    ------
    edge_index : torch.LongTensor, shape [2, E]
        Directed edges (i, j). Returns an empty tensor of shape [2, 0] if no edges.
    """
    # construct edge indexes when there is region information
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
            edge_index = knn_graph(torch.Tensor(pos_region), 
                                   k = n_neighbors, 
                                   batch = torch.LongTensor(np.zeros(pos_region.shape[0])), 
                                   loop = True)
        elif graph_methods == "radius":
            edge_index = radius_graph(torch.Tensor(pos_region), 
                                      r = n_radius, 
                                      batch = torch.LongTensor(np.zeros(pos_region.shape[0])), 
                                      loop = True)
        for (i, j) in zip(edge_index[1].numpy(), edge_index[0].numpy()):
            edge_list.append([locs[i], locs[j]])

    # optional: print average neighbors (directed)
    if verbose:
        N = pos.shape[0]
        E = len(edge_list)
        avg = (E / N) if N > 0 else 0.0
        print(f"Average neighbors per node (directed): {avg:.2f} "
              f"(edges={E}, nodes={N})")

    # return as [2, E] tensor
    if len(edge_list) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.LongTensor(edge_list).T


def get_single_edge_index(
    pos, 
    graph_methods = "knn",
    n_neighbors = None,
    n_radius = None,
    verbose: bool = True,
    ):
    """
    Build a graph on a single region (or the whole set).

    Inputs
    ------
    pos : np.ndarray, shape [N, d]
        Coordinate matrix.
    graph_methods : {"knn", "radius"}, default "knn"
        Graph construction method.
    n_neighbors : int, required if graph_methods == "knn"
        Number of neighbors (must be > 0).
    n_radius : float, required if graph_methods == "radius"
        Neighborhood radius (must be > 0).

    Output
    ------
    edge_index : torch.LongTensor, shape [2, E]
        Directed edges (i, j). Returns an empty tensor of shape [2, 0] if no edges.
    """
    # construct edge indexes in one region
    if graph_methods not in ["knn", "radius"]:
        raise ValueError("graph_methods must be either 'knn' or 'radius'")

    if graph_methods == "knn" and (n_neighbors is None or n_neighbors <= 0):
        raise ValueError("n_neighbors must be a positive integer for knn method")

    if graph_methods == "radius" and (n_radius is None or n_radius <= 0):
        raise ValueError("n_radius must be a positive value for radius method")

    edge_list = []
    if graph_methods == "knn":
        edge_index = knn_graph(torch.Tensor(pos),
                               k=n_neighbors,
                               batch=torch.LongTensor(np.zeros(pos.shape[0])),
                               loop=False)
    elif graph_methods == "radius":
        edge_index = radius_graph(torch.Tensor(pos),
                                  r=n_radius,
                                  batch=torch.LongTensor(np.zeros(pos.shape[0])),
                                  loop=False)
    for (i, j) in zip(edge_index[1].numpy(), edge_index[0].numpy()):
        edge_list.append([i, j])

    # optional: print average neighbors (directed)
    if verbose:
        N = pos.shape[0]
        E = len(edge_list)
        avg = (E / N) if N > 0 else 0.0
        print(f"Average neighbors per node (directed): {avg:.2f} "
              f"(edges={E}, nodes={N})")

    # return as [2, E] tensor
    if len(edge_list) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.LongTensor(edge_list).T


def get_expr_edge_index(
    expr,
    n_neighbors = 20,
    mode = "connectivity",
    metric = "correlation",
    include_self = False,
):
    """
    Build a KNN graph from a feature/expression matrix using scikit-learn.

    Inputs
    ------
    expr : np.ndarray, shape [N, p]
        Feature (expression) matrix.
    n_neighbors : int, default 20
        Number of neighbors.
    mode : {"connectivity", "distance"}, default "connectivity"
        Graph construction mode.
    metric : str, default "correlation"
        Distance/affinity metric passed to kneighbors_graph.
    include_self : bool, default False
        Whether to include self-edges.

    Output
    ------
    edges : list of tuple (i, j)
        Directed edges in COO order (row -> col) as a list of pairs.
    """
    adj = kneighbors_graph(
        expr,
        n_neighbors,
        mode = mode,
        metric = metric,
        include_self = include_self,
    )
    edge_list = list(zip(adj.tocoo().row, adj.tocoo().col))
    return edge_list


def edge_lists_intersection(edges1, edges2):
    """
    Direction-sensitive intersection of two edge sets.

    Inputs
    ------
    edges1 : list of (i, j) or torch.LongTensor of shape [2, E]
        First edge set.
    edges2 : list of (i, j) or torch.LongTensor of shape [2, E]
        Second edge set.

    Output
    ------
    edges : list of tuple (i, j)
        Intersection as a list of directed edges.
    """
    # support both list-of-pairs and torch.LongTensor of shape [2, E]
    def to_tuple_list(edges):
        if isinstance(edges, torch.Tensor):
            if edges.ndim != 2 or edges.shape[0] != 2:
                raise ValueError("Edge tensor must have shape [2, E].")
            arr = edges.t().cpu().numpy()  # [E, 2]
            return [tuple(map(int, e)) for e in arr]
        # assume iterable of pairs (list or np.ndarray rows)
        return [tuple(e) for e in edges]

    set1 = set(to_tuple_list(edges1))
    set2 = set(to_tuple_list(edges2))
    return list(set1 & set2)


def get_consensus_edges(spatial, 
                        *omics, 
                        target_neighbors=8, 
                        max_iter=20
                       )-> torch.LongTensor:
    """
    Binary-search a neighbor count so that the intersection between a spatial
    graph and a feature graph yields an average degree close to target_neighbors.

    Inputs
    ------
    spatial : np.ndarray, shape [N, d]
        Coordinate matrix.
    *omics : np.ndarray(s), each shape [N, p_k]
        One or more omics/feature matrices concatenated horizontally.
    target_neighbors : int, default 8
        Desired average number of neighbors in the intersection (0 < target < N).
    max_iter : int, default 20
        Maximum number of binary-search iterations.

    Output
    ------
    edge_index : torch.LongTensor, shape [2, E]
        Intersection edges (i, j) as a tensor. Empty tensor [2, 0] if none.
    """
    n_cells = spatial.shape[0]
    assert all(omic.shape[0] == n_cells for omic in omics), "Omics data dimension mismatch"
    assert 0 < target_neighbors < n_cells, "Invalid target_neighbors"
    low, high = 4, min(80, n_cells - 1)
    combined_omics = np.hstack(omics)

    for _ in range(max_iter):
        n_neighbors = (low + high) // 2
        edge_index_spatial = get_single_edge_index(spatial, n_neighbors=n_neighbors)  # [2, E] tensor
        edge_index_feat = get_expr_edge_index(combined_omics, n_neighbors=n_neighbors)  # list of pairs

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



def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    Parameters
    ----------
    X
        Input matrix
    Returns
    -------
    X_tfidf
        TF-IDF normalized matrix
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
    **kwargs,
	) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)

    Parameters
    ----------
    adata
        Input dataset
    n_components
        Number of dimensions to use
    use_highly_variable
        Whether to use highly variable features only, stored in
        ``adata.var['highly_variable']``. By default uses them if they
        have been determined beforehand.
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.utils.extmath.randomized_svd`

    Returns
    -------
    adata : anndata.AnnData
        The input AnnData object with LSI results stored in `adata.obsm["X_lsi"]`.
    """
    if "random_state" not in kwargs:
        kwargs["random_state"] = 0  # Keep deterministic as the default behavior
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
    adata,
    resolution: list = list(np.arange(0.01, 2.5, 0.01)),
):
    """
    Optimize Leiden clustering resolution by maximizing the Calinski–Harabasz score.

    Inputs
    ------
    adata : anndata.AnnData
        AnnData object whose .X matrix (samples × features) is used for scoring.
        This function will overwrite/replace adata.obs["leiden"] repeatedly during the search.
    resolution : list of float, default list(np.arange(0.01, 2.5, 0.01))
        Candidate Leiden resolution values to evaluate.

    Outputs
    -------
    res : float
        The resolution value with the highest Calinski–Harabasz score.
        Also prints: "Best resolution: {res}"

    Side Effects
    ------------
    - Runs sc.tl.leiden multiple times and leaves the *last* computed Leiden labels
      in adata.obs["leiden"].
    """
    scores = []
    for r in resolution:
        sc.tl.leiden(adata, resolution=r, flavor="igraph", n_iterations=2, directed=False)
        s = calinski_harabasz_score(adata.X, adata.obs["leiden"])
        scores.append(s)
    cl_opt_df = pd.DataFrame({"resolution": resolution, "score": scores})
    best_idx = np.argmax(cl_opt_df["score"])
    res = cl_opt_df.iloc[best_idx, 0]
    print("Best resolution: ", res)
    return res


def _priori_cluster(
    adata,
    eval_cluster_n: int = 7,
    res_min: float = 0.01,
    res_max: float = 2.5,
    res_step: float = 0.01,
):
    """
    Find a Leiden resolution that yields a target number of clusters.

    Inputs
    ------
    adata : anndata.AnnData
        AnnData object to be clustered. This function overwrites adata.obs["leiden"]
        during the search.
    eval_cluster_n : int, default 7
        Desired number of Leiden clusters.
    res_min : float, default 0.01
        Minimum resolution to try (inclusive).
    res_max : float, default 2.5
        Maximum resolution to try (inclusive, best-effort).
    res_step : float, default 0.01
        Step size between consecutive resolutions.

    Output
    ------
    res : float
        The first (highest) resolution in the generated range (searched in descending order)
        that produces exactly `eval_cluster_n` unique Leiden labels.
        Also prints: "Best resolution: {res}"

    Notes
    -----
    - Search order remains descending to match the original behavior.
    - If no resolution yields exactly `eval_cluster_n`, returns the last tried value
      (same behavior as before).
    """
    # basic validation
    if res_step <= 0:
        raise ValueError("res_step must be > 0")
    if res_max < res_min:
        raise ValueError("res_max must be >= res_min")

    # build candidate resolutions (try to include res_max despite float rounding)
    resolutions = np.arange(res_min, res_max, res_step)
    if resolutions.size == 0:
        raise ValueError("Empty resolution grid. Check res_min/res_max/res_step.")

    # search from high to low (original behavior)
    for res in sorted(resolutions.tolist(), reverse=True):
        sc.tl.leiden(adata, resolution=res, flavor="igraph", n_iterations=2, directed=False)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        if count_unique_leiden == int(eval_cluster_n):
            break

    print("Best resolution: ", res)
    return res


def mclust_R(
    adata, 
    num_cluster, 
    modelNames='EEE', 
    used_obsm="DIRAC_embed", 
    random_seed=2020,
    key_added="DIRAC",
):
    """Clustering using the R package 'mclust' via rpy2.

    Inputs
    ------
    adata : anndata.AnnData
        AnnData object. Embeddings must be in `adata.obsm[used_obsm]` (shape: N × d).
    num_cluster : int
        Number of clusters to fit in mclust (passed to R's Mclust).
    modelNames : str, default 'EEE'
        Mclust covariance model string (see mclust documentation).
    used_obsm : str, default 'emb_pca'
        Key in `adata.obsm` where the embedding used for clustering is stored.
    random_seed : int, default 2020
        Random seed for both NumPy and R (via set.seed in R).
    key_added : str, default 'mclust'
        Column name to store the resulting cluster labels in `adata.obs`.

    Outputs
    -------
    None
        Results are stored in-place in `adata.obs[key_added]` as categorical ints.

    Requirements & Side Effects
    ---------------------------
    - Requires rpy2 and R package 'mclust' installed and available.
    - Sets NumPy and R random seeds.
    - Populates `adata.obs[key_added]` with integer labels (as pandas 'category').
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


def seed_torch(seed=1029):
    """
    Set random seeds for Python, NumPy, and PyTorch (CPU & CUDA) for reproducibility.

    Inputs
    ------
    seed : int, default 1029
        Seed value used across Python's `random`, NumPy, and PyTorch.

    Outputs
    -------
    None

    Side Effects
    ------------
    - Sets environment variable PYTHONHASHSEED.
    - Calls torch.manual_seed / cuda.manual_seed / cuda.manual_seed_all.
    - Sets:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
      (may reduce performance but improves determinism where applicable).
    """
    random.seed(seed)    
    os.environ['PYTHONHASHSEED'] = str(seed)    
    np.random.seed(seed)    
    torch.manual_seed(seed)    
    torch.cuda.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed) 
    # if you are using multi-GPU.    
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
    Combine multiple single-modality AnnData objects (e.g., RNA, ATAC, ADT)
    that represent the same cells into one feature-concatenated AnnData.

    Inputs (model inputs)
    ---------------------
    adatas : Dict[str, AnnData]
        Mapping from modality name (e.g., 'RNA', 'ATAC', 'ADT') to AnnData.
        The dict insertion order determines the feature block order.
    prefixes : Optional[Dict[str, str]], default=None
        Per-modality prefixes for feature names (e.g., {"ADT": "ADT_"}).
        If missing, defaults to f"{mod.upper()}_".
    align_obs : bool, default=True
        If True, reindex every AnnData to match the first modality's `obs_names`
        (requires identical cell sets). If False, raises on any mismatch.
    preserve_obsm : Iterable[str], default=("spatial",)
        Keys in `.obsm` to copy from the first modality if present.
    dtype : np.dtype, default=np.float32
        Output matrix dtype.

    Outputs (model outputs)
    -----------------------
    Returns
    -------
    AnnData
        - X: **dense** NumPy array with features concatenated across modalities.
        - obs: from the first modality (after optional alignment).
        - var: DataFrame indexed by prefixed feature names, with:
            * feature_types: modality label for each feature.
            * original_feature: original feature name from its source AnnData.
        - uns:
            * modality_combine: string like "RNA+ADT+ATAC" (in input order).
            * modalities: dict mapping modality -> number of features.
        - obsm: selected keys (e.g., 'spatial') copied from the first modality.

    Raises
    ------
    ValueError
        If `adatas` is empty, or cells differ across modalities and cannot be
        aligned (when `align_obs=True`), or orders differ (when `align_obs=False`).
    """
    if not adatas:
        raise ValueError("`adatas` is empty. Provide at least one AnnData.")

    # Preserve insertion order of the input dict
    modalities = list(adatas.keys())
    ref_key = modalities[0]
    ref = adatas[ref_key]

    # Align / validate shared cells
    ref_index = ref.obs.index
    for k, ad in adatas.items():
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

    # Build dense blocks and var
    dense_blocks = []
    var_index = []
    feature_types = []
    original_features = []

    for mod in modalities:
        ad = adatas[mod]
        X = ad.X

        # Always convert to dense ndarray
        if hasattr(X, "toarray"):  # catches scipy.sparse matrices
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
        # dtype=dtype,
        uns={
            "modality_combine": "+".join(modalities),
            "modalities": {m: adatas[m].n_vars for m in modalities},
        },
    )

    # Optionally carry over selected obsm entries from the reference modality
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
    use_raw: bool = False
) -> List[str]:
    """
    Cluster Top Genes (ctg)

    Selects marker (differentially expressed) genes across clusters using Scanpy's
    `rank_genes_groups`, then returns a unique list of the top-ranked gene names.

    Model Inputs
    ------------
    - adata_sc : AnnData
        A single-cell AnnData object containing the expression matrix and metadata.
        Must include a categorical column in `.obs` that identifies clusters.
    - cluster_label : str
        The name of the column in `adata_sc.obs` that encodes cluster identities
        (e.g., "leiden", "louvain", or a custom column).
    - n_genes : int, optional (default: 150)
        The number of top-ranked genes to take per cluster before de-duplicating.
    - min_cells : int, keyword-only (default: 3)
        Minimum number of cells a gene must be expressed in to be retained prior
        to ranking.
    - method : {"wilcoxon", "t-test", "logreg"}, keyword-only (default: "wilcoxon")
        Statistical test used by `sc.tl.rank_genes_groups`.
    - use_raw : bool, keyword-only (default: False)
        Whether to use `adata.raw` for differential testing.

    Model Outputs
    -------------
    - List[str]
        A de-duplicated list of gene symbols ranked as markers across all clusters.
        The list aggregates the top `n_genes` genes from each cluster and returns
        the unique gene names (order not guaranteed).

    Processing Steps
    ----------------
    1) Copy the input AnnData to avoid in-place modification.
    2) Filter genes expressed in fewer than `min_cells` cells.
    3) Library-size normalize and log1p-transform the counts.
    4) Run `rank_genes_groups` grouped by `cluster_label`.
    5) Collect the top `n_genes` names per cluster and return the unique set.

    Raises
    ------
    ValueError
        If `cluster_label` is not found in `adata_sc.obs` or `n_genes <= 0`.
    """
    # --- Basic validation ---
    if cluster_label not in adata_sc.obs.columns:
        raise ValueError(
            f"`cluster_label` '{cluster_label}' not found in adata_sc.obs."
            f" Available columns: {list(adata_sc.obs.columns)}"
        )
    if n_genes <= 0:
        raise ValueError("`n_genes` must be a positive integer.")
    if method not in {"wilcoxon", "t-test", "logreg"}:
        raise ValueError("`method` must be one of: 'wilcoxon', 't-test', 'logreg'.")

    # --- Work on a copy to avoid side effects ---
    adata = adata_sc.copy()

    # --- Minimal preprocessing for DE ---
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # --- Differential expression ---
    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_label,
        method=method,
        use_raw=use_raw
    )

    # --- Extract top names per cluster; handle Scanpy's storage format robustly ---
    names = adata.uns["rank_genes_groups"]["names"]
    # Convert to a DataFrame with clusters as columns
    markers_df = pd.DataFrame(names)

    # Clip in case n_genes > available rows
    top_n = min(n_genes, markers_df.shape[0])
    top_df = markers_df.iloc[:top_n, :]

    # Flatten to 1D array, drop duplicates via np.unique, and convert to Python list
    unique_markers = list(np.unique(top_df.to_numpy().ravel()))

    return unique_markers