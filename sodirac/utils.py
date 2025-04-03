#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/17/23 2:58 PM
# @Author  : Chang Xu
# @File    : utils.py
# @Email   : changxu@nus.edu.sg

import torch
import numpy as np
import anndata
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
from typing import Union, Callable, Any, Iterable, List, Optional
from sklearn.metrics import pairwise_distances
from torch_geometric.nn import knn_graph, radius_graph
import anndata


def make_one_hot(
    labels: torch.LongTensor,
    C=2,
	) -> torch.FloatTensor:
    """
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.LongTensor or torch.cuda.LongTensor
        [N, 1], where N is batch size.
        Each value is an integer representing correct classification.
    C : int
        number of classes in labels.

    Returns
    -------
    target : torch.FloatTensor or torch.cuda.FloatTensor
        [N, C,], where C is class number. One-hot encoded.
    """
    if labels.ndimension() < 2:
        labels = labels.unsqueeze(1)
    one_hot = torch.zeros(
        [
            labels.size(0),
            C,
        ],
        dtype=torch.float32,
        device=labels.device,
    )
    target = one_hot.scatter_(1, labels, 1)

    return target


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


def get_multi_edge_index(
    pos, 
    regions, 
	graph_methods = "knn",
	n_neighbors = None,
	n_radius = None,
    ):
    # construct edge indexes when there is region information
	if not isinstance(pos, np.ndarray) or not isinstance(regions, np.ndarray):
		raise ValueError("pos and regions must be numpy arrays")

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
	return edge_list


def get_single_edge_index(
    pos, 
	graph_methods = "knn",
	n_neighbors = None,
	n_radius = None,
    ):
    # construct edge indexes in one region
    if not isinstance(pos, np.ndarray):
        raise ValueError("pos must be a numpy array")

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
    return edge_list


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
	resolution: list = list(np.arange(0.1, 2.5, 0.01)),
	):
	scores = []
	for r in resolution:
		sc.tl.leiden(adata, resolution=r)
		s = calinski_harabasz_score(adata.X, adata.obs["leiden"])
		scores.append(s)
	cl_opt_df = pd.DataFrame({"resolution": resolution, "score": scores})
	best_idx = np.argmax(cl_opt_df["score"])
	res = cl_opt_df.iloc[best_idx, 0]
	print("Best resolution: ", res)
	return res



def _priori_cluster(
	adata,
	eval_cluster_n=7,
	):
	for res in sorted(list(np.arange(0.03, 2.5, 0.01)), reverse=True):
		sc.tl.leiden(adata, random_state=0, resolution=res)
		count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
		if count_unique_leiden == eval_cluster_n:
			break
	print("Best resolution: ", res)
	return res


def mclust_R(
	adata, 
	num_cluster, 
	modelNames='EEE', 
	used_obsm='emb_pca', 
	random_seed=2020,
	key_added = "mclust",
	):
	"""\
	Clustering using the mclust algorithm.
	The parameters are the same as those in the R package mclust.
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



# def mclust_R(
#     adata, 
#     num_cluster, 
#     modelNames='EEE', 
#     used_obsm='emb_pca', 
#     random_seed=2020,
#     key_added="mclust"
# ):
#     """\
#     Clustering using the mclust algorithm.
#     The parameters are the same as those in the R package mclust.
#     """
#     import numpy as np
#     import rpy2.robjects as robjects
#     from rpy2.robjects import r, pandas2ri, numpy2ri
#     import pandas as pd

#     # Set seed for reproducibility
#     np.random.seed(random_seed)
    
#     # Activate numpy2ri converter
#     numpy2ri.activate()
    
#     # Load mclust library in R
#     r.library("mclust")
    
#     # Set random seed in R
#     r['set.seed'](random_seed)
    
#     # Get the Mclust function from R
#     rmclust = r['Mclust']
    
#     # Check if used_obsm exists in adata
#     if used_obsm not in adata.obsm:
#         raise ValueError(f"{used_obsm} not found in adata.obsm")
    
#     # Convert the data to R format
#     pca_data = adata.obsm[used_obsm]
#     r_pca_data = numpy2ri.numpy2rpy(pca_data)
    
#     # Run Mclust
#     try:
#         res = rmclust(r_pca_data, num_cluster, modelNames)
#     except Exception as e:
#         raise RuntimeError(f"Mclust clustering failed: {e}")
    
#     # Check if the result is NULL
#     if res.rclass[0] == "NULL":
#         raise ValueError("Mclust returned NULL, indicating the clustering did not succeed.")
    
#     # Extract clustering results
#     mclust_res = np.array(res[-2])
    
#     # Store the results in adata.obs
#     adata.obs[key_added] = pd.Categorical(mclust_res.astype('int'))














