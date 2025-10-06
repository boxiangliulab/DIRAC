#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/17/23 2:58 PM
# @Author  : Chang Xu
# @File    : dataprep.py
# @Email   : changxu@nus.edu.sg


from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import scipy.sparse as sp
import torch
from scipy import stats
from scipy.spatial import distance
from torch_sparse import SparseTensor
import networkx as nx


class graph:
    """
    Graph builder over points (e.g., spatial coordinates or embeddings).

    Parameters
    ----------
    data : np.ndarray
        Shape [N, d]. Coordinates or feature vectors used for neighbor search.
    rad_cutoff : float
        Radius cutoff used when `distType == "Radius"`.
    k : int
        Number of neighbors used by kNN-style methods.
    distType : str, default "euclidean"
        Graph type / distance metric. Supported options include:
        - "spearmanr": top-k by Spearman correlation.
        - "BallTree": sklearn BallTree kNN.
        - "KDTree": sklearn KDTree kNN.
        - "kneighbors_graph": sklearn kneighbors_graph connectivity.
        - "Radius": sklearn NearestNeighbors with radius.
        - Any metric listed in SciPy `cdist` (see below in `graph_computing`).

    Attributes
    ----------
    data : np.ndarray
        Input data.
    distType : str
        Distance or method type.
    k : int
        kNN count.
    rad_cutoff : float
        Radius cutoff for "Radius" mode.
    num_cell : int
        Number of nodes (N).

    Notes
    -----
    Class name intentionally kept as `graph` (lowercase) to preserve external
    interface compatibility.
    """

    def __init__(
        self,
        data: np.ndarray,
        rad_cutoff: float,
        k: int,
        distType: str = "euclidean",
    ) -> None:
        super(graph, self).__init__()
        self.data = data
        self.distType = distType
        self.k = k
        self.rad_cutoff = rad_cutoff
        self.num_cell = data.shape[0]

    def graph_computing(self) -> List[Tuple[int, int]]:
        """
        Build an edge list according to the selected method/metric.

        Parameters
        ----------
        None (uses constructor attributes).

        Returns
        -------
        graphList : List[Tuple[int, int]]
        Directed edges (i, j) produced by the chosen neighbor rule.

        Notes
        -----
        Supported `distType` values:

        - "spearmanr": Uses Spearman correlation across rows of `self.data`
          (axis=1), then selects top-(k+1) indices per node and adds k edges.
        - "BallTree": Uses sklearn.neighbors.BallTree with k+1 query (self + k).
        - "KDTree": Uses sklearn.neighbors.KDTree with k+1 query (self + k).
        - "kneighbors_graph": Uses sklearn.neighbors.kneighbors_graph in
          connectivity mode, no self-loops.
        - "Radius": Uses sklearn.neighbors.NearestNeighbors(radius=rad_cutoff),
          excludes zero distances (self).
        - Any metric in the following SciPy list will trigger an on-the-fly
          `cdist`-based neighbor selection with a simple cutoff heuristic
          (boundary = mean + std of the top-k distances from each node):
          ["euclidean","braycurtis","canberra","mahalanobis","chebyshev","cosine",
           "jensenshannon","mahalanobis","minkowski","seuclidean","sqeuclidean",
           "hamming","jaccard","jensenshannon","kulsinski","mahalanobis","matching",
           "minkowski","rogerstanimoto","russellrao","seuclidean","sokalmichener",
           "sokalsneath","sqeuclidean","wminkowski","yule"]

        Raises
        ------
        ValueError
            If `distType` is not supported.
        """
        dist_list = [
            "euclidean",
            "braycurtis",
            "canberra",
            "mahalanobis",
            "chebyshev",
            "cosine",
            "jensenshannon",
            "mahalanobis",
            "minkowski",
            "seuclidean",
            "sqeuclidean",
            "hamming",
            "jaccard",
            "jensenshannon",
            "kulsinski",
            "mahalanobis",
            "matching",
            "minkowski",
            "rogerstanimoto",
            "russellrao",
            "seuclidean",
            "sokalmichener",
            "sokalsneath",
            "sqeuclidean",
            "wminkowski",
            "yule",
        ]

        if self.distType == "spearmanr":
            SpearA, _ = stats.spearmanr(self.data, axis=1)
            graphList: List[Tuple[int, int]] = []
            for node_idx in range(self.data.shape[0]):
                tmp = SpearA[node_idx, :].reshape(1, -1)
                res = tmp.argsort()[0][-(self.k + 1) :]
                for j in np.arange(0, self.k):
                    graphList.append((node_idx, res[j]))

        elif self.distType == "BallTree":
            from sklearn.neighbors import BallTree

            tree = BallTree(self.data)
            dist, ind = tree.query(self.data, k=self.k + 1)
            indices = ind[:, 1:]
            graphList = []
            for node_idx in range(self.data.shape[0]):
                for j in np.arange(0, indices.shape[1]):
                    graphList.append((node_idx, indices[node_idx][j]))

        elif self.distType == "KDTree":
            from sklearn.neighbors import KDTree

            tree = KDTree(self.data)
            dist, ind = tree.query(self.data, k=self.k + 1)
            indices = ind[:, 1:]
            graphList = []
            for node_idx in range(self.data.shape[0]):
                for j in np.arange(0, indices.shape[1]):
                    graphList.append((node_idx, indices[node_idx][j]))

        elif self.distType == "kneighbors_graph":
            from sklearn.neighbors import kneighbors_graph as sk_kng

            A = sk_kng(self.data, n_neighbors=self.k, mode="connectivity", include_self=False)
            A = A.toarray()
            graphList = []
            for node_idx in range(self.data.shape[0]):
                indices = np.where(A[node_idx] == 1)[0]
                for j in np.arange(0, len(indices)):
                    graphList.append((node_idx, indices[j]))

        elif self.distType == "Radius":
            from sklearn.neighbors import NearestNeighbors

            nbrs = NearestNeighbors(radius=self.rad_cutoff).fit(self.data)
            distances, indices = nbrs.radius_neighbors(self.data, return_distance=True)
            graphList = []
            for node_idx in range(indices.shape[0]):
                for j in range(indices[node_idx].shape[0]):
                    if distances[node_idx][j] > 0:
                        graphList.append((node_idx, indices[node_idx][j]))
            print("%.4f neighbors per cell on average." % (len(graphList) / self.data.shape[0]))

        elif self.distType in dist_list:
            graphList = []
            for node_idx in range(self.data.shape[0]):
                tmp = self.data[node_idx, :].reshape(1, -1)
                distMat = distance.cdist(tmp, self.data, self.distType)
                res = distMat.argsort()[: self.k + 1]
                tmpdist = distMat[0, res[0][1 : self.k + 1]]
                boundary = np.mean(tmpdist) + np.std(tmpdist)
                for j in np.arange(1, self.k + 1):
                    if distMat[0, res[0][j]] <= boundary:
                        graphList.append((node_idx, res[0][j]))
                    else:
                        pass

        else:
            raise ValueError(
                f"{self.distType!r} does not support. Disttype must in {dist_list}"
            )

        return graphList

    def List2Dict(self, graphList: List[Tuple[int, int]]) -> Dict[int, List[int]]:
        """
        Convert an edge list into an adjacency list dictionary.

        Parameters
        ----------
        graphList : List[Tuple[int, int]]
            Directed edges (i, j).

        Returns
        -------
        graphdict : Dict[int, List[int]]
            Adjacency list, mapping node -> list of out-neighbors.

        Notes
        -----
        Ensures all nodes [0, num_cell) appear as keys, including those with
        no neighbors (empty lists).
        """
        graphdict: Dict[int, List[int]] = {}
        tdict: Dict[int, str] = {}
        for edge in graphList:
            end1 = edge[0]
            end2 = edge[1]
            tdict[end1] = ""
            tdict[end2] = ""
            if end1 in graphdict:
                tmplist = graphdict[end1]
            else:
                tmplist = []
            tmplist.append(end2)
            graphdict[end1] = tmplist

        for i in range(self.num_cell):
            if i not in tdict:
                graphdict[i] = []

        return graphdict

    def mx2SparseTensor(self, mx: sp.spmatrix) -> SparseTensor:
        """
        Convert a SciPy sparse matrix to a torch `SparseTensor`.

        Parameters
        ----------
        mx : sp.spmatrix
            Input matrix (will be converted to COO, float32).

        Returns
        -------
        adj_t : SparseTensor
            Transposed sparse tensor (as returned by `.t()` in the original code).

        Notes
        -----
        Keeps values as float32. Matches original transpose behavior.
        """
        mx = mx.tocoo().astype(np.float32)
        row = torch.from_numpy(mx.row).to(torch.long)
        col = torch.from_numpy(mx.col).to(torch.long)
        values = torch.from_numpy(mx.data)
        adj = SparseTensor(row=row, col=col, value=values, sparse_sizes=mx.shape)
        adj_ = adj.t()
        return adj_

    def pre_graph(self, adj: sp.spmatrix) -> SparseTensor:
        """
        Preprocess an adjacency matrix with symmetric normalization.

        Parameters
        ----------
        adj : sp.spmatrix
            Sparse adjacency (no self loops).

        Returns
        -------
        adj_norm : SparseTensor
            Normalized adjacency as a `SparseTensor`.

        Notes
        -----
        Applies A_hat = D^{-1/2} (A + I) D^{-1/2}.
        """
        adj = sp.coo_matrix(adj)
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return self.mx2SparseTensor(adj_normalized)

    def main(self) -> Dict[str, Any]:
        """
        Build graph, preprocess adjacency, and package tensors for downstream use.

        Parameters
        ----------
        None

        Returns
        -------
        graph_dict : Dict[str, Any]
            {
              "adj_norm": SparseTensor,     # normalized adjacency (SparseTensor)
              "adj_label": torch.FloatTensor,  # adjacency with self-loops (dense FloatTensor)
              "norm_value": float           # scalar normalization factor (as in original)
            }

        Notes
        -----
        - Uses NetworkX to convert adjacency list to a SciPy adjacency matrix.
        - `adj_label` = A (no self-loop) + I.
        - The `norm_value` follows the exact original expression.
        """
        adj_mtx = self.graph_computing()
        graphdict = self.List2Dict(adj_mtx)
        adj_org = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))

        # Store original adjacency (without diagonal) and remove self-loops
        adj_pre = adj_org
        adj_pre = adj_pre - sp.dia_matrix((adj_pre.diagonal()[np.newaxis, :], [0]), shape=adj_pre.shape)
        adj_pre.eliminate_zeros()

        # Preprocessing
        adj_norm = self.pre_graph(adj_pre)
        adj_label = adj_pre + sp.eye(adj_pre.shape[0])
        adj_label = torch.FloatTensor(adj_label.toarray())
        norm = adj_pre.shape[0] * adj_pre.shape[0] / float(
            (adj_pre.shape[0] * adj_pre.shape[0] - adj_pre.sum()) * 2
        )

        graph_dict = {
            "adj_norm": adj_norm,
            "adj_label": adj_label,
            "norm_value": norm,
        }
        return graph_dict


def combine_graph_dict(dict_1: Dict[str, Any], dict_2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Block-diagonally combine two graph dictionaries returned by `graph.main()`.

    Parameters
    ----------
    dict_1 : Dict[str, Any]
        First graph dict with keys: "adj_norm" (SparseTensor), "adj_label" (FloatTensor),
        "norm_value" (float).
    dict_2 : Dict[str, Any]
        Second graph dict, same schema as `dict_1`.

    Returns
    -------
    graph_dict : Dict[str, Any]
        Combined graph dict with:
        - "adj_norm": SparseTensor (block-diagonal of the dense forms, then re-sparsified)
        - "adj_label": torch.FloatTensor (block-diagonal)
        - "norm_value": float (mean of the two input `norm_value`s)

    Notes
    -----
    Converts `adj_norm` to dense for block-diagonal composition, then back to
    `SparseTensor` to match the original behavior.
    """
    tmp_adj_norm = torch.block_diag(dict_1["adj_norm"].to_dense(), dict_2["adj_norm"].to_dense())

    graph_dict = {
        "adj_norm": SparseTensor.from_dense(tmp_adj_norm),
        "adj_label": torch.block_diag(dict_1["adj_label"], dict_2["adj_label"]),
        "norm_value": float(np.mean([dict_1["norm_value"], dict_2["norm_value"]])),
    }
    return graph_dict

