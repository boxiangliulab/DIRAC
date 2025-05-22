#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced spatial simulation with multi-omics data generation
"""
import random
import numpy as np
from pandas import get_dummies
from anndata import AnnData
from scipy.stats import norm, poisson
dtp = "float32"
random.seed(101)
import scanpy as sc
import sys
sys.path.append("/home/project/11003054/changxu/Projects/DIRAC/Section-2/nsf-paper-main")
from utils import misc, preprocess

# Enhanced pattern generation functions
def squares(size=12):
    """Generate square patterns of given size"""
    A = np.zeros([size,size])
    quarter = size//4
    A[quarter:3*quarter, quarter:3*quarter] = 1
    return A

def circles(size=12):
    """Generate circle patterns of given size"""
    A = np.zeros((size, size))
    center = size//2
    radius = size//3
    for i in range(size):
        for j in range(size):
            if (i - center)**2 + (j - center)**2 < radius**2:
                A[i, j] = 1
    return A

def triangles(size=12):
    """Generate triangle patterns of given size"""
    A = np.zeros((size, size))
    for i in range(size):
        A[i, :i+1] = 1
    return A

def stripes(size=12, orientation='horizontal'):
    """Generate stripe patterns"""
    A = np.zeros((size, size))
    if orientation == 'horizontal':
        for i in range(0, size, 2):
            A[i, :] = 1
    else:  # vertical
        for j in range(0, size, 2):
            A[:, j] = 1
    return A

def checkerboards(size=12):
    """Generate checkerboard patterns"""
    A = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                A[i, j] = 1
    return A

def crosses(size=12):
    """Generate cross patterns"""
    A = np.zeros((size, size))
    center = size//2
    thickness = max(1, size//6)
    A[center-thickness:center+thickness, :] = 1
    A[:, center-thickness:center+thickness] = 1
    return A


def ggblocks(nside=64, block_size = 32):
    """
    Generate large blocky spatial patterns where each pattern is a separate class.
    Includes a five-pointed star pattern.
    - Each pattern occupies a distinct block in the spatial grid
    - block_size must divide nside
    - Returns: (num_patterns, nside*nside) array where each row is one pattern
    """
    if nside % block_size != 0:
        raise ValueError("nside must be divisible by block_size")

    n_blocks = nside // block_size
    num_patterns = min(4, n_blocks**2)  # We'll generate up to 4 distinct patterns

    # Define pattern generators
    def square_pattern(size):
        A = np.zeros((size, size))
        q = size // 10
        A[q:-q, q:-q] = 1
        return A

    def cross_pattern(size, thickness=5):
        """
        Generate a cross pattern with adjustable thickness.
    
        Parameters:
        -----------
        size : int
            The size of the pattern (size x size)
        thickness : int
            The width of the cross bars (default: 3)
    
        Returns:
        --------
        numpy.ndarray
            A 2D array with the cross pattern
        """
        A = np.zeros((size, size))
        c = size // 2  # Center position
    
        # Calculate bar boundaries
        half_thickness = thickness // 2
        start = c - half_thickness
        end = c + half_thickness + (thickness % 2)  # Handle odd thickness
    
        # Make sure we stay within bounds
        start = max(0, start)
        end = min(size, end)
    
        # Draw thicker cross
        A[:, start:end] = 1  # Vertical bar
        A[start:end, :] = 1  # Horizontal bar
    
        return A

    def diag_pattern(size):
        A = np.eye(size)
        A += np.fliplr(np.eye(size))
        return (A > 0).astype(float)

    def circle_pattern(size):
        A = np.zeros((size, size))
        r = size // 2
        cx, cy = size // 2, size // 2
        for i in range(size):
            for j in range(size):
                if (i - cx) ** 2 + (j - cy) ** 2 < r ** 2:
                    A[i, j] = 1
        return A

    def star_pattern(size):
        """Generate a five-pointed star pattern"""
        A = np.zeros((size, size))
        center = size // 2
        radius = size // 2 - 2
        
        # Create star polygon coordinates
        angles = np.linspace(0, 2*np.pi, 6)[:-1]  # 5 points
        star_points = []
        for i, angle in enumerate(angles):
            # Outer point
            x = center + radius * np.cos(angle)
            y = center + radius * np.sin(angle)
            star_points.append((x, y))
            # Inner point (for star indentation)
            inner_angle = angle + 2*np.pi/10  # Offset by 36 degrees
            x = center + (radius/2.5) * np.cos(inner_angle)
            y = center + (radius/2.5) * np.sin(inner_angle)
            star_points.append((x, y))
        
        # Convert to integer coordinates
        star_points = np.array(star_points).astype(int)
        
        # Fill the polygon
        from matplotlib.path import Path
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        points = np.vstack((x.flatten(), y.flatten())).T
        path = Path(star_points)
        mask = path.contains_points(points)
        A = mask.reshape(size, size).astype(float)
        
        return A


    # Replace circle pattern with star pattern
    patterns = [square_pattern, circle_pattern, star_pattern, cross_pattern]
    
    # Initialize output array (num_patterns, nside*nside)
    output = np.zeros((num_patterns, nside*nside))

    for pattern_idx in range(num_patterns):
        # Create empty grid for this pattern
        grid = np.zeros((nside, nside))
        
        # Calculate block position for this pattern
        row = pattern_idx // n_blocks
        col = pattern_idx % n_blocks
        
        # Only place this pattern in its designated block
        if row < n_blocks and col < n_blocks:
            start_x = row * block_size
            start_y = col * block_size
            block_pattern = patterns[pattern_idx % len(patterns)](block_size)
            grid[start_x:start_x+block_size, start_y:start_y+block_size] = block_pattern
        
        # Store this pattern's grid
        output[pattern_idx, :] = grid.flatten()

    return output


def quilt(size=12):
    """Original quilt patterns"""
    patterns = [
        squares(size),
        crosses(size),
        circles(size),
        triangles(size)
    ]
    L = len(patterns)
    pattern_size = patterns[0].size
    A = np.zeros([L, pattern_size])
    for i, pattern in enumerate(patterns):
        A[i, :] = pattern.flatten()
    return A

def sqrt_int(x):
    z = int(round(x**.5))
    if x == z**2:
        return z
    else:
        raise ValueError("x must be a square integer")

def gen_spatial_factors(scenario="quilt", nside=36):
    """
    Generate spatial factors for different scenarios
    """
    if scenario == "quilt":
        A = quilt()
    elif scenario == "ggblocks":
        A = ggblocks(nside)
    elif scenario == "both":
        A1 = quilt()
        A2 = ggblocks(nside)
        A = np.vstack((A1, A2))
    else:
        raise ValueError("scenario must be 'quilt', 'ggblocks' or 'both'")
    
    unit = sqrt_int(A.shape[1])
    assert nside % unit == 0
    ncopy = nside // unit
    N = nside**2
    L = A.shape[0]
    A = A.reshape((L, unit, unit))
    A = np.kron(A, np.ones((1, ncopy, ncopy)))
    F = A.reshape((L, N)).T
    return F

def gen_spatial_coords(N):
    """Generate spatial coordinates"""
    X = misc.make_grid(N)
    X[:, 1] = -X[:, 1]  # make the display the same
    return preprocess.rescale_spatial_coords(X)

def gen_nonspatial_factors(N, L=3, nzprob=0.2, seed=101):
    """Generate non-spatial factors"""
    rng = np.random.default_rng(seed)
    return rng.binomial(1, nzprob, size=(N, L))

def gen_loadings(Lsp, Lns=3, Jsp=0, Jmix=500, Jns=0, 
                 expr_mean=20.0, mix_frac_spat=0.1, 
                 modality='RNA', seed=101):
    """
    Generate loadings matrix for different modalities
    """
    rng = np.random.default_rng(seed)
    J = Jsp + Jmix + Jns
    
    # Adjust parameters based on modality
    if modality == 'ADT':
        expr_mean = expr_mean  # ADT typically has higher counts
        mix_frac_spat = 0.2  # ADT often more spatially structured
    elif modality == 'ATAC':
        expr_mean = expr_mean  # ATAC typically has lower counts
        mix_frac_spat = 0.05  # Chromatin accessibility may be less spatially structured
    
    if Lsp > 0:
        w = rng.choice(Lsp, J, replace=True)
        W = get_dummies(w).to_numpy(dtype=dtp)
    else:
        W = np.zeros((J, 0))
    
    if Lns > 0:
        v = rng.choice(Lns, J, replace=True)
        V = get_dummies(v).to_numpy(dtype=dtp)
    else:
        V = np.zeros((J, 0))
    
    # Pure spatial features
    W[:Jsp, :] *= expr_mean
    V[:Jsp, :] = 0
    
    # Mixed features
    W[Jsp:(Jsp+Jmix), :] *= (mix_frac_spat * expr_mean)
    V[Jsp:(Jsp+Jmix), :] *= ((1 - mix_frac_spat) * expr_mean)
    
    # Pure non-spatial features
    W[(Jsp+Jmix):, :] = 0
    V[(Jsp+Jmix):, :] *= expr_mean
    
    return W, V

def add_gaussian_noise(data, snr=10, seed=101):
    """Add Gaussian noise to data with specified signal-to-noise ratio"""
    rng = np.random.default_rng(seed)
    signal_power = np.mean(data**2)
    noise_power = signal_power / snr
    noise = rng.normal(0, np.sqrt(noise_power), data.shape)
    return data + noise

# def add_gaussian_noise(data, mean=2, variance=0.5, seed=101):
#     """Add Gaussian noise to data with specified mean and variance
    
#     Parameters:
#     -----------
#     data : numpy.ndarray
#         Input data matrix
#     mean : float
#         Mean of the Gaussian noise (default: 2)
#     variance : float
#         Variance of the Gaussian noise (default: 0.5)
#     seed : int
#         Random seed for reproducibility
#     """
#     rng = np.random.default_rng(seed)
#     std_dev = np.sqrt(variance)  # Standard deviation is square root of variance
#     noise = rng.normal(mean, std_dev, data.shape)
#     return data + noise

def sim_multiomics(scenario, nside=36, nzprob_nsp=0.2, bkg_mean=0.2, 
                  nb_shape=10.0, seed=101, **kwargs):
    """
    Simulate multi-omics spatial data (RNA, ADT, ATAC)
    """
    # Generate common spatial structure
    if scenario == "both":
        F1 = gen_spatial_factors(nside=nside, scenario="ggblocks")
        F2 = gen_spatial_factors(nside=nside, scenario="quilt")
        F = np.hstack((F1, F2))
    else:
        F = gen_spatial_factors(scenario=scenario, nside=nside)
    
    rng = np.random.default_rng(seed)
    N = nside**2
    X = gen_spatial_coords(N)
    U = gen_nonspatial_factors(N, L=3, nzprob=nzprob_nsp, seed=seed)
    
    # Initialize empty AnnData object with correct number of observations
    adata = AnnData(X=np.empty((N, 0)))  # Initialize with N obs and 0 vars
    
    # Add spatial coordinates and factors
    adata.obsm["spatial"] = X
    adata.obsm["spfac"] = F
    adata.obsm["nsfac"] = U
    
    # Simulate each modality
    modalities = {
        'RNA': {'Jsp': 200, 'Jmix': 800, 'Jns': 200, 'expr_mean': 20},
        'ADT': {'Jsp': 50, 'Jmix': 100, 'Jns': 20, 'expr_mean': 100},
        'ATAC': {'Jsp': 200, 'Jmix': 800, 'Jns': 200, 'expr_mean': 10}
    }
    
    for modality, params in modalities.items():
        # Update params with any modality-specific kwargs
        mod_params = params.copy()
        mod_params.update(kwargs.get(modality, {}))
        
        # Generate loadings for this modality
        modality_seed = seed + abs(hash(modality)) % (2**32)
        W, V = gen_loadings(F.shape[1], Lns=U.shape[1], 
                    modality=modality, seed=modality_seed, 
                    **mod_params)
        
        # Simulate counts
        Lambda = bkg_mean + F @ W.T + U @ V.T
        r = nb_shape
        
        # Different count models for different modalities
        if modality == 'ATAC':
            # ATAC data often has many zeros and follows ZIP or ZINB
            counts = rng.poisson(Lambda)
            counts = add_gaussian_noise(counts, snr=20, seed = modality_seed)
            zero_mask = rng.random(Lambda.shape) < 0.5  # 30% additional zeros
            counts[zero_mask] = 0
        elif modality == 'RNA':
            counts = rng.negative_binomial(Lambda, r/(Lambda+r))
            counts = add_gaussian_noise(counts, snr=20, seed = modality_seed)
            zero_mask = rng.random(Lambda.shape) < 0.4
            counts[zero_mask] = 0
        else:
            counts = rng.negative_binomial(r, r/(Lambda+r))
            counts = add_gaussian_noise(counts, snr=20, seed = modality_seed)
        
        # Add modality-specific Gaussian noise
        # counts = add_gaussian_noise(counts, snr=20, seed = modality_seed)
        counts = np.clip(counts, 0, None).astype(int)
        
        # Store in AnnData layers
        adata.obsm[f"counts_{modality}"] = counts
        adata.uns[f"loadings_{modality}"] = np.hstack((W, V))
    
    # Shuffle indices
    idx = list(range(adata.shape[0]))
    random.shuffle(idx)
    adata = adata[idx, :].copy()
    
    return adata

def sim2anndata(locs, outcome, spfac, spload, nsfac=None, nsload=None):
    """
    Compatibility function to match original function signature
    """
    obsm = {"spatial": locs, "spfac": spfac, "nsfac": nsfac}
    varm = {"spload": spload, "nsload": nsload}
    ad = AnnData(outcome, obsm=obsm, varm=varm)
    ad.layers = {"counts": ad.X.copy()}
    idx = list(range(ad.shape[0]))
    random.shuffle(idx)
    ad = ad[idx, :]
    return ad

def sim(scenario, nside=36, nzprob_nsp=0.2, bkg_mean=0.2, 
       nb_shape=10.0, seed=101, multiomics=False, **kwargs):
    """
    Main simulation function that can handle both single and multi-omics
    """
    if multiomics:
        return sim_multiomics(scenario, nside=nside, nzprob_nsp=nzprob_nsp,
                            bkg_mean=bkg_mean, nb_shape=nb_shape,
                            seed=seed, **kwargs)
    else:
        # Original single-modality simulation
        if scenario == "both":
            F1 = gen_spatial_factors(nside=nside, scenario="ggblocks")
            F2 = gen_spatial_factors(nside=nside, scenario="quilt")
            F = np.hstack((F1, F2))
        else:
            F = gen_spatial_factors(scenario=scenario, nside=nside)
        
        rng = np.random.default_rng(seed)
        N = nside**2
        X = gen_spatial_coords(N)
        W, V = gen_loadings(F.shape[1], seed=seed, **kwargs)
        U = gen_nonspatial_factors(N, L=V.shape[1], nzprob=nzprob_nsp, seed=seed)
        Lambda = bkg_mean + F @ W.T + U @ V.T
        r = nb_shape
        Y = rng.negative_binomial(r, r/(Lambda+r))
        Y = add_gaussian_noise(Y, snr=20, seed=seed)
        Y = np.clip(Y, 0, None).astype(int)
        
        return sim2anndata(X, Y, F, W, nsfac=U, nsload=V)

adata = sim("ggblocks", multiomics=True, nside=64, seed=1)

def assign_unique_labels(matrix):
    row_strs = ['_'.join(map(str, row)) for row in matrix]
    unique_rows, labels = np.unique(row_strs, return_inverse=True)
    return labels

labels_S1 = assign_unique_labels(adata.obsm['spfac'])
adata.obs['ground_truth'] = labels_S1
adata.obs['ground_truth'] = adata.obs['ground_truth'].astype('category')
# sc.pl.embedding(adata, basis='spatial', color=['ground_truth'], s=100)

adata_RNA = AnnData(X=adata.obsm['counts_RNA'], obs=adata.obs)
adata_RNA.obsm['spatial'] = adata.obsm['spatial']
adata_ADT = AnnData(X=adata.obsm['counts_ADT'], obs=adata.obs)
adata_ADT.obsm['spatial'] = adata.obsm['spatial']
adata_ATAC = AnnData(X=adata.obsm['counts_ATAC'], obs=adata.obs)
adata_ATAC.obsm['spatial'] = adata.obsm['spatial']

# adata_RNA.write("/home/project/11003054/changxu/Projects/DIRAC/Section-2/simulations/sp_multi_omics/data/sim_RNA.h5ad")
# adata_ATAC.write("/home/project/11003054/changxu/Projects/DIRAC/Section-2/simulations/sp_multi_omics/data/sim_ATAC.h5ad")
# adata_ADT.write("/home/project/11003054/changxu/Projects/DIRAC/Section-2/simulations/sp_multi_omics/data/sim_ADT.h5ad")

def plot_figure(ad, res=0.1):
    sc.pp.filter_genes(ad, min_cells=3)
    sc.pp.normalize_total(ad)
    sc.pp.log1p(ad)
    sc.pp.scale(ad)
    sc.tl.pca(ad, n_comps=50)
    sc.pp.neighbors(ad)
    sc.tl.umap(ad)
    sc.tl.leiden(ad, resolution = res, key_added="Cluster")
    sc.pl.umap(ad, color=["Cluster"])
    sc.pl.embedding(ad, basis='spatial', color=['ground_truth',"Cluster"], s=100)

plot_figure(adata_RNA, res=0.000001)
plot_figure(adata_ADT, res=0.000001)
plot_figure(adata_ATAC, res=0.00001)