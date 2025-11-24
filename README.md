# DIRAC (Domain Invariant Respresentation through Adversatial Calibration)

[![stars-badge](https://img.shields.io/github/stars/boxiangliulab/DIRAC?logo=GitHub&color=yellow)](https://github.com/boxiangliulab/DIRAC/stargazers)
[![pypi-badge](https://img.shields.io/pypi/v/sodirac)](https://pypi.org/project/sodirac)
[![docs-badge](https://readthedocs.org/projects/scglue/badge/?version=latest)](https://rundirac.readthedocs.io/en/latest/?badge=latest)
[![build-badge](https://github.com/gao-lab/GLUE/actions/workflows/build.yml/badge.svg)](https://github.com/EsdenRun/DIRAC/actions/workflows/build.yml)
[![coverage-badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/Jeff1995/e704b2f886ff6a37477311b90fdf7efa/raw/coverage.json)](https://github.com/EsdenRun/DIRAC/actions/workflows/build.yml)
[![license-badge](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Spatially resolved integration of multi-omics with DIRAC highlights cell-specific remodeling

DIRAC is a Python package, written in [PyTorch](https://pytorch.org/) and based on [Scanpy](https://scanpy.readthedocs.io/en/stable/).

DIRAC is a graph neural network to integrate spatial multi-omic data into a unified domain-invariant embedding space and to automate cell-type annotation by transferring labels from reference spatial or single-cell multi-omic data.

DIRAC primarily includes two integration paradigms: vertical integration and horizontal integration, which differ in their selection of anchors. In vertical integration, multiple data modalities from the same cells are jointly analyzed, using cell correspondences in single-cell data or spot correspondences in spatial data as anchors for alignment. In horizontal integration, the same data modality from distinct groups of cells is aligned using genomic features as anchors. The best way to familiarize yourself with DIRAC is to check out [Jupyter notebooks](https://github.com/boxiangliulab/DIRAC_notebook), [our tutorial](https://github.com/boxiangliulab/DIRAC/tree/main/docs/source/notebooks), and [our documentation](https://dirac-tutorial.readthedocs.io/en/latest/).


![Model architecture](https://raw.githubusercontent.com/boxiangliulab/DIRAC/main/docs/source/_static/Workflow.png)

For more details, please check out our [publication](https://github.com/EsdenRun/DIRAC).

## Directory structure

```
.
├── sodirac                 # Main Python package
├── data                    # Data files
├── docs                    # Documentation files
├── environment.yaml        # Reproducible Python environment via conda
├── requirements.yaml       # Python packages required for issuing DIRAC
├── LICENSE
└── README.md
```

## How to install DIRAC

To install DIRAC, make sure you have [PyTorch](https://pytorch.org/) and [PyG](https://pyg.org/) installed. For more details on dependencies, refer to the `environment.yml` file.

### Step 1: Set Up Conda Environment
```
conda create -n dirac-env python=3.9 r-base=4.1.3 rpy2 r-mclust r-yarrr
```
> - **R version:** Any R version `>= 4.1` works. We use `r-base=4.1.3` in the example above.

### Step 2: Install PyTorch and PyG

Activate the environment and install PyTorch and PyG. Adjust the installation commands based on your CUDA version or choose the CPU version if necessary.

> **Important:**
> - The commands below are **examples** based on **PyTorch 2.1.0**. You should choose the appropriate PyTorch build (CUDA or CPU) according to your own server hardware and driver setup.
> - We recommend first selecting and installing PyTorch by following the official instructions at [PyTorch Get Started](https://pytorch.org/get-started/locally/), and then installing the matching PyG wheels.
> - The suffix in `pyg_lib` (e.g. `+pt21cu118`, `+pt21cu121`, `+pt21cpu`) **must match** the PyTorch build you install.

* General Installation Command
```
conda activate dirac-env
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install pyg_lib==0.3.1+pt21cu118 torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch_geometric==2.3.1
```
* Tips for selecting the correct CUDA version
  - Run the following command to verify CUDA version:
  ```
  nvcc --version
  ```
  - Alternatively, use:
  ```
  nvidia-smi
  ```
* Modify installation commands based on CUDA
  - For CUDA 12.1
    ```
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    pip install pyg_lib==0.3.1+pt21cu121 torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
    pip install torch_geometric==2.3.1
    ```
  - For CPU-only
    ```
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
    pip install pyg_lib==0.3.1+pt21cpu torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
    pip install torch_geometric==2.3.1
    ```

### Step 3: Install dirac from shell
```
    pip install sodirac
```

### Step 4: Import DIRAC in your jupyter notebooks or/and scripts 
```
    import sodirac as sd
```

> Installing within a
> [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
> is recommended.


## Usage

Please checkout the documentations and tutorials at
[dirac.readthedocs.io](https://dirac-tutorial.readthedocs.io/en/latest/) and [dirac.reproducing.io](https://github.com/boxiangliulab/DIRAC_notebook).
