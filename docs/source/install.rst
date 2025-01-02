Installation guide
==================


************
Main package
************

The ``DIRAC`` package can be installed via conda using one of the following commands:

.. code-block:: bash
    :linenos:

    pip install sodirac

.. note::
    To avoid potential dependency conflicts, installing within a
    `conda environment <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__
    is recommended.


*********************
Optional dependencies
*********************

Some functions in the ``DIRAC`` package use metacell aggregation via k-Means clustering,
which can receive significant speed up with the `faiss <https://github.com/facebookresearch/faiss>`__ package.

You may install ``faiss`` following the official `guide <https://github.com/facebookresearch/faiss/blob/main/INSTALL.md>`__.

Now you are all set. Proceed to `tutorials <tutorials.rst>`__ for how to use the ``DIRAC`` package.





## How to install DIRAC

To install DIRAC, make sure you have [PyTorch](https://pytorch.org/) and [PyG](https://pyg.org/) installed. For more details on dependencies, refer to the `environment.yml` file.

### Step 1: Set Up Conda Environment
```
conda create -n dirac-env python=3.9 r-base=4.3.1 rpy2 r-mclust r-yarrr
```

### Step 2: Install PyTorch and PyG

Activate the environment and install PyTorch and PyG. Adjust the installation commands based on your CUDA version or choose the CPU version if necessary.

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
