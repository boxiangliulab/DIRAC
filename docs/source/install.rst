Installation guide
==================

How to install DIRAC
====================

To install DIRAC, make sure you have [PyTorch](https://pytorch.org/) and [PyG](https://pyg.org/) installed.

Step 1: Set Up Conda Environment
---------------------------------
To set up the environment, run the following command:

.. code-block:: bash
    :linenos:

    conda create -n dirac-env python=3.9 r-base=4.3.1 rpy2 r-mclust r-yarrr

Step 2: Install PyTorch and PyG
-------------------------------
Activate the environment and install PyTorch and PyG. Adjust the installation commands based on your CUDA version, or choose the CPU version if necessary.

* General Installation Command

.. code-block:: bash
    :linenos:

    conda activate dirac-env
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
    pip install pyg_lib==0.3.1+pt21cu118 torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
    pip install torch_geometric==2.3.1

* Tips for Selecting the Correct CUDA Version
---------------------------------------------

To verify your CUDA version, you can run:

.. code-block:: bash
    :linenos:
    
    nvcc --version

Alternatively, use:

.. code-block:: bash
    :linenos:

    nvidia-smi

* Modify Installation Commands Based on CUDA Version
--------------------------------------------------

  - For CUDA 12.1
  ----------------
  .. code-block:: bash
      pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
      pip install pyg_lib==0.3.1+pt21cu121 torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
      pip install torch_geometric==2.3.1

  - For CPU-only Installation
  --------------------------------
  .. code-block:: bash
      pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
      pip install pyg_lib==0.3.1+pt21cpu torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
      pip install torch_geometric==2.3.1

Step 3: Install DIRAC
----------------------

To install DIRAC, run the following command:

.. code-block:: bash
    pip install sodirac

Step 4: Import DIRAC in Your Jupyter Notebooks or Scripts
--------------------------------------------------------

To use DIRAC in your code, import it as follows:

.. code-block:: python
    import sodirac as sd



.. note::
    To avoid potential dependency conflicts, installing within a
    `conda environment <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__
    is recommended.

Now you are all set. Proceed to `tutorials <tutorials.rst>`__ for how to use the ``DIRAC`` package.






