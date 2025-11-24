Installation Guide
==================

To install DIRAC, you must first have `PyTorch <https://pytorch.org/>`_ and `PyTorch Geometric (PyG) <https://pyg.org/>`_ installed.

Step 1: Set Up Conda Environment
--------------------------------
Start by setting up a `conda` environment. Run the following command:

.. code-block:: bash
    :linenos:

    conda create -n dirac-env python=3.9 r-base=4.3.1 rpy2 r-mclust r-yarrr

.. note::

   Any R version ``>= 4.1`` works.
   We use ``r-base=4.3.1`` in the example above.

Step 2: Install PyTorch and PyG
-------------------------------
Activate the environment and install PyTorch and PyG. Make sure to adjust the installation commands based on your CUDA version, or choose the CPU-only version if necessary.

.. warning::

   The commands below are examples based on **PyTorch 2.1.0**.
   You must choose the appropriate PyTorch build (CUDA or CPU)
   according to your own server hardware and driver setup.

   We recommend first selecting and installing PyTorch by following
   the official instructions at `PyTorch Get Started <https://pytorch.org/get-started/locally/>`__,
   and then installing the matching PyG wheels.

   The suffix in ``pyg_lib`` (e.g. ``+pt21cu118``, ``+pt21cu121``, ``+pt21cpu``)
   must match the PyTorch build you install; otherwise you may encounter
   "No matching distribution found" errors.

* **General Installation Command**:

.. code-block:: bash
    :linenos:

    conda activate dirac-env
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
    pip install pyg_lib==0.3.1+pt21cu118 torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
    pip install torch_geometric==2.3.1

* **Tips for Selecting the Correct CUDA Version**:
    
    To verify your CUDA version, you can run the following command:

    .. code-block:: bash
        :linenos:
        
        nvcc --version

    Alternatively, use:

    .. code-block:: bash
        :linenos:

        nvidia-smi

* **Modify Installation Commands Based on CUDA Version**:
    
    - For CUDA 12.1:

    .. code-block:: bash
        :linenos:
        
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
        pip install pyg_lib==0.3.1+pt21cu121 torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
        pip install torch_geometric==2.3.1

    - For CPU-only Installation:

    .. code-block:: bash
        :linenos:

        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
        pip install pyg_lib==0.3.1+pt21cpu torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
        pip install torch_geometric==2.3.1

Step 3: Install DIRAC
----------------------
After installing PyTorch and PyG, install the DIRAC package by running:

.. code-block:: bash
    :linenos:

    pip install sodirac

Step 4: Import DIRAC in Your Jupyter Notebooks or Scripts
--------------------------------------------------------
To use DIRAC in your code, import it as follows:

.. code-block:: python
    :linenos:

    import sodirac as sd

.. note::
    To avoid potential dependency conflicts, it is recommended to install DIRAC within a
    `conda environment <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__.

Now you are all set! Proceed to the `tutorials <tutorials.rst>`__ for guidance on how to use the ``DIRAC`` package.






