Installation guide
==================

*********************
How to install DIRAC
*********************

To install DIRAC, make sure you have [PyTorch](https://pytorch.org/) and [PyG](https://pyg.org/) installed. 

Step 1: Set Up Conda Environment
---------------------------------
.. code-block:: bash
    :linenos:

    conda create -n dirac-env python=3.9 r-base=4.3.1 rpy2 r-mclust r-yarrr

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

Now you are all set. Proceed to `tutorials <tutorials.rst>`__ for how to use the ``DIRAC`` package.





