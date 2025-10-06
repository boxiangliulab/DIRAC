.. _quickstart:

QuickStart
==========

``DIRAC`` provides **horizontal** (label transfer across datasets/platforms) and **vertical** (multi-omics integration) analysis.  
This QuickStart focuses on two **simulated datasets** and walks you through the end-to-end workflow:

- **NSF** — spatial multi-omics (RNA, ADT, optional ATAC) with joint embedding, clustering, ARI evaluation, and optional subgraph training.  
- **scMultiSim** — horizontal annotation (RNA→RNA, RNA+ATAC→RNA+ATAC), confidence-based *novel type discovery*, and UMAP mixing checks.

What you’ll learn
-----------------
- Preprocess single-cell/spatial data (``normalize_total → log1p → scale``; optional PCA/LSI).  
- Build spatial graphs (k-NN; optional radius; multi-batch vs single-sample).  
- Train DIRAC (``annotate_app`` / ``integrate_app``) and write back embeddings/predictions.  
- Evaluate results (Accuracy/Precision/Recall/F1, ARI) and visualize (spatial, UMAP).  
- Use **confidence filtering** to mark low-confidence predictions as ``"unassigned"`` and catch **missing/novel** cell types.

Prerequisites
-------------
- Python ≥ 3.9, and: ``scanpy``, ``anndata``, ``numpy``, ``pandas``, ``matplotlib`` (and ``torch`` inside DIRAC).  
- Local clone of the DIRAC codebase (added to ``sys.path``).  
- (Optional) R + ``mclust`` for MCLUST clustering.

Data
----
- **NSF** (simulated spatial multi-omics): RNA/ADT (+ ATAC) ``.h5ad`` files.  
- **scMultiSim** (simulated multi-omics, e.g. *mask = 0.3* means ~30% random zeros in RNA/ATAC): ``source_*`` and ``target_*`` ``.h5ad`` files.

.. note::
   Place datasets under ``DIRAC-main/data/`` in folders referenced by the notebooks (see each notebook’s first cell for exact paths).

At a glance
-----------
1. Load reference/target AnnData.  
2. Preprocess per modality.  
3. Build spatial graphs.  
4. Pack data with ``_get_data(...)`` (optionally ``num_parts_*`` for subgraphs).  
5. Build model with ``_get_model(...)`` (e.g., ``opt_GNN="SAGE"`` or ``"GAT"``).  
6. Train (``_train_dirac_integrate`` or ``_train_dirac_annotate``).  
7. Evaluate + visualize; save ``.h5ad``/figures/metrics.

----

The following notebooks are included:

.. nbgallery::
   notebooks/run-NSF.ipynb
   notebooks/run-scMultiSim.ipynb

Notebook details
----------------

**notebooks/run-NSF.ipynb — Spatial multi-omics (NSF)**
   - Load **RNA** and **ADT** (optionally **ATAC**).  
   - Preprocess (HVGs/PCA for RNA; standardization for ADT; LSI optional for ATAC).  
   - Build spatial **k-NN** graph (tune ``n_neighbors``; radius graph optional).  
   - Two-omics integration (RNA+ADT), then extend to **three omics** (add ATAC).  
   - Optional **subgraph** training (``subgraph=True``; control ``num_parts``).  
   - Cluster with **MCLUST** or Leiden; compute **ARI** vs ground truth.  
   - Plot **spatial** maps and **UMAP**; save embeddings and outputs.

**notebooks/run-scMultiSim.ipynb — Horizontal annotation (scMultiSim)**
   - Single-modality (**RNA→RNA**) and dual-modality (**RNA+ATAC→RNA+ATAC**).  
   - Preprocess each modality; **concatenate** features and specify ``split_list``.  
   - Build **multi-batch** graph for source (if needed) + **single-sample** graph for target.  
   - Run ``annotate_app`` to **transfer labels**; write back embeddings/predictions.  
   - **Confidence filtering** (e.g., ``confidence_threshold=0.9``) to mark low-confidence cells as ``"unassigned"`` (novel/missing types).  
   - Optional **mixing check**: UMAP colored by *Omics* and cluster labels.  
   - Report metrics (Accuracy/Precision/Recall/F1) and **Unassigned Rate**; save results (NPZ/JSON, figures, ``.h5ad``).

Tips & troubleshooting
----------------------
- Ensure required fields exist: ``obs["cell.type"]`` (labels), ``obsm["spatial"]`` (coords), and ``obs["batch"]`` for multi-batch graphs.  
- Verify ``split_list`` aligns with feature concatenation (``(0, dim_RNA)``, ``(dim_RNA, dim_RNA+dim_ATAC)``).  
- Start with ``n_neighbors=8–12``; adjust for platform resolution/spot density.  
- Use subgraphs for large tissues (``num_parts_* = max(1, n//200)``).  
- Tune model defaults: ``n_hiddens=128``, ``n_outputs=64``, ``epochs=200–400``, balance via ``lamb``/``scale_loss``.  
- For portability, save arrays via **NPZ** and label maps via **JSON**.

