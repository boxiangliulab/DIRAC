.. _tutorials:

Tutorials
=========

``DIRAC`` provides end-to-end workflows on **real datasets** covering both vertical (multi-omics integration) and horizontal (cross-dataset label transfer) use cases. This section focuses on:

- **Glioblastoma** — Spatial **RNA + Protein (ADT)** integration, joint embedding, clustering, and spatial visualization.
- **Alzheimer’s disease (AD)** — Horizontal annotation of **stereo-seq bin100** samples using **DLPFC (10x Visium, normal)** as a reference, with **marker-driven training** and **confidence-based novel type discovery**.

What you’ll learn
-----------------
- Preprocess per modality (``normalize_total → log1p → scale``; HVGs/PCA for RNA).
- Build spatial graphs (k-NN; multi-batch for reference vs single-sample for target).
- Train DIRAC:
  - ``integrate_app`` for multi-omics integration (Glioblastoma).
  - ``annotate_app`` for label transfer (AD).
- Evaluate and visualize (spatial maps, optional UMAP; ARI / Accuracy / Precision / Recall / F1).
- Use **marker genes** (from the reference) and **confidence filtering** (e.g., 0.9) to mark low-confidence predictions as ``"unassigned"`` when references are incomplete.

Notebooks
---------
.. nbgallery::
   notebooks/run-Glioblastoma.ipynb
   notebooks/run-AD-samples.ipynb
   notebooks/DBit-seq-reproduce.ipynb
