from anndata import read_h5ad
from anndata.io import read_csv, read_loom, read_text

from ._anntorchdataset import AnnTorchDataset
from ._datasets import (
    annotation_simulation,
    brainlarge_dataset,
    breast_cancer_dataset,
    cellxgene,
    cortex,
    dataset_10x,
    frontalcortex_dropseq,
    heart_cell_atlas_subsampled,
    mouse_ob_dataset,
    pbmc_dataset,
    pbmc_seurat_v4_cite_seq,
    pbmcs_10x_cite_seq,
    prefrontalcortex_starmap,
    purified_pbmc_dataset,
    retina,
    smfish,
    spleen_lymph_cite_seq,
    synthetic_iid,
)
from ._manager import AnnDataManager, AnnDataManagerValidationCheck
from ._phenomics import read_phenomics, setup_phenomics_anndata
from ._preprocessing import (
    add_dna_sequence,
    organize_cite_seq_10x,
    organize_multiome_anndatas,
    poisson_gene_selection,
    reads_to_fragments,
)
from ._read import read_10x_atac, read_10x_multiome

__all__ = [
    "AnnTorchDataset",
    "AnnDataManagerValidationCheck",
    "AnnDataManager",
    "poisson_gene_selection",
    "organize_cite_seq_10x",
    "pbmcs_10x_cite_seq",
    "spleen_lymph_cite_seq",
    "dataset_10x",
    "purified_pbmc_dataset",
    "brainlarge_dataset",
    "synthetic_iid",
    "pbmc_dataset",
    "cortex",
    "smfish",
    "breast_cancer_dataset",
    "mouse_ob_dataset",
    "retina",
    "prefrontalcortex_starmap",
    "frontalcortex_dropseq",
    "annotation_simulation",
    "read_h5ad",
    "read_csv",
    "read_loom",
    "read_text",
    "read_10x_atac",
    "read_10x_multiome",
    "heart_cell_atlas_subsampled",
    "organize_multiome_anndatas",
    "pbmc_seurat_v4_cite_seq",
    "add_dna_sequence",
    "reads_to_fragments",
    "cellxgene",
    "read_phenomics",
    "setup_phenomics_anndata",
]
