# scCSGRN

scCSGRN is a method for fine-grained inference of cell-type-specific gene regulatory networks (GRNs) from single-cell RNA sequencing data. It combines metacell-based expression denoising with a dual-stream feature learning model to predict whether a directed transcription factor-target gene pair has a regulatory relationship.

The corresponding manuscript is titled:

```text
scCSGRN: Fine-grained Inference of Cell-type Specific Gene Regulatory Networks from Single-cell RNA Sequencing Data via Dual-Stream Feature Learning
```

## Method Overview

scCSGRN follows three main stages:

1. Data preprocessing and metacell construction.
2. Dual-stream feature learning.
3. GRN inference and downstream analysis.

For each cell type, scCSGRN builds a metacell expression matrix to reduce sparsity and technical noise in scRNA-seq data. The model then learns two types of features:

- gene-pair expression features through a GEM-style representation and a VAE module;
- GRN structural features through a Graph Transformer module.

The learned expression and graph-structure features are concatenated and passed to a prediction layer for binary classification of TF-target gene pairs.

## Repository Structure

```text
scCSGRN/
├── README.md
├── data_preprocess/
│   ├── dataProcess_data.py      # Preprocess raw data into model inputs
│   ├── utils_data.py            # Data filtering, metacell construction, pair generation
│   ├── MetaQ-main/              # MetaQ code used for metacell construction
│   ├── rawData/                 # Raw scRNA-seq data, TF lists, and prior GRNs
│   └── scCSGRN_data/            # Processed data used by training
└── scCSGRN/
    ├── main.py                  # Training and evaluation entry point
    ├── model/TransformerE.py    # scCSGRN model
    ├── utils_GRN.py             # DGL graph construction and graph feature utilities
    ├── utils_gz.py              # Metrics and helper functions
    └── scCSGRN_GRN/             # Output directory for prediction results
```

## Environment

The current Python environment uses the following main package versions:

```text
Python            3.11.6
torch             2.3.0+cu121
dgl               2.2.1+cu121
torch_geometric   2.7.0
scanpy            1.9.6
numpy             1.26.0
pandas            2.3.3
scikit-learn      1.1.3
networkx          3.5
scipy             1.11.3
```

## Data Layout

Raw expression data are expected under:

```text
data_preprocess/rawData/scRNA-Seq/{dataset}/{cell_type}/ExpressionData.csv
data_preprocess/rawData/scRNA-Seq/{dataset}/{cell_type}/GeneOrdering.csv
```

Prior GRN files and TF lists are expected under:

```text
data_preprocess/rawData/Networks/human/
data_preprocess/rawData/Networks/mouse/
data_preprocess/rawData/Networks/human-tfs.csv
data_preprocess/rawData/Networks/mouse-tfs.csv
```

After preprocessing, each cell type should contain files such as:

```text
data_preprocess/scCSGRN_data/{dataset}/{cell_type}/celltype_grnboost2_GRN/Top500/
├── matrix.npy
├── label.npy
├── pair_list_id_tf.npy
├── pair_list_id_target.npy
├── Final_expression.csv
├── Final_gene_list.csv
├── Final_GRN_pos.csv
└── Info.csv
```

## Current Local Data

The current local copy keeps runnable raw and processed files for:

```text
Mtab-11821 / B_cells
```

Only the following cell type currently has complete local data:

```text
B_cells
```

Other dataset or cell-type directories may still exist, but their `ExpressionData.csv`, `GeneOrdering.csv`, or processed files may have been removed. To run preprocessing or training for other datasets/cell types, restore the corresponding raw and processed data first and update the dataset/cell-type lists in the scripts.

## Preprocessing

Run preprocessing from the `data_preprocess` directory:

```bat
cd \scCSGRN\data_preprocess
python dataProcess_data.py
```

Important: the current local copy keeps complete raw expression files only for `Mtab-11821 / Monocytes`. Before rerunning preprocessing on this local data, set:

```python
dataset_names = ['Mtab-11821']
```

and keep only `B_cells` in the `Mtab-11821` cell-type list.

The preprocessing script generates model-ready `.npy` and `.csv` files under:

```text
data_preprocess/scCSGRN_data/
```

## Training

Run training from the inner `scCSGRN` directory:

```bat
cd scCSGRN\scCSGRN
python main.py
```

The current default training configuration in `scCSGRN/main.py` is:

```python
dataset_names = ['Mtab-11821']
GRN_types = ['celltype_grnboost2_GRN']
Rank_nums = [500]
iteration = 5
```

For the current local data, the training script should run:

```text
B_cells
```

## Output

Training results are written to:

```text
scCSGRN/scCSGRN_GRN/
```

A typical output path is:

```text
scCSGRN/scCSGRN_GRN/Mtab-11821/B_cells/celltype_grnboost2_GRN/Top500/scCSGRN_GRN.csv
```

Each result file records metrics from repeated independent tests, including:

```text
AUROC
AUPR
Accuracy
F1
SPE
MCC
Precision
Recall
times
```

## Evaluation Protocol

The training script splits TF-target pairs into training, validation, and test sets in a 3:1:1 ratio. It repeats the experiment five times by default and reports common binary-classification metrics such as AUROC, AUPR, Accuracy, F1, Precision, Recall, specificity, and MCC.

## Notes

- The code uses relative paths in several places, so run scripts from the directories shown above.
- The current code uses `celltype_grnboost2_GRN` as the prior GRN directory name.
- `MetaQ-main` is used by preprocessing to construct metacell expression matrices.
- If you restore all six datasets mentioned in the manuscript, update `dataset_names` in both preprocessing and training scripts as needed.
