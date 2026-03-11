# HC-GAD

HC-GAD is an unsupervised graph anomaly detection framework for multiplex and heterogeneous graphs.  
The model follows a closed-loop design of unified feature projection, conflicting view construction, robust shared encoding, nonlinear consistency constraint, and automatic anomaly discrimination.

This repository is mainly used for thesis experiments. The code supports multiple `.mat` graph datasets, different encoder choices, dual-view augmentation, DCOR-based consistency learning, and several unsupervised scoring heads.

## Method Overview

The framework consists of the following key parts:

- Unified input projection:
  map heterogeneous node features from different views into a shared latent space with a linear projector.
- Dual-view augmentation:
  one masked reconstruction view and one context-mixed perturbation view are used to simulate missing information and structural-attribute mismatch.
- Shared graph encoder:
  supports `SGCN`, `GraphSAGE`, and `GAT`.
- Nonlinear consistency constraint:
  combines reconstruction objectives with DCOR loss, and optionally InfoNCE loss.
- Unsupervised scoring:
  supports reconstruction score, contextual score, structural prior score, and auxiliary scoring heads such as `IsolationForest`, `LOF`, and `kNN`.
- Automatic thresholding:
  selects anomaly thresholds without labeled supervision.

## Project Structure

- `config.py`: global hyperparameter configuration
- `data.py`: dataset loading and multiplex graph construction
- `augment.py`: dual-view augmentation
- `layers.py`: projector, encoders, and decoder
- `model.py`: HC-GAD model definition
- `loss.py`: reconstruction loss, DCOR loss, and InfoNCE loss
- `utils.py`: anomaly scoring, threshold selection, and evaluation
- `main.py`: training and inference entry
- `data/`: dataset directory

## Environment

Recommended environment:

- Python 3.9+
- PyTorch
- PyTorch Geometric
- NumPy
- SciPy
- scikit-learn

Example installation:

```bash
pip install torch torch_geometric numpy scipy scikit-learn
```

If GPU is available, please install the correct PyTorch version according to your CUDA environment.

## Quick Start

1. Put the dataset file into the `data/` directory.
2. Set the default dataset path and hyperparameters in `config.py`.
3. Run:

```bash
python main.py
```

The program will automatically:

- load the dataset
- train the model
- compute anomaly scores
- select an unsupervised threshold
- report evaluation metrics when labels are available

## Dataset Sources

The datasets used in this project can be obtained from the following sources:

- Retail / Retail_Rocket:
  [Alibaba Tianchi Competition](https://tianchi.aliyun.com/competition/entrance/231719/information/)
- Amazon:
  [CARE-GNN data repository](https://github.com/YingtongDou/CARE-GNN/tree/master/data)
- YelpChi:
  [DGL Fraud Dataset documentation](https://docs.dgl.ai/api/python/dgl.data.html#fraud-dataset)
- Facebook:
  the `.mat` benchmark version used in this project is consistent with the injected graph anomaly datasets collected in the General GAD benchmark resources:
  [Awesome Deep Graph Anomaly Detection](https://github.com/mala-lab/Awesome-Deep-Graph-Anomaly-Detection)
  and its linked [General GAD data folder](https://drive.google.com/drive/folders/1OpH6wM0T6zW6F3KODMl2A4a7h6nZ7eXq)
- ACM:
  the `.mat` benchmark version used in this project corresponds to the same General GAD benchmark collection:
  [Awesome Deep Graph Anomaly Detection](https://github.com/mala-lab/Awesome-Deep-Graph-Anomaly-Detection)
  and the linked [General GAD data folder](https://drive.google.com/drive/folders/1OpH6wM0T6zW6F3KODMl2A4a7h6nZ7eXq)
- BlogCatalog:
  the `.mat` benchmark version used in this project corresponds to the same General GAD benchmark collection:
  [Awesome Deep Graph Anomaly Detection](https://github.com/mala-lab/Awesome-Deep-Graph-Anomaly-Detection)
  and the linked [General GAD data folder](https://drive.google.com/drive/folders/1OpH6wM0T6zW6F3KODMl2A4a7h6nZ7eXq)

Note:
for `Facebook.mat`, `ACM.mat`, and `BlogCatalog.mat`, this repository uses the commonly used injected-anomaly `.mat` benchmark format rather than the original raw network files.

## Configuration

Most experiment settings are controlled in `config.py`, including:

- dataset path: `data_path`
- encoder type: `encoder_type`
- perturbation ratios: `attr_mask_ratio`, `edge_mask_ratio`, `mix_ratio`
- loss weights: `lambda_dcor`, `lambda_nce`
- score mode: `score_mode`
- auxiliary score fusion: `enable_aux_fusion`, `aux_score_type`, `aux_fusion_weight`

For large graphs, the following parameters are also important:

- `max_full_infer_nodes`
- `infer_batch_size`

## Output

When ground-truth labels are available, the program reports:

- `AUC`
- `AP`
- `Acc`
- `Precision`
- `Recall`
- `F1`

It also prints reference results from several scoring heads, such as:

- `ReconRef`
- `CMeanRef`
- `CStdRef`
- `ComboRef`
- `AuxRef`

If a dataset does not contain labels, the program skips AUC/AP and only reports the unsupervised threshold and the number of detected anomalies.

## Supported Data Formats

This code currently supports several `.mat` layouts, including:

- `Attributes / Label / Network`
- `features / label / homo`
- `feature + edge(object sparse matrices)`
- `net_*` multi-relation adjacency matrices

## Reproducibility

To improve reproducibility, the codebase uses:

- fixed random seeds
- deterministic PyTorch algorithms
- unified config-based experiment control

For repeated experiments, you can refer to the provided scripts:

- `run_amazon_10seeds.py`
- `run_yelpchi_10seeds.py`
- `run_retail_10seeds.py`

## Citation

If you use this code in your paper or project, please cite it together with your thesis title, author information, and final experimental setting.
