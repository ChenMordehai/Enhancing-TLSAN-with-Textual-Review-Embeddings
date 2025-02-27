# Enhancing-TLSAN-with-Textual-Review-Embeddings

This repository contains the implementation of our study on enhancing the [Time-aware Long- and Short-term Attention Network (TLSAN)](https://www.sciencedirect.com/science/article/pii/S0925231221002605) model by incorporating textual review embeddings to improve recommendation systems.

## Overview

Recommendation systems have become integral to various online platforms, assisting users in discovering content that aligns with their preferences. The TLSAN model has been a significant advancement in this domain, effectively capturing temporal dynamics and latent semantic relationships. However, to further refine its performance, integrating textual data from item reviews can provide deeper insights into item characteristics.

In this project, we propose an enhancement to the TLSAN model by incorporating textual review embeddings. By transforming review texts into dense vector representations, we aim to capture the semantic nuances of item characteristics, thereby improving the accuracy and relevance of recommendations.

## Repository Structure

The repository is organized as follows:

- `README.md`: This document provides an overview of the project, its structure, and instructions for usage.

- `replicated_results/`: Contains the results replicated from the original TLSAN model for baseline comparison.
  - `tlsan/`: Directory with the original TLSAN model's outputs.
  - `exp_train_out_jobs.zip`: Archived training output, containing experiments and results of other SOTA models.

- `tlsan_text/`: Directory housing the enhanced TLSAN model with textual review embeddings.
  - `add_embedding.py`
  - `build_dataset.py`
  - `concate.py`
  - `input.py`: Handles input data formatting and batching for training and evaluation.
  - `model.py`: Defines the architecture of the enhanced TLSAN model.
  - `remap_id.py`
  - `tlsan_text_out_train_and_eval.zip`: Archived outputs from training and evaluation of the enhanced model.
  - `train.py`: Script to train the enhanced TLSAN model.

## Getting Started

To replicate our study or apply the enhanced TLSAN model to your dataset, follow these steps:

1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/ChenMordehai/Enhancing-TLSAN-with-Textual-Review-Embeddings.git
   cd Enhancing-TLSAN-with-Textual-Review-Embeddings
   ```

2. **Install Dependencies**: Ensure you have Python 3.5 installed. Install the required packages using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**: Place your dataset in the `data/` directory. Use `build_dataset.py` to preprocess the data and generate textual embeddings:
   ```bash
   python tlsan_text/build_dataset.py --review_emb_df data/your_dataset.csv --dataset data/your_dataset.pkl
   ```

4. **Train the Model**: Train the enhanced TLSAN model using:
   ```bash
   python tlsan_text/train.py --dataset 'Name of the dataset to process'
   ```


## Dependencies

- Python 3.5
- PyTorch
- NumPy
- Pandas

For a complete list, refer to `requirements.txt`.


## Acknowledgments

We extend our gratitude to the developers of the [original TLSAN model](https://www.sciencedirect.com/science/article/pii/S0925231221002605) and the contributors to the open-source libraries utilized in this project.

---
