# 🔁 Reproducibility Instructions

This document provides the full set of instructions to reproduce our project results from scratch, including data setup, environment configuration, training, and evaluation.

---

## 🧱 Project Structure

```bash
.
├── configs/                   # Contains all the configurations to run your python scripts
├── data/                   # Contains code for dataset managment
├── dataset/                   # Contains raw and processed datasets
├── distributions/                   # Contains code for distributions
├── evaluate/                    # Contains code for evaluation
├── init/                    # Contains code for initialiazation
├── models/                    # Contains code for modules
├── ops/                    # Contains code for operations
├── out/                    # Contains the outputs of the program. This folder will be auto-generated
├── job_scripts/                    # Contains the all the slurm job
├── slurm_out/                    # Contains the outputs of the slurm job. This folder will be auto-generated
├── trained_models/                    # Contains the pretrained models
├── wandb/                    # Contains the outputs of the wandb scheduler. This folder will be auto-generated
├── requirements.txt        # Python dependencies
├── README.md               # README file
├── README_2.md              # The original README file
├── REPRO.md                # This file
```

---

## ⚙️ Environment Setup


Setup project by running the following commands:



```bash
# Example -- overwrite if needed
conda create -n rq-vae python=3.9
source activate rq-vae
pip install -r requirements.txt
```

or execute the `install_enviroment.job` script

---

## 📂 Download & Prepare Datasets

All your dataset is being downloaded and prepared by the code in the folder `data`. If you want to add more datasets follow their example.

---

## ⚙️ Configuration

Set your parameters in the config file before training. As a training configuration example see `configs/rqvae_amazon.gin` and for evaluation configuration example see `configs/decoder_amazon.gin`


---

## 🚀 5. Training

### Baselines

Run the following command to train the baseline:

```bash
python  train_rqvae.py configs/decoder_amazon.gin
```

Alternatively, execute the following slurm job:

```bash
sbatch job_scripts/run_training_amazon.job
```

---

## 📈 Evaluation

After training, evaluate all models with:

```bash
python train_decoder.py configs/decoder_amazon.gin
```

Alternatively, execute the following slurm job:

```bash
sbatch job_scripts/run_evaluation_amazon.job
```

---

## ⚙️ Gini Coefficient & Intra-list Diversity (ILD)

You can use the diversity metrics only if 
    1. You have a category and you have defined it in the .gin configuration
    2. You have not created the dataset

If you dont have both then delete the dataset folder and re-run it so that the categories can be created!

---

---
# Important Implementation Note
## Reproducibility Considerations
According to the creator of the open-sourced implementation used ([issue #41](https://github.com/EdoardoBotta/RQ-VAE-Recommender/issues/41)), exact numerical results are not reproducible. This is expected in this implementation and does not invalidate the reproducibility assessment. This likely is the result of inconsistent RQ-VAE training, where the utilisation of the codebook is not adequate.

---
## 📦 Dependencies / References

This project repository uses the following frameworks / refers to the following papers:

- [github repository](https://github.com/EdoardoBotta/RQ-VAE-Recommender)
- [paper](https://arxiv.org/abs/2305.05065)


