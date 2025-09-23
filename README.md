# README

## Description
This repository contains the main scripts used for the analyses presented in the paper:  
**“Exploring ProtBERT embedding variants for predicting AAV vector viability in machine-guided protein design.”**

### Scripts
- `01_Data_gather_dupl_elim_and_data_split.py` – Data gathering, preprocessing, and splitting.  
- `02_Mutation_landscape_analysis_viability_and_design_strategy.py` – Analysis of mutation landscapes for sequence groups based on viability and design strategy.  
- `03_OHE_generation.py` – One-hot encoding (OHE) generation.  
- `04_Final_reps_generation_sampling.py` – Generation of ProtBERT embeddings and 30 random samplings to create the 30 data splits.  
- `05_LogR.py` – Logistic Regression classifier for supervised learning tasks.  
- `06_Pairs_analysis_excluded.py` – Analysis of model-representation pairs on excluded sequences (a similar script exists for included sequences).  
- `07_Evaluate_mutations_on_embeddings.py` – Controlled mutation schemes and their impact on embedding differences.  
- `08_Fine_tuning.py` – Fine-tuning architecture for task-specific adaptation of embeddings.  
- `09_Visualizing_fine_tuned_embeddings.py` – Visualization of fine-tuned embeddings.  

---

## Note on Included Scripts
This repository is a curated version of the code used in the study. It contains the essential scripts required to reproduce the main analyses, while internal variations, exploratory scripts, or intermediate versions are not included.

- Some scripts represent generalizable workflows, while variations used in internal testing or intermediate steps may have been omitted.  
- For example, only the Logistic Regression classifier is provided, but the workflow can be adapted to other models by modifying relevant parameters.  
- Similarly, only the primary embedding generation scripts are included; alternative concatenation strategies or later variations are not provided, but the main scripts capture the essential processing logic.  
- Mutation landscape analysis is provided for selected sequence groups; additional analyses (e.g., representation-model pair groups) were performed in separate scripts that are not included.  

---

## Usage
1. **Install dependencies**  
   Ensure all required Python packages are installed. A `requirements.txt` file is provided for easy installation:  
   ```bash
   pip install -r requirements.txt
2. **Run scripts in order**  
   Follow the order of scripts as described in the paper to reproduce the main analyses and figures.  

3. **Configure input/output paths**  
   Update input and output directories in the scripts as needed.  

---

## Authors
- Ana Filipa Rodrigues  
- Lucas Ferraz  
- Pedro Giesteira Cotovio  
- Cátia Pesquita  

---

## Acknowledgments
This work was supported by the **LASIGE Research Unit** (ref. UID/00408/2025) and partially supported by the **CancerScan project**, funded by the European Union’s Horizon Europe Research and Innovation Action (EIC Pathfinder Open) under grant agreement No. **101186829**.  

Pedro Cotovio and Lucas Ferraz acknowledge **Fundação para a Ciência e a Tecnologia** for PhD grants **2022.10557.BD** and **2025.04034.BD**, respectively.

