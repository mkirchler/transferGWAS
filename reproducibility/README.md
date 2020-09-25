# Reproducing paper results

Here we provide instructions on how to reproduce the results from our paper. In order to do so, you will need to have access to UK Biobank data; specifically, you will need genetic microarray and imputed data, retinal fundus imaging data, and phenotypic data. 

We provide the list of individuals we used in our analysis (`indiv.txt`), as well as the filenames of all images we used (in `both.csv`).

## Preprocessing data

### Genetic data
You can preprocess all genetic data with `../run_bolt/preprocess_ma.sh` and `../run_bolt/preprocess_img.sh` (insert `INDIV=../reproducibility/indiv.txt`, and set the input and output files correspondingly)

### Covariate data
You can preprocess all covariate data with `../run_bolt/preprocess_cov.py`, just pass your covariate file, `indiv.txt` and keep the other parameters as default.

### Image data
Use 
```bash
python resize.py PATH_TO_LEFT PATH_TO_OUT --size 672
python resize.py PATH_TO_RIGHT PATH_TO_OUT --size 672
```
where `PATH_TO_LEFT` and `PATH_TO_RIGHT` are the paths to left and right retinal fundus directories (UKB id 21015 and 21016, respectively). Note that both will be merged into the same `PATH_TO_OUT`.


## Running the GWAS

### Step 1 - pre-training

We provide the trained weights for our EyePACS-trained ResNet50 (see `../download_models.sh`). Alternatively, you can train the network from scratch again, following the steps in `../multi-task-pretraining`. The weights for the ImageNet-trained ResNet50 are embedded in the pytorch library.

### Step 2 - feature condensation

Use `../export_embeddings/export_embeddings.py` with `both.csv` as input (see `README.md` therein).

### Step 3 - BOLT-LMM association analysis

In `../run_bolt/`, fill out the `config.toml` file (especially the `.bed` and optionally `.bgen` files) and then run the `run_gwas.py` script.
