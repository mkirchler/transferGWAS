# transferGWAS

transferGWAS is a method for performing genome-wide association studies on whole images. This repository provides code to run your own transferGWAS on UK Biobank or your own data. transferGWAS has 3 steps: 1. pretraining, 2. feature condensation, and 3. LMM association analysis. Since the three steps require different compute infrastructure (GPU vs CPU server) and different parts can take longer time (e.g. pretraining can take a few days on a GPU), the three parts are kept separate. 

* **pretraining:** provides code for training your own models on retinal fundus scans. This is mostly for reproducibility purposes - we provide our trained models in `models`, and if you want to train on your own data you will probably want to adapt to that data.

* **feature condensation:** this is just a short script to go from trained model to low-dimensional feature condensation. If you want to run on your own data, you will maybe need to write a few lines of `pytorch` code to properly read in your data into a `Dataset` (there's no one-size-fits-all here, unfortunately).

* **LMM association analysis:** this part is a wrapper for the BOLT-LMM association analysis, including some basic preprocessing steps. If you have experience with performing GWAS, you probably won't need this part.

* **models:** here you can find our pretrained models.

* **simulation study:** This is the code for the simulation study.


## Python

All parts require python 3.6+, and all deep learning parts are built in `pytorch`. We recommend using some up-to-date version of anaconda and then creating a new environment from the `environment.yml`:
```bash
conda env create --file environment.yml
```

If you want to run part of the (non-deep learning) code on a CPU-only machine, use the `environment-cpu.yml` file on this machine (note that this won't install any of the `pytorch` libraries):
```bash
conda env create --file environment-cpu.yml
```

## Getting started

If you don't want to train your own network, just:

* start with **feature condensation**, either with ImageNet-pretrained CNN or with the EyePACS CNN in `models`; then
* run the **LMM association analysis** on those condensed embeddings.

If you do want to train your own network, first check out the **pretraining** part.

