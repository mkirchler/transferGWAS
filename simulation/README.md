# Simulation study for transferGWAS

This is the code for the simulation study on synthetic images.
The simulation study is divided into several steps:

0. You need to have a trained StyleGAN2. For retinal fundus images we provide the trained weights in `../models/stylegan2_healthy` (has to be downloaded with `../download_models.sh`). If you want to run the simulation on different data, you can check the documentation of `stylegan2_pytorch` (here we use version `0.17.11`; using newer versions might require a partial re-write of code in step 2) on how to train your own model. In this simulation study by default we use ImageNet-pretrained networks for the transferGWAS part, but you can use another network as well (using `../pretraining/`) (GPU required).
1. In the first stage of the simulation study, we create latent codes from genetic data. You need to have access to preprocessed genetic data (CPU-only).
2. In the second step we create synthetic images from the latent codes using the StyleGAN2 (GPU required)
3. In the third step, we condense the embeddings; this corresponds to stage 2 in the transferGWAS framework (GPU required).
4. In the fourth step, we perform the GWAS on the condensed embeddings; this corresponds to stage 3 in the transferGWAS framework (CPU-only).
5. Finally, we summarize power and type-1 error rate.


### Setup and run

To run the whole simulation, you will first need to prepare the genetic and covariate data as indicated in the `../lmm/README.md`. If you only want to create the synthetic dataset it's enough to only prepare the `.bed` files.

You can configure the pipeline via the `config.toml` file. Importantly, you will need to fill in the `bed` and `fam` file for the genetic data, as well as potentially the covariate data (`cov`) for the GWA analysis (if you already created the covariates as in `../lmm/preprocessing_cov.py`, the default values should work). All other parameters are set to reasonable defaults.

The whole pipeline can be run with:
```bash
python run_simulation.py config.toml --stages 1 2 3 4 --verbose
```
or you can only run any subset of stages (make sure that all previous stages have finished before):
```bash
python run_simulation.py config.toml --stages 3 4 --verbose
```

Note that this will always run **all** seeds, sample-sizes, and explained variances. I.e., if you specify 10 `seeds`, 3 `sample_sizes` and 3 `exp_vars` in your `config.toml`, this will run `3*3*10=90` simulations. If you specified `first_pc = 0` and `last_pc = 9` this will require 900 runs of BOLT-LMM (i.e. it will take some time).

If your machine has multiple GPUs, the code will use the first listed GPU by default (necessary for stages 2 & 3). To change that, it's easiest to set the environment variable first as e.g. `CUDA_VISIBLE_DEVICES=2`.

Summary statistics will be written to the `results` directory. 
