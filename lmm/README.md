# transferGWAS analysis with BOLT-LMM

This is a simple wrapper for basic preprocessing and running BOLT-LMM. You can ignore it and directly use BOLT-LMM (or any other GWAS software) with the phenotypes condensed in TODO. If you don't specify a path to your BOLT-LMM installations, the script will download it on its own.

## Genetic preprocessing

You first need to preprocess your genetic data. Using `plink2` you can e.g. fill out/modify `preprocessing_ma.sh` for microarray data, and `preprocessing_imp.sh` for imputed data.

In both cases you will need to provide a text file with individuals, having two columns for `FID` and `IID` (but no header!). E.g. to extract the indivduals from the embeddings file (output of `compute_embeddings.py`) you can run (replace `embeddings.txt` by your corresponding filename):
```bash
python -c "import pandas as pd; pd.read_csv('embeddings.txt', sep=' ')[['FID', 'IID']].to_csv('indiv.txt', sep=' ', header=False, index=False)"
```

## Covariates preprocessing

To adjust for covariates in the GWAS, you need to prepare a space-separated file, with first two columns `FID` and `IID`.

For the standard UK Biobank covariate-file, you can use `preprocessing_cov.py`. It automatically extracts `sex` (`31-0.0`), `age` (`21022-0.0`), `assessment_center` (`54-0.0`), and `geno_batch` (`22000-0.0`, binarized in positive and negative), and the `genet_PC_{1:40}` (`22009-0.1 - 22009-0.40`). If you need additional covariates, you can pass key-value pairs (UKB code + name) via `--add_cov` e.g. like in (replace your path to the pheno-file):
```bash
python preprocessing_cov.py path/to/phenotypes.csv \
        --indiv indiv.txt \             # optional: only extract covariates for those individuals
        --output covariates.txt \       # optional: specify where to save
        --add_cov 21000-0.0 ethn \      # optional: more covariates
        --add_cov 20116-0.0 smoke
```

## Running a GWAS

If you want to use the wrapper for BOLT-LMM, we recommend to fill out the `config.toml` file and then run the GWAS via:
```bash
python run_lmm.py --config config.toml
```

If you have already downloaded BOLT-LMM you can specify the path in the `config.toml`, otherwise just leave the line and it will get downloaded automatically.

If you run on the imputed data, BOLT-LMM will likely complain in the first run that there are missing data in the `bgen` files and write a `bolt.in_plink_but_not_imputed.FID_IID.X.txt` (with `X` the number of missing data points). Just add this in the `config.toml` to the `remove` variable. 


Alternatively to the configuration file, you can also leave out the `--config` flag and specify all parameters (if `--config` is specified, all other arguments are ignored); see `python run_lmm.py -h` for details.


