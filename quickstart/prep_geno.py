import os
from os.path import join
import uuid
import argparse

import pandas as pd
import numpy as np

import toml
from tqdm import tqdm

from pyplink import PyPlink
BOLT_DIR = '../BOLT-LMM_v2.3.4'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--sample_size',
            default=5000,
            type=int,
            help='Sample size of synthetic data',
            )
    parser.add_argument(
            '--n_geno',
            default=100000,
            type=int,
            help='Number of SNPs to create',
            )
    parser.add_argument(
            '--n_causal',
            default=100,
            type=int,
            help='Number of causal SNPs',
            )
    parser.add_argument(
            '--exp_var',
            default=0.5,
            type=float,
            help='explained variance of the causal SNPs',
            )
    parser.add_argument(
            '--seed',
            default=271828,
            type=int,
            help='Random number generator seed',
            )
    parser.add_argument(
            '--threads',
            default=8,
            type=int,
            help='Number of threads to use',
            )
    args = parser.parse_args()

    prepare_simulation(args.sample_size, args.n_geno, args.seed, args.exp_var, args.n_causal, args.threads)


def prepare_simulation(sample_size, n_geno=1000, seed=321, exp_var=0.5, n_causal=10, threads=8):
    run_id = uuid.uuid4()
    run_dir = join('runs', f'run_{run_id}')
    os.makedirs(run_dir, exist_ok=True)
    print(f'creating dummy data in {run_dir}...')

    # create indiv file
    indiv_path = join(run_dir, 'indiv.txt')
    start = 10000000
    iid = pd.DataFrame([[i, i] for i in range(start, start+sample_size)], columns=['FID', 'IID'])
    iid.to_csv(indiv_path, sep=' ', header=False, index=False)

    # create covariate file
    cov_path = join(run_dir, 'covariates.txt')
    age = np.random.randint(30, 81, sample_size)
    sex = np.random.randint(0, 2, sample_size)
    iid['age'] = age
    iid['sex'] = sex
    iid.to_csv(cov_path, sep=' ', header=True, index=False)

    # create toml file
    config_path = join(run_dir, 'config.toml')
    config = {
            # general parameters
            'seeds': [seed],
            'exp_vars': [exp_var],
            'n_causal': n_causal,
            'sample_sizes': [sample_size],
            'wdir': run_dir,

            # genetic parameters
            'bed': join(run_dir, 'chr%s'),
            'fam': join(run_dir, 'chr1.fam'),
            'normalize': True,
            'indiv': indiv_path,
            'mid_buffer': 0,

            # GAN parameters
            'stylegan2_models': '../models/',
            'stylegan2_name': 'stylegan2_healthy',
            'psi': 0.4,
            'diff_noise': False,
            'mult_scale': 2.0,

            # feature condensation parameters
            'n_pcs': 100,
            'size': 448,
            'tfms': 'tta',
            'model': 'resnet50',
            'pretraining': 'imagenet',
            'spatial': 'mean',
            'layers': ['L4'],

            # GWAS parameters
            'first_pc': 0,
            'last_pc': 9,
            'cov': cov_path,
            'cov_columns': ['sex'],
            'qcov_columns': ['age'],
            'threads': threads,
            'bolt': '../BOLT-LMM_v2.3.4/',
            'ref_map': "",
            'ldscores': "",
            }
    toml.dump(config, open(config_path, 'w'))

    # create genetic data
    iid['father'] = 0
    iid['mother'] = 0
    iid['sex'] += 1
    iid['pheno'] = -9
    fam_df = iid[['FID', 'IID', 'father', 'mother', 'sex', 'pheno']]

    ld_df = pd.read_csv(join(BOLT_DIR, 'tables', 'LDSCORE.1000G_EUR.tab.gz'), sep='\t')
    ld_df = ld_df.sample(n_geno).sort_values(['CHR', 'BP'])
    G = (np.random.rand(n_geno, sample_size, 2) > ld_df.MAF.values.reshape(-1, 1, 1)).sum(-1)

    last_ind = 0
    for chromo, chromo_df in tqdm(ld_df.groupby('CHR')):
        gen_path = join(run_dir, 'chr%d'%chromo)

        n_chromo = len(chromo_df)
        chromo_G = G[last_ind:(last_ind+n_chromo)]
        last_ind += n_chromo

        with PyPlink(gen_path, 'w') as bed:
            for g in chromo_G:
                bed.write_genotypes(g)

        fam_df.to_csv(gen_path+'.fam', sep='\t', header=False, index=False)

        bim_df = pd.DataFrame(np.array([
            n_chromo * [chromo],
            chromo_df.SNP.values,
            n_chromo * [0],
            chromo_df.BP.values,
            n_chromo * ['C'],
            n_chromo * ['A'],
            ]).T)
        bim_df.to_csv(gen_path+'.bim', sep='\t', header=False, index=False)

if __name__ == '__main__':
    main()
