import os
from os.path import join
import pickle
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm

from pyplink import PyPlink

from util import LATENT_DIR, get_latent_pkl, get_latent_bolt



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('geno_temp', type=str, help='path to bed files, without file-extensions')
    parser.add_argument('--n_causal', dest='n_causal', default=100, type=int)
    parser.add_argument('--exp_var', dest='exp_var', default=0.5, type=float)
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--seed', dest='seed', default=123, type=int)
    parser.add_argument('--mid_buffer', dest='mid_buffer', default=int(2e6), type=int)
    parser.add_argument('--indiv', dest='indiv', default='indiv.txt', type=str)
    parser.add_argument('--wdir', dest='wdir', default='.', type=str)

    args = parser.parse_args()
    chromos = range(1, 23)

    if args.indiv:
        indiv = get_indiv(args.indiv)
    else:
        indiv = False

    
    print('starting simulation')
    simulate_full_data(
            geno_temp=args.geno_temp,
            indiv=indiv,
            chromos=chromos,
            normalize=args.normalize,
            n_causal=args.n_causal,
            exp_var=args.exp_var,
            mid_buffer=args.mid_buffer,
            seed=args.seed,
            wdir=args.wdir,
            )


def get_indiv(path):
    '''load individuals'''
    indiv = pd.read_csv(path, sep=' ', header=None)[0].values
    return indiv


def load_bed(geno_temp, chromo=1, indiv=None, mid_buffer=2e6):
    '''load bed-file and split in effect and null SNPs
    
    # Parameters:
    chromo (int): chromosome number
    indiv (None or np.ndarray):
                        if None, use all individuals in bed-file
                        if np.array, use elements as indivduals
    mid_buffer (int): number of bp up- and downstream of center of chromosome to leave out

    # Returns:
    G (pd.DataFrame): DataFrame with one indiv per row, one SNP per column
    eff_snps, null_snps (np.ndarray): array of ints with position of effect- and null-SNPs
    rsid (np.ndarray) array of str with rsid names of SNPs in G
    '''
    print(chromo)
    bed = PyPlink(geno_temp % chromo)
    print(bed.get_nb_markers())
    fam = bed.get_fam()
    if indiv is None:
        indiv = fam.iid.astype(int)

    ind = np.isin(fam.iid.astype(int), indiv)
    indiv = fam.loc[ind, 'iid'].astype(int).values

    G = []
    rsids = []
    removed = 0
    for g in tqdm(bed.iter_geno(), total=bed.get_nb_markers()):
        rs = g[0]
        gen = g[1][ind]
        g_ind = gen == -1
        if g_ind.mean() < 0.1:
            gen[g_ind] = gen[~g_ind].mean()
            G.append(gen)
            rsids.append(rs)
        else:
            removed += 1
    print(f'removed {removed} SNPs due to missing>10%')

    G = pd.DataFrame(
            np.array(G).T,
            index=indiv,
            columns=['c%d:%d'%(chromo, x) for x in range(len(rsids))],
            )

    bim = bed.get_bim().loc[rsids]
    mid = bim.pos.min() + (bim.pos.max()  - bim.pos.min()) // 2
    eff_snps = np.where(bim.pos < mid - mid_buffer)[0]
    null_snps = np.where(bim.pos > mid + mid_buffer)[0]

    return G, eff_snps, null_snps, np.array(rsids)


def simulate_full_data(
        geno_temp,
        indiv=None,
        chromos=range(1, 23),
        normalize=False,
        n_pheno=512,
        n_causal=100,
        exp_var=0.5,
        mid_buffer=2e6,
        seed=123,
        wdir='.',
        ):
    '''create simulated phenotype

    Build simulated latent code for input chromosomes. Save results to default pickle and space-separated value file for use in other parts of the pipeline to corresponding `latent` directory

    # Parameters
    chromos (list): list of ints, all chromos to include
    normalize (bool): normalize genetic data before computing phenotype
    n_pheno (int): dimensionality of output phenotype
    n_causal (int): number of causal SNPs
    exp_var (float in [0, 1]): percentage of explained variance by causal SNPs
    mid_buffer (int): how much space to keep before and after midpoint of chromosomes to effect/null snps
    seed (int or None): random seed
    '''
    latent_dir = join(wdir, LATENT_DIR)
    os.makedirs(latent_dir, exist_ok=True)

    Gs, effs, nulls, rsids = [], [], [], []
    ind = 0
    print('loading genotype data...')
    for chromo in chromos:
        G, eff, null, rsid = load_bed(geno_temp, chromo=chromo, indiv=indiv, mid_buffer=mid_buffer)
        Gs.append(G)
        effs.append(eff+ind)
        nulls.append(null+ind)
        rsids.append(rsid)
        ind += G.shape[1]

    G = pd.concat(Gs, axis=1)
    eff = np.concatenate(effs)
    null = np.concatenate(nulls)
    rsid = np.concatenate(rsids)
    print('simulating latent code...')
    pheno, causal, W, ge, ne = simulate_pheno(
            G,
            eff,
            n_pheno=n_pheno,
            n_causal=n_causal,
            normalize=normalize,
            rat_explained_variance=exp_var,
            seed=seed,
            )

    s_pkl = join(latent_dir, get_latent_pkl(exp_var, n_causal, seed))
    s_bolt = join(latent_dir, get_latent_bolt(exp_var, n_causal, seed))
    pickle.dump([pheno, eff, null, causal, W, rsids], open(s_pkl, 'wb'))
    pheno['FID'] = pheno['IID'] = pheno.index
    pheno = pheno[['FID', 'IID']+list(range(512))]
    pheno.columns = list(pheno.columns[:2]) + ['pheno_%d'%d for d in range(512)]
    pheno.to_csv(s_bolt, sep=' ', index=False)

    s_causal = join(wdir, 'causal_variants.csv')
    causal_df = pd.DataFrame(np.concatenate(rsids)[causal])
    causal_df.to_csv(s_causal, index=False, header=False)


def simulate_pheno(
        G,
        eff_snps,
        n_pheno=512,
        n_causal=100,
        rat_explained_variance=0.5,
        normalize=True,
        seed=123,
        ):
    '''build latent code from genetic data

    Simulate a n_pheno-dimensional phenotype from input data and noise in specified ratio. Effect sizes are drawn from standard normal distribution and then normalized. 

    # Parameters:
    G (pd.DataFrame): DataFrame of genetic data
    eff_snps (np.ndarray): array of ints of SNPs in G (columns) that act as potential causal SNPs
    n_pheno (int): dimensionality of output phenotype
    n_causal (int): number of causal SNPs
    rat_explained_variance (float in [0, 1]): percentage of explained variance by causal SNPs
    normalize (bool): normalize G
    seed (int or None): random seed

    # Returns:
    pheno (np.ndarray): array of phenotypes
    causal (np.ndarray): subset of eff_snps that act as causal SNPs
    W (np.ndarray): Weight matrix for causal SNPs
    gen_effect (np.ndarray): genetic signal in pheno
    noise (np.ndarray): noise signal in pheno
    '''
    n_samples = len(G)
    if seed is not None:
        np.random.seed(seed)
    W = np.random.randn(n_causal, n_pheno)
    causal = np.random.choice(eff_snps, n_causal, replace=False)
    G_causal = G.iloc[:, causal]
    if normalize:
        G_causal = (G_causal - G_causal.mean()) / G_causal.std()
    gen_effect = G_causal @ W

    gen_effect = (gen_effect - gen_effect.mean()) / gen_effect.std()
    noise = np.random.randn(n_samples, n_pheno)
    pheno = rat_explained_variance**0.5 * gen_effect + (1-rat_explained_variance)**0.5 * noise
    return pheno, causal, W, gen_effect, noise

if __name__ == '__main__':
    main()
