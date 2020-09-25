# import shutil
import os
from os.path import join
import numpy as np
import pandas as pd
import pickle
# from glob import glob
# import sys
import argparse

# from scipy.stats import pearsonr
# from matplotlib import pyplot as plt

from pyplink import PyPlink
from tqdm import tqdm
# from subprocess import Popen, PIPE
# import toml

# BASE_PRJ = '/mnt/projects/ukbiobank/derived/projects/deep_test/'
# BASE_MA = join(BASE_PRJ, 'genetics', 'microarray')
# GEN_TEMPL = join(BASE_MA, 'ukb_chr%d.caucasian_maf-0.001_ld-0.8')
# BASE_PHENO = join(BASE_PRJ, 'phenotypes')
# INDIV = join(BASE_PHENO, 'test_pop.txt')
# EXCL1 = join(BASE_PHENO, 'bolt.in_plink_but_not_imputed.FID_IID.134.txt')
# EXCL2 = join(BASE_PHENO, 'remove_IID.txt')
# LATENT_DIR = 'latent'
# LATENT_BOLT_TEMPL = 'exp_var%.3f_nc%d_seed%d.txt'
# LATENT_PKL_TEMPL = 'exp_var%.3f_nc%d_seed%d.pkl'


# CONFIG = toml.load('config.toml')

# INDIV = CONFIG['indiv']
# GEN_TEMPL = CONFIG['bed']
# LATENT_DIR = CONFIG['latent_dir']
# LATENT_BOLT_TEMPL = CONFIG['latent_bolt_templ']
# LATENT_PKL_TEMPL = CONFIG['latent_pkl_templ']

from util import LATENT_DIR, get_latent_pkl, get_latent_bolt
# LATENT_DIR = 'latent'
# LATENT_BOLT_TEMPL = 'exp_var%.3f_nc%d_seed%d.txt'
# LATENT_PKL_TEMPL = 'exp_var%.3f_nc%d_seed%d.pkl'
os.makedirs(LATENT_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('geno_temp', type=str, help='path to bed files, without file-extensions')
    parser.add_argument('--n_causal', dest='n_causal', default=100, type=int)
    parser.add_argument('--exp_var', dest='exp_var', default=0.5, type=float)
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--seed', dest='seed', default=123, type=int)
    parser.add_argument('--indiv', dest='indiv', default='indiv.txt', type=str)
    # parser.add_argument('--param_select', dest='param_select', action='store_true')
    # parser.add_argument('--debug', dest='debug', action='store_true')

    args = parser.parse_args()
    # if args.debug and args.param_select:
    #     raise ValueError("can't have both debug and param_select mode")
    # if args.debug:
    #     chromos = range(21, 23)
    # elif args.param_select:
    #     chromos = range(13, 23)
    # else:
    #     chromos = range(1, 23)

    # TODO change back
    chromos = range(21, 23)
    # config = dict(
    #         chromos=chromos,
    #         normalize=args.normalize,
    #         n_causal=args.n_causal,
    #         exp_var=args.exp_var,
    #         seed=args.seed,
    #         )

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
            seed=args.seed,
            )


def get_indiv(path):
    '''load individuals'''
    indiv = pd.read_csv(path, sep=' ', header=None)[0].values
    return indiv

    # indiv = pd.read_csv(INDIV, sep=' ', header=None)[0].values
    # excl1 = pd.read_csv(EXCL1, sep=' ', header=None)[0].values
    # excl2 = pd.read_csv(EXCL2, sep=' ').iloc[0].values
    # excl = set(np.union1d(excl1, excl2))
    # indiv = np.array(sorted([x for x in indiv if not x in excl]))
    # return indiv


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
    bed = PyPlink(geno_temp % chromo)
    fam = bed.get_fam()
    if indiv is None:
        indiv = fam.iid.astype(int)
        # indiv = get_indiv()
    # elif indiv == False:

    ind = np.isin(fam.iid.astype(int), indiv)
    indiv = fam.loc[ind, 'iid'].astype(int).values

    G = []
    rsids = []
    for g in tqdm(bed.iter_geno(), total=bed.get_nb_markers()):
        G.append(g[1][ind])
        rsids.append(g[0])

    G = pd.DataFrame(
            np.array(G).T,
            index=indiv,
            columns=['c%d:%d'%(chromo, x) for x in range(bed.get_nb_markers())],
            )

    bim = bed.get_bim()
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
        seed=123,
        ):
    '''create simulated phenotype

    Build simulated latent code for input chromosomes. Save results to default pickle and space-separated value file for use in other parts of the pipeline to corresponding `latent` directory

    # Parameters
    chromos (list): list of ints, all chromos to include
    normalize (bool): normalize genetic data before computing phenotype
    n_pheno (int): dimensionality of output phenotype
    n_causal (int): number of causal SNPs
    exp_var (float in [0, 1]): percentage of explained variance by causal SNPs
    seed (int or None): random seed
    '''
    Gs, effs, nulls, rsids = [], [], [], []
    ind = 0
    for chromo in chromos:
        G, eff, null, rsid = load_bed(geno_temp, chromo=chromo, indiv=indiv)
        Gs.append(G)
        effs.append(eff+ind)
        nulls.append(null+ind)
        rsids.append(rsid)
        ind += G.shape[1]

    G = pd.concat(Gs, axis=1)
    eff = np.concatenate(effs)
    null = np.concatenate(nulls)
    rsid = np.concatenate(rsids)
    pheno, causal, W, ge, ne = simulate_pheno(
            G,
            eff,
            n_pheno=n_pheno,
            n_causal=n_causal,
            normalize=normalize,
            rat_explained_variance=exp_var,
            seed=seed,
            )

    s_pkl = join(LATENT_DIR, get_latent_pkl(exp_var, n_causal, seed))
    s_bolt = join(LATENT_DIR, get_latent_bolt(exp_var, n_causal, seed))
    pickle.dump([pheno, eff, null, causal, W, rsids], open(s_pkl, 'wb'))
    pheno['FID'] = pheno['IID'] = pheno.index
    pheno = pheno[['FID', 'IID']+list(range(512))]
    pheno.columns = list(pheno.columns[:2]) + ['pheno_%d'%d for d in range(512)]
    pheno.to_csv(s_bolt, sep=' ', index=False)


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




## util ## 

# def mafs(G):
#     ac1 = 2*(G==0).sum(0) + (G==1).sum(0)
#     ac2 = 2*(G==2).sum(0) + (G==1).sum(0)
#     ac = ac1 + ac2
#     maf = np.minimum(ac1 / ac, ac2 / ac)
#     return maf

# def aggregate(pcs=range(10)):
#     dfs = [pd.read_csv('out_%d.qassoc'%d, delim_whitespace=True) for d in pcs]
#     df = dfs[0]
#     df['p_pc%d'%pcs[0]] = df.P
#     for i in pcs[1:]:
#         df['p_pc%d'%i] = dfs[i].P
#     df['minp'] = df[['p_pc%d'%d for d in pcs]].min(1)
#     return df




#### TODO delete ? 

## plotting ##

#def plot_bolt_output(fn='tmp_dump', dim=0, up_to_chr=False, thres=None):
#    bolt_out = pd.read_csv(join('output', fn+'bolt_pheno_%d.caucasian_maf-0.001_ld-0.8.txt'%dim), sep='\t')
#    bolt_out.P_BOLT_LMM_INF = bolt_out.P_BOLT_LMM_INF.astype(float)
#    if True: #with_rsid:
#        P, eff, null, causal, W, R = pickle.load(open(join('data', fn+'.pkl'), 'rb'))
#        if isinstance(R, list):
#            R = np.concatenate(R)
#        effr = R[eff]
#        nullr = R[null]
#        causalr = R[causal]
#        if True:
#            bolt_out.index = bolt_out.SNP
#            bolt_out['ind'] = np.arange(len(bolt_out))
#            eff = np.intersect1d(bolt_out.SNP, effr)
#            null = np.intersect1d(bolt_out.SNP, nullr)
#            causal = np.intersect1d(bolt_out.SNP, causalr)
#            #eff = bolt_out.loc[np.intersect1d(bolt_out.SNP.values, effr)] 
#        else:
#            eff = np.where(np.isin(bolt_out.SNP.values, effr, assume_unique=True))
#            null = np.where(np.isin(bolt_out.SNP.values, nullr, assume_unique=True))
#            causal = np.where(np.isin(bolt_out.SNP.values, causalr, assume_unique=True))
#        if thres is None:
#            thres = 0.05 / len(bolt_out)
#        power = (bolt_out.loc[causal].P_BOLT_LMM_INF<thres).mean()
#        print(f'{power*100:.2f}% of causal SNPs are below {thres:.2E}')
#        wandb.log({'power': power}, commit=False)
#    else:
#        P, eff, null, causal, W = pickle.load(open(join('data', f'{fn}_dim{dim}.pkl'), 'rb'))
#    if up_to_chr:
#        bolt_out = bolt_out.loc[bolt_out.CHR<=up_to_chr]
#    #pvs = bolt_out.P_BOLT_LMM_INF.values
#    plot_pvs(bolt_out, eff, null, causal, fn, dim=dim)#f'{fn}_dim{dim}')


#def plot_pvs(df, eff, null, causal, fn, dim=None):
#    plt.figure(figsize=(40, 20))
#    #pvs[pvs<1e-320] = 1e-322
#    df.loc[df.P_BOLT_LMM_INF<1e-320, 'P_BOLT_LMM_INF'] = 1e-322
#    if True:
#        plt.scatter(df.ind, -np.log10(df.P_BOLT_LMM_INF), marker='.', color='gray', label='buffer')
#        plt.scatter(df.loc[eff, 'ind'], -np.log10(df.loc[eff, 'P_BOLT_LMM_INF']), marker='.', label='causal+ld')
#        plt.scatter(df.loc[causal, 'ind'], -np.log10(df.loc[causal, 'P_BOLT_LMM_INF']), marker='o', edgecolor='red', color='none', label='causal snp')
#        plt.scatter(df.loc[null, 'ind'], -np.log10(df.loc[null, 'P_BOLT_LMM_INF']), marker='.', label='non-causal')
#    else:
#        plt.scatter(np.arange(len(pvs)), -np.log10(pvs), marker='.', color='gray', label='buffer')
#        plt.scatter(np.arange(len(pvs))[eff], -np.log10(pvs[eff]), marker='.', label='causal+ld')
#        plt.scatter(np.arange(len(pvs))[causal], -np.log10(pvs[causal]), marker='o', edgecolor='red', color='none', label='causal snp')
#        plt.scatter(np.arange(len(pvs))[null], -np.log10(pvs[null]), marker='.', label='non-causal')
#    plt.hlines(-np.log10(0.05/len(df)), xmin=0, xmax=len(df))
#    plt.hlines(-np.log10(5e-8), xmin=0, xmax=len(df))
#    plt.legend()
#    s = join('figures', f'{fn}_mhat_dim{dim}.png')# fn+'mhat.png')
#    plt.savefig(s)
#    wandb.log({f'manhattan dim {dim}': plt}, commit=False)
#    wandb.log({f'manhattan_png dim {dim}': wandb.Image(s)}, commit=False)

#    qq(df.P_BOLT_LMM_INF.values, f'full_dim_{dim}', fn)
#    qq(df.loc[eff, 'P_BOLT_LMM_INF'].values, f'causal+ld_dim {dim}', fn)
#    qq(df.loc[null, 'P_BOLT_LMM_INF'].values, f'non-causal_dim_{dim}', fn)


#def qq(pvs, title=None, fn=None):
#    plt.figure(figsize=(10, 10))
#    p = np.sort(pvs)
#    x = np.arange(1, len(p)+1)/len(p)
#    plt.scatter(-np.log10(x), -np.log10(p))
#    plt.plot(-np.log10(x), -np.log10(x))
#    plt.title(title)
#    if not fn is None:
#        s = join('figures', f'{fn}_{title}.png')
#        plt.savefig(s)
#        wandb.log({f'{title}_png': wandb.Image(s)}, commit=False)
#    wandb.log({title: plt}, commit=False)



#def create_gen(n_subs=1000, n_snps=10000, n_causal=5, n_pheno=512, effect_size=1., scale=1., sex=False):
#    G = np.random.randint(0, 3, (n_subs, n_snps))
#    G = pd.DataFrame(G, index=[f'sam_{i}' for i in range(len(G))], columns=[f'rs{i}' for i in range(G.shape[1])])
#    W = np.zeros((n_pheno, n_snps))
#    causal = np.random.choice(n_snps, n_causal, replace=False)
#    for p in range(n_pheno):
#        beta = effect_size*2*(np.random.rand(n_causal)-0.5)
#        W[p, causal] = beta
#    phenos = G @ W.T + np.random.normal(scale=scale, size=(n_subs, n_pheno))
#    if sex:
#        sex = pd.DataFrame(np.random.randint(0, 2, n_subs), index=[f'sam_{i}' for i in range(n_subs)])
#        w_sex = 0.1*np.random.randn(n_pheno, 1)
#        phenos += sex @ w_sex.T
#    else:
#        sex = None
#        w_sex = None
#    return phenos, G, sex, W, w_sex


#def create_synth_data(T, out_bed_fn='gen', out_img_dir='img_data', n_subs=100, n_snps=1000, n_causal=5, effect_size=1., rnd_std=1., same_noise=True, psi=None, rmdir=True, sex=False):
#    P, G, S, W, w_sex = create_gen(n_subs, n_snps, n_causal, n_pheno=512, effect_size=1., scale=rnd_std, sex=sex)
#    write_to_plink(G, out_bed_fn, sex=S)

#def write_to_plink(G, fn, sex=None, pheno=None):
#    with PyPlink(fn, "w") as pedfile:
#        for genotypes in G.values.T:
#            pedfile.write_genotypes(genotypes)

#    with open(f"{fn}.fam", "w") as fam_file:
#        for i in G.index:
#            s = 0 if sex is None else sex.loc[i].values[0]+1
#            p = -9 if pheno is None else pheno.loc[i]
#            print(f'{i} {i} 0 0 {s} {p}', file=fam_file)

#    with open(f"{fn}.bim", "w") as bim_file:
#        for i, rs in enumerate(G.columns):
#            print(f'1\t{rs}\t0\t{i+1}\tA\tT', file=bim_file)


#def get_linear_correlations(G, p):#eff_snps, null_snps, causal
#    pvs = []
#    ccs = []
#    for _, g in tqdm(G.iteritems(), total=G.shape[1]):
#        cc, pv = pearsonr(g, p)
#        ccs.append(cc)
#        pvs.append(pv)
#    return np.array(ccs), np.array(pvs)

#def plot_linear_correlations(G, p, causal, eff, null):
#    ccs, pvs = get_linear_correlations(G, p)
#    plot_pvs(pvs, eff, null, causal)

#def baseline_multidim_experiment(
#        chromos=[21, 22],
#        normalize=True,
#        n_causal=100,
#        exp_var=0.5,
#        seed=123,
#        dims=range(10),
#        threads=60,
#        skip_synth=False,
#        skip_bolt=False,
#        ):
#    direc = LATENT_DIR
#    save_str = f'exp_var{exp_var:.3f}_nc{n_causal}'
#    if not skip_synth:
#        simulate_full_data(
#                chromos=chromos,
#                normalize=normalize,
#                n_causal=n_causal,
#                exp_var=exp_var,
#                seed=seed,
#                #save=join(direc, save_str),
#                )
#    if not skip_bolt:
#        for dim in dims:
#            cmd = ['bash', 'run_bolt.sh', save_str+'bolt', str(dim), str(threads), str(chromos[0]), str(chromos[-1])]

#            process = Popen(cmd, stdout=PIPE, stderr=PIPE)
#            stdout, stderr = process.communicate()

#            plot_bolt_output(save_str, dim=dim)
#            wandb.log()
#    else:
#        for dim in dims:
#            plot_bolt_output(save_str, dim=dim)
#            wandb.log()


#def baseline_experiment(
#        chromos=[21, 22],
#        normalize=True,
#        n_causal=100,
#        exp_var=0.5,
#        seed=123,
#        dim=0,
#        threads=60,
#        skip_synth=False,
#        skip_bolt=False,
#        ):
#    direc = 'data'
#    save_str = f'exp_var{exp_var:.3f}_nc{n_causal}'
#    if not skip_synth:
#        simulate_full_data(
#                chromos=chromos,
#                normalize=normalize,
#                n_causal=n_causal,
#                exp_var=exp_var,
#                seed=seed,
#                #save=join(direc, save_str),
#                )
#    if not skip_bolt:
#        cmd = ['bash', 'run_bolt.sh', save_str+'bolt', str(dim), str(threads), str(chromos[0]), str(chromos[-1])]

#        process = Popen(cmd, stdout=PIPE, stderr=PIPE)
#        stdout, stderr = process.communicate()
        
#    plot_bolt_output(save_str, dim=dim)


