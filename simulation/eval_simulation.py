from os.path import join
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm

from util import LATENT_DIR, OUT_DIR, get_latent_pkl, get_out

def eval_simulation(
        gan_name='stylegan2_healthy',
        exp_vars=[0.5],
        n_causal=1250,
        mult_scale=2.,
        model='resnet50',
        layer='L4',
        size=448,
        tfms='tta',
        spatial='mean',
        sample_sizes=[6000],
        seeds=[123],
        pcs=range(10),
        thres=None,
        wdir='.',
        ):
    results = []
    for exp_var in exp_vars:
        for sample_size in sample_sizes:
            powers = []
            t1ers = []
            t1es = []
            for seed in tqdm(seeds):
                power, t1er, t1e, _, _ = eval_single(
                        gan_name=gan_name,
                        exp_var=exp_var,
                        n_causal=n_causal,
                        mult_scale=mult_scale,
                        model=model,
                        layer=layer,
                        size=size,
                        tfms=tfms,
                        spatial=spatial,
                        sample_size=sample_size,
                        seed=seed,
                        pcs=pcs,
                        thres=thres,
                        wdir=wdir,
                        )
                powers.append(power)
                t1ers.append(t1er)
                t1es.append(t1e)
            mean_power = np.mean(powers)
            sem_power = np.std(powers) / np.sqrt(len(powers))

            mean_t1er = np.mean(t1ers)
            sem_t1er = np.std(t1ers) / np.sqrt(len(t1ers))

            mean_t1e = np.mean(t1es)
            sem_t1e = np.std(t1es) / np.sqrt(len(t1es))

            results.append([exp_var, sample_size, mean_power, sem_power, mean_t1er, sem_t1er, mean_t1e, sem_t1e])
    cols = [
            'Explained Variance of Latent Code',
            'Sample Size',
            'Empirical Power',
            'SEM (Power)',
            'Empirical Type-1 Error Rate',
            'SEM (T1ER)',
            'Mean Number of Type-1 Errors',
            'SEM (T1E)',
            ]
    results = pd.DataFrame(results, columns=cols)
    return results


def eval_single(
        gan_name='stylegan2_healthy',
        exp_var=0.5,
        n_causal=1250,
        mult_scale=2.,
        model='resnet50',
        layer='L4',
        size=448,
        tfms='tta',
        spatial='mean',
        sample_size=6000,
        seed=123,
        pcs=range(10),
        thres=None,
        wdir='.',
        ):
    latent_fn = join(wdir, LATENT_DIR, get_latent_pkl(exp_var, n_causal, seed))
    print(latent_fn)
    P, eff, null, causal, W, R = pickle.load(open(latent_fn, 'rb'))
    if isinstance(R, list):
        R = np.concatenate(R)
    effr = R[eff]
    nullr = R[null]
    causalr = R[causal]

    fn = join(wdir, OUT_DIR, get_out(
            gan_name,
            exp_var,
            n_causal,
            mult_scale,
            model.split('.')[0],
            layer,
            spatial,
            size,
            tfms,
            seed,
            sample_size,
        ))
 
    dfs = []
    for pc in pcs:
        try:
            bolt_out = pd.read_csv(fn%pc, sep='\t')
            bolt_out.P_BOLT_LMM_INF = bolt_out.P_BOLT_LMM_INF.astype(float)
            dfs.append(bolt_out)
        except:
            bolt_out_dummy = dfs[0].copy()
            bolt_out_dummy.P_BOLT_LMM_INF = 1.
            dfs.append(bolt_out_dummy)
    df = dfs[0]
    for pc in pcs:
        df[f'P_BOLT_PC_{pc}'] = dfs[pc].P_BOLT_LMM_INF
    df.index = df.SNP
    df['ind'] = np.arange(len(df))
    null = np.intersect1d(df.SNP, nullr)
    causal = np.intersect1d(df.SNP, causalr)

    if thres is None:
        thres = 0.05 / len(df)

    pvs = np.array([d.P_BOLT_LMM_INF.values for d in dfs])
    df['BONFERRONI_MIN'] = bonferroni_min(pvs.T)

    power = (df.loc[causal].BONFERRONI_MIN<thres).mean()
    t1er = (df.loc[null].BONFERRONI_MIN<thres).mean()
    t1e = (df.loc[null].BONFERRONI_MIN<thres).sum()

    columns = ['SNP', 'CHR', 'BP', 'GENPOS', 'ALLELE1', 'ALLELE0', 'A1FREQ', 'F_MISS', 'BONFERRONI_MIN'] + [f'P_BOLT_PC_{pc}' for pc in pcs]
    df = df[columns]
    fn = join(wdir,
            OUT_DIR,
            'aggregate_' + get_out(
            gan_name,
            exp_var,
            n_causal,
            mult_scale,
            model.split('.')[0],
            layer,
            spatial,
            size,
            tfms,
            seed,
            sample_size,
        ) % '')
    df.to_csv(fn, index=False)

    return power, t1er, t1e, (dfs, null, causal), df

def bonferroni_min(pvs):
    return (pvs.shape[1] * pvs.min(1)).clip(max=1.)
