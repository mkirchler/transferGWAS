from os.path import join
import argparse
from subprocess import Popen, PIPE
import toml

import pandas as pd
import numpy as np
from eval_simulation import eval_simulation


from util import EMB_DIR, OUT_DIR, \
        PATH_TO_EXPORT, PATH_TO_GWAS, \
        get_out, get_img_dir, get_emb, \
        get_aggregate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            'config',
            default='config.toml',
            type=str,
            help='Path to configurations',
            )
    parser.add_argument(
            '--stages',
            dest='stages',
            default=1,
            nargs='+',
            type=int,
            help='',
            )
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    args = parser.parse_args()

    configs = toml.load(args.config)
    if isinstance(args.stages, int):
        stages = [args.stages]
    else:
        stages = args.stages
    if isinstance(configs['layers'], str):
        configs['layers'] = [configs['layers']]

    verbose = args.verbose
    if 1 in stages:
        print('running stage 1...')
        simulation_stage_1(**configs, verbose=verbose)
    if 2 in stages:
        print('running stage 2...')
        simulation_stage_2(**configs, verbose=verbose)
    if 3 in stages:
        print('running stage 3...')
        simulation_stage_3(**configs, verbose=verbose)
    if 4 in stages:
        print('running stage 4...')
        simulation_stage_4(**configs, verbose=verbose)
    if 5 in stages:
        print('aggregating results...')
        simulation_stage_5(**configs, verbose=verbose)


def simulation_stage_1(
        bed,
        indiv=None,
        n_causal=1250,
        exp_vars=[0.5],
        normalize=True,
        seeds=[123],
        mid_buffer=2e6,
        verbose=False,
        **kwargs,
        ):
    '''run synthesis of latent codes from genetic data'''
    for seed in seeds:
        for exp_var in exp_vars:
            print(seed, exp_var)
            config = []
            config += [bed]
            config += f'--n_causal {n_causal}'.split()
            config += f'--exp_var {exp_var}'.split()
            config += f'--seed {seed}'.split()
            config += f'--mid_buffer {mid_buffer:.0f}'.split()
            if indiv:
                config += f'--indiv {indiv}'.split()
            if normalize:
                config += [f'--normalize']

            cmd = ['python', '../simulation/synth_latent.py'] + config
            print(cmd)
            if verbose:
                process = Popen(cmd)
            else:
                process = Popen(cmd, stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()

def simulation_stage_2(
        n_causal=1250,
        exp_vars=[0.5],
        stylegan2_models='../models',
        gan_name='stylegan2_healthy',
        mult_scale=2.0,
        diff_noise=False,
        seeds=[123],
        verbose=True,
        psi=0.4,
        **kwargs,
        ):
    '''run synthesis of images from latent codes'''
    for seed in seeds:
        for exp_var in exp_vars:
            config = []
            config += f'--n_causal {n_causal}'.split()
            config += f'--exp_var {exp_var}'.split()
            config += f'--psi {psi:.1f}'.split()
            config += f'--gan_name {gan_name}'.split()
            config += f'--mult_scale {mult_scale:.3f}'.split()
            config += f'--seed {seed}'.split()
            config += f'--models_dir {stylegan2_models}'.split()
            if diff_noise:
                config += [f'--diff_noise']

            cmd = ['python', '../simulation/synth_img.py'] + config
            if verbose:
                process = Popen(cmd)
            else:
                process = Popen(cmd, stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()


def simulation_stage_3(
        n_causal=1250,
        exp_vars=[0.5],
        seeds=[123],
        mult_scale=2.0,
        gan_name='stylegan2_healthy',
        tfms='tta',
        n_pcs=50,
        model='resnet50',
        pretraining='imagenet',
        spatial='mean',
        size=448,
        layers=['L4'],
        verbose=False,
        **kwargs,
        ):
    '''condense synthetic images to low-dimensional embeddings'''
    for seed in seeds:
        for exp_var in exp_vars:
            img_csv = get_img_dir(gan_name, exp_var, n_causal, mult_scale, seed) + '.csv'
            # img_csv = join(
            #         IMG_DIR,
            #         IMG_DIR_TEMPL % (gan_name, exp_var, n_causal, mult_scale, seed),
            #         ) + '.csv'
            emb = get_emb(
                    gan_name,
                    exp_var,
                    n_causal,
                    mult_scale,
                    model.split('.')[0],
                    '%s',
                    spatial,
                    size,
                    tfms,
                    seed,
                    '%s',
                    )
            config = []
            config += f'--save_str {emb}'.split()
            config += f'--img_size {size}'.split()
            config += f'--tfms {tfms}'.split()
            config += f'--n_pcs {n_pcs}'.split()
            config += f'--model {model}'.split()
            config += f'--pretraining {pretraining}'.split()
            #config += f'--layer {layer}'.split()
            config += ['--layer'] + layers

            cmd = ['python', PATH_TO_EXPORT, img_csv, EMB_DIR] + config
            if verbose:
                process = Popen(cmd)
            else:
                process = Popen(cmd, stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()


def simulation_stage_4(
        bed,
        fam,
        cov='../run_lmm/covariates.txt',
        bolt='../run_lmm/BOLT-LMM_v2.3.4/',
        first_pc=0,
        last_pc=9,
        threads=30,
        ref_map='',
        ldscores='',
        cov_columns=['sex', 'assessment_center', 'geno_batch'],
        qcov_columns=['age', 'genet_PC_{1:10}'],
        n_causal=1250,
        exp_vars=[0.5],
        seeds=[123],
        sample_sizes=[25000],
        mult_scale=2.0,
        gan_name='stylegan2_healthy',
        tfms='tta',
        n_pcs=50,
        model='resnet50',
        spatial='mean',
        size=448,
        layers=['L4'],
        verbose=True,
        **kwargs,
        ):
    '''run GWAS on embeddings'''
    for seed in seeds:
        for exp_var in exp_vars:
            for sample_size in sample_sizes:
                for layer in layers:
                    emb = join(
                            EMB_DIR,
                            get_emb(
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
                                '',
                                )
                            )
                    remove_fn_sample = join(
                            EMB_DIR,
                            get_emb(
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
                                f'_ss{sample_size}',
                                ),
                            )

                    out_fn = get_out(
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
                        )
                    embeddings = pd.read_csv(emb, sep=' ').loc[:, 'IID']
                    sample = embeddings.sample(sample_size, random_state=seed)
                    remove_iid = [x for x in embeddings.values if x not in sample.values]
                    remove_iid = pd.DataFrame(
                            np.array([remove_iid, remove_iid]).T,
                            columns=['FID', 'IID'],
                            )
                    remove_iid.to_csv(remove_fn_sample, index=False, sep=' ', header=False)

                    config = []
                    config += f'--bed {bed%"{1:22}"}.bed'.split()
                    config += f'--bim {bed%"{1:22}"}.bim'.split()
                    config += f'--fam {fam}'.split()
                    config += f'--cov {cov}'.split()
                    config += f'--model lmmInfOnly'.split()
                    config += sum([f'--cov_cols {cc}'.split() for cc in cov_columns], [])
                    config += sum([f'--qcov_cols {qc}'.split() for qc in qcov_columns], [])
                    config += f'--emb {emb}'.split()
                    config += f'--first_pc {first_pc}'.split()
                    config += f'--last_pc {last_pc}'.split()
                    config += f'--out_dir {OUT_DIR}'.split()
                    config += f'--out_fn {out_fn}'.split()
                    config += f'--bolt {bolt}'.split()
                    config += f'--threads {threads}'.split()
                    if not ref_map:
                        ref_map = join(bolt, 'tables/genetic_map_hg19_withX.txt.gz')
                    config += f'--ref_map {ref_map}'.split()
                    if not ldscores:
                        ldscores = join(bolt, 'tables/LDSCORE.1000G_EUR.tab.gz')
                    config += f'--ldscores {ldscores}'.split()

                    config += f'--remove {remove_fn_sample}'.split()

                    cmd = ['python', PATH_TO_GWAS] + config
                    if verbose:
                        process = Popen(cmd)
                    else:
                        process = Popen(cmd, stdout=PIPE, stderr=PIPE)
                    stdout, stderr = process.communicate()

def simulation_stage_5(
        n_causal=1250,
        exp_vars=[0.5],
        seeds=[123],
        sample_sizes=[25000],
        mult_scale=2.0,
        gan_name='stylegan2_healthy',
        tfms='tta',
        n_pcs=10,
        model='resnet50',
        spatial='mean',
        size=448,
        layers=['L4'],
        verbose=True,
        **kwargs,
        ):
    for layer in layers:
        fn = get_aggregate_fn(
                gan_name,
                n_causal,
                mult_scale,
                model.split('.')[0],
                layer,
                spatial,
                size,
                tfms,
                )
        df = eval_simulation(
            gan_name=gan_name,
            exp_vars=exp_vars,
            n_causal=n_causal,
            mult_scale=mult_scale,
            model=model,
            layer=layer,
            size=size,
            tfms=tfms,
            spatial=spatial,
            sample_sizes=sample_sizes,
            seeds=seeds,
            pcs=range(n_pcs),
            thres=5e-8,
            )
        df.to_csv(fn+'tresh5e-8.csv', index=False)

if __name__ == '__main__':
    main()
