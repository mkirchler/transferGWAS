from os.path import join
import argparse
from subprocess import Popen, PIPE
from tqdm import tqdm
import toml
import time

import pandas as pd

#DEBUG = False
#N_PCS = 100

## SEEDS
##SEEDS = [234, 513, 786, 936, 278, 896, 983, 367, 961, 241]

## seeds just for 
#SEEDS = [3156, 9432, 6458, 5308, 6748, 5479, 5487, 3762, 5748, 5265]

## PARAM
#N_CAUSALS = [1250, 2500, 5000, 10000]
#EXP_VARS = [0.3, 0.5, 0.7]
##EXP_VARS = [0.25, 0.5, 0.75]
##EXP_VARS = [0.1, 0.25, 0.5, 0.75, 0.9]
##EXP_VARS = [0.1, 0.3, 0.5, 0.7, 0.9]
#SAMPLE_SIZES = [6000, 12000, 24000, 46731]
##SAMPLE_SIZES = [5841, 11683, 23366, None]


## check this for one or two constant values of
#SIZES = [224, 448]
##SIZES = [448]
#TFMS = ['tta', 'basic']
##TFMS = ['tta']
#MODELS = ['res50IN']
## res18
##LAYERS = ['layer2.1.conv2 layer3.1.conv2 layer4.1.conv2 layer4.1.bn2']
## res50
#LAYERS = ['layer2.3.conv3 layer3.5.conv3 layer4.2.conv3 layer4.2.bn3']

##LAYERS = ['layer1.0.conv3 layer1.1.conv3 layer1.2.conv3 layer2.0.conv3 layer2.1.conv3 layer2.2.conv3 layer2.3.conv3']
#SPATIAL = ['mean', 'max']
##SPATIAL = ['mean']

## CONST
#NORMALIZE = True
#PSI = 0.4
#DIFF_NOISE = False
#THREADS = 18
##SEED = 123
#GAN_NAME = 'healthy'
#MULT_SCALE = 2.
#PLOTTING = False


#POWER_PARAMS = {
#        'ss': {'sample_size': [6000, 12000, 24000, 46731], 'n_causal': 1250, 'exp_var': 0.5},
#        #'nc': {'n_causal': [100, 500, 1250, 2500, 5000, 10000], 'sample_size': 46731, 'exp_var': 0.5},
#        'ev': {'exp_var': [0.7], 'sample_size': 46731, 'n_causal': 1250},
#        }
#PP = [(x, POWER_PARAMS['ss']['n_causal'], POWER_PARAMS['ss']['exp_var']) for x in POWER_PARAMS['ss']['sample_size']]
##PP += [(POWER_PARAMS['nc']['sample_size'], x, POWER_PARAMS['nc']['exp_var']) for x in POWER_PARAMS['nc']['n_causal']]
##PP += [(POWER_PARAMS['ev']['sample_size'], POWER_PARAMS['ev']['n_causal'], x) for x in POWER_PARAMS['ev']['exp_var']]
#PP = list(set(PP))
IMG_DIR = 'images'
IMG_DIR_TEMPL = 'exp_%s_var%.3f_nc%d_sc%.2f_seed%d'
EMB_DIR = 'embeddings'
EMB_TEMPL = 'exp_%s_var%.3f_nc%d_sc%.2f_%s_%s_%s_s%d_%s_seed%d%s.txt'
OUT_TEMPL = 'exp_%s_var%.3f_nc%d_sc%.2f_%s_%s_%s_s%d_%s_seed%d_ss%d_pheno_%s'

PATH_TO_EXPORT = '../export_embeddings/export_embeddings.py'
PATH_TO_GWAS = '../run_bolt/run_gwas.py'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default='config.toml', type=str, help='Path to configurations')
    parser.add_argument(
            '--stages',
            dest='stages',
            default=1,
            nargs='+',
            type=int,
            help='',
            )
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    # parser.add_argument('--seed', dest='seed', default=123, type=int)
    # parser.add_argument('--simulation', dest='simulation', action='store_true')
    # parser.add_argument('--param', dest='param', action='store_true')
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



    # if args.param and args.simulation:
    #     raise ValueError('cannot do power simulation (--simulation) & param sweep (--param) at the same time')


    #run_all_stage_2(verbose=True)
    #run_all_stage_3(spatials=['max'], verbose=True)
    #run_all_stage_4(spatials=['max'], sample_sizes=[500], verbose=True)

    #run_all_stage_1(n_causals=[2500], exp_vars=[0.5], verbose=False)
    #test_stage_2_params()

    # if args.simulation:
    #     verbose = args.verbose
    #     seed = args.seed
    #     if seed == -1:
    #         for seed in SEEDS:
    #             if args.stage == 1:
    #                 simulation_stage_1(verbose=verbose, seed=seed)
    #             elif args.stage == 2:
    #                 simulation_stage_2(verbose=verbose, seed=seed)
    #             elif args.stage == 3:
    #                 simulation_stage_3(verbose=verbose, seed=seed)
    #             elif args.stage == 4:
    #                 simulation_stage_4(verbose=verbose, seed=seed)
    #     else:
    #         if args.stage == 1:
    #             simulation_stage_1(verbose=verbose, seed=seed)
    #         elif args.stage == 2:
    #             simulation_stage_2(verbose=verbose, seed=seed)
    #         elif args.stage == 3:
    #             simulation_stage_3(verbose=verbose, seed=seed)
    #         elif args.stage == 4:
    #             simulation_stage_4(verbose=verbose, seed=seed)

    # elif args.param:
    #     verbose = args.verbose
    #     model = MODELS[0]
    #     layer = LAYERS[0]
    #     if args.stage == 1:
    #         param_stage_1(verbose=verbose, seed=seed)
    #     elif args.stage == 2:
    #         param_stage_2(verbose=verbose, seed=seed)
    #     elif args.stage == 3:
    #         param_stage_3(model=model, layer=layer, verbose=verbose, seed=seed)
    #     elif args.stage == 4:
    #         param_stage_4(model=model, layer=layer, verbose=verbose)

# def param_stage_1(verbose=False, seed=123):
#     n_causal = 1000
#     exp_var = 0.5

#     config = []
#     config += f'--n_causal {n_causal}'.split()
#     config += f'--exp_var {exp_var:.3f}'.split()
#     config += f'--seed {seed}'.split()
#     if NORMALIZE:
#         config.append('--normalize')
#     config.append('--param_select')

#     cmd = ['python', 'synth_latent.py'] + config
#     if verbose:
#         process = Popen(cmd)
#     else:
#         process = Popen(cmd, stdout=PIPE, stderr=PIPE)
#     stdout, stderr = process.communicate()


def simulation_stage_1(
        bed,
        indiv=None,
        n_causal=1250,
        exp_vars=[0.5],
        normalize=True,
        seeds=[123],
        verbose=False,
        **kwargs,
        ):
    for seed in seeds:
        for exp_var in exp_vars:
            print(seed, exp_var)
            config = []
            config += [bed]
            config += f'--n_causal {n_causal}'.split()
            config += f'--exp_var {exp_var}'.split()
            config += f'--seed {seed}'.split()
            if indiv:
                config += f'--indiv {indiv}'.split()
            if normalize:
                config += [f'--normalize']

            cmd = ['python', 'synth_latent.py'] + config
            print(cmd)
            if verbose:
                process = Popen(cmd)
            else:
                process = Popen(cmd, stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()

# def simulation_stage_1(seed=123, verbose=False):
#     model = 'res50IN'
#     for _, n_causal, exp_var in [x for x in PP if x[0]==46731]:
#         config = []
#         config += f'--n_causal {n_causal}'.split()
#         config += f'--exp_var {exp_var:.3f}'.split()
#         config += f'--seed {seed}'.split()
#         if NORMALIZE:
#             config.append('--normalize')

#         cmd = ['python', 'synth_latent.py'] + config
#         if verbose:
#             process = Popen(cmd)
#         else:
#             process = Popen(cmd, stdout=PIPE, stderr=PIPE)
#         stdout, stderr = process.communicate()


# def param_stage_2(verbose=False, seed=123):
#     n_causal = 1000
#     exp_var = 0.5

#     config = []
#     config += f'--seed {seed}'.split()
#     config += f'--n_causal {n_causal}'.split()
#     config += f'--exp_var {exp_var:.3f}'.split()

#     config += f'--psi {PSI:.1f}'.split()
#     config += f'--gan_name {GAN_NAME}'.split()
#     config += f'--mult_scale {MULT_SCALE:.3f}'.split()
#     if DIFF_NOISE:
#         config.append('--diff_noise')
#     if DEBUG:
#         config.append('--debug')

#     cmd = ['python', 'synth_img.py'] + config
#     if verbose:
#         process = Popen(cmd)
#     else:
#         process = Popen(cmd, stdout=PIPE, stderr=PIPE)
#     stdout, stderr = process.communicate()

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

            cmd = ['python', 'synth_img.py'] + config
            if verbose:
                process = Popen(cmd)
            else:
                process = Popen(cmd, stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()


# def simulation_stage_2(seed=123, verbose=False):
#     model = 'res50IN'
#     for _, n_causal, exp_var in [x for x in PP if x[0]==46731]:
#         config = []
#         config += f'--seed {seed}'.split()
#         config += f'--n_causal {n_causal}'.split()
#         config += f'--exp_var {exp_var:.3f}'.split()

#         config += f'--psi {PSI:.1f}'.split()
#         config += f'--gan_name {GAN_NAME}'.split()
#         config += f'--mult_scale {MULT_SCALE:.3f}'.split()
#         if DIFF_NOISE:
#             config.append('--diff_noise')

#         cmd = ['python', 'synth_img.py'] + config
#         if verbose:
#             process = Popen(cmd)
#         else:
#             process = Popen(cmd, stdout=PIPE, stderr=PIPE)
#         stdout, stderr = process.communicate()


# def simulation_stage_3(
#         verbose=False,
#         seed=123,
#         ):
#     size = 448
#     model = 'res50IN'
#     layer='layer4.2.conv3'
#     tfms = 'tta'
#     spatial = 'mean'
#     n_pcs = N_PCS
#     for _, n_causal, exp_var in [x for x in PP if x[0]==46731]:

#         config = []
#         config += f'--seed {seed}'.split()
#         config += f'--n_causal {n_causal}'.split()
#         config += f'--exp_var {exp_var:.3f}'.split()

#         config += f'--tfms {tfms}'.split()
#         config += f'--model {model}'.split()
#         config += f'--layer {layer}'.split()
#         config += f'--size {size}'.split()
#         config += f'--spatial {spatial}'.split()

#         config += f'--gan_name {GAN_NAME}'.split()
#         config += f'--mult_scale {MULT_SCALE:.3f}'.split()
#         config += f'--n_pcs {n_pcs}'.split()

#         cmd = ['python', 'compute_embeddings.py'] + config
#         if verbose:
#             process = Popen(cmd)
#         else:
#             process = Popen(cmd, stdout=PIPE, stderr=PIPE)
#         stdout, stderr = process.communicate()

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
    for seed in seeds:
        for exp_var in exp_vars:
            img_csv = join(
                    IMG_DIR,
                    IMG_DIR_TEMPL % (gan_name, exp_var, n_causal, mult_scale, seed),
                    ) + '.csv'
            emb = EMB_TEMPL % (
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



# def param_stage_3(model='res50IN', layer=LAYERS[0], verbose=False, seed=123):
#     n_causal = 1000
#     exp_var = 0.5

#     pbar = tqdm(total=len(SIZES)*len(TFMS)*len(SPATIAL))
#     for size in SIZES:
#         for tfms in TFMS:
#             for spatial in SPATIAL:
#                 if not verbose:
#                     s = f'sz: {size}, tfm: {tfms}, sp: {spatial}'
#                     pbar.set_description(s)

#                 config = []
#                 config += f'--seed {seed}'.split()
#                 config += f'--n_causal {n_causal}'.split()
#                 config += f'--exp_var {exp_var:.3f}'.split()

#                 config += f'--tfms {tfms}'.split()
#                 config += f'--model {model}'.split()
#                 config += f'--layer {layer}'.split()
#                 config += f'--size {size}'.split()
#                 config += f'--spatial {spatial}'.split()

#                 config += f'--gan_name {GAN_NAME}'.split()
#                 config += f'--mult_scale {MULT_SCALE:.3f}'.split()
#                 config += f'--n_pcs {N_PCS}'.split()
#                 if DEBUG:
#                     config.append('--debug')

#                 cmd = ['python', 'compute_embeddings.py'] + config
#                 if verbose:
#                     process = Popen(cmd)
#                 else:
#                     process = Popen(cmd, stdout=PIPE, stderr=PIPE)
#                 stdout, stderr = process.communicate()
#                 pbar.update(1)


def simulation_stage_4(
        n_causal=1250,
        exp_vars=[0.5],
        seeds=[123],
        sample_sizes=[25000],
        mult_scale=2.0,
        gan_name='stylegan2_healthy',
        tfms='tta',
        n_pcs=50,
        model='resnet50',
        pretraining='imagenet',
        spatial='mean',
        size=448,
        layers=['L4'],
        verbose=True,
        **kwargs,
        ):


    for seed in seeds:
        for exp_var in exp_vars:
            for sample_size in sample_sizes:
                for layer in layers:
                    emb = join(EMB_DIR, EMB_TEMPL % (
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
                            ))
                    remove_fn_sample = join(EMB_DIR, EMB_TEMPL % (
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
                            ))
                    out_fn = OUT_TEMPL % (
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
                            '%d',
                        )
                    embeddings = pd.read_csv(emb, sep=' ').loc[:, 'IID']
                    sample = embeddings.sample(sample_size, random_state=seed)
                    remove_iid = [x for x in embeddings.IID if not x in sample.IID.values]
                    remove_iid = pd.DataFrame(np.array([remove_iid, remove_iid]).T)
                    remove_iid.to_csv(remove_fn_sample, index=False, header=False)

                    config = []
                    config += f'--bed {bed}.bed'.split() 
                    config += f'--bim {bim}.bim'.split() 
                    config += f'--fam {fam}'.split()
                    config += f'--cov {cov}'.split()
                    config += f'--model lmmInfOnly'.split()
                    config += sum([f'--cov_cols {cc}'.split() for cc in cov_columns], [])
                    config += sum([f'--qcov_cols {qc}'.split() for qc in qcov_columns], [])
                    config += f'--emb {emb}'.split()
                    config += f'--first_pc {first_pc}'.split()
                    config += f'--last_pc {last_pc}'.split()
                    config += f'--out_dir {OUT_DIR}'.split()
                    config += f'--out {out_fn}'.split()
                    config += f'--threads {threads}'.split()
                    config += f'--ref_map {ref_map}'.split()
                    config += f'--ldscores {ldscores}'.split()

                    config += f'--remove {remove_fn_sample}'.split()
                    # if remove:
                    #     config += f'--remove {remove}'.split() 

                    cmd = ['python', PATH_TO_GWAS] + config
                    if verbose:
                        process = Popen(cmd)
                    else:
                        process = Popen(cmd, stdout=PIPE, stderr=PIPE)
                    stdout, stderr = process.communicate()




#def simulation_stage_4(verbose=False, seed=123):
#    size = 448
#    model = 'res50IN'
#    layer='layer4.2.conv3'
#    tfms = 'tta'
#    spatial = 'mean'
#    n_pcs = 10
#    #for sample_size, n_causal, exp_var in [x for x in PP if x[0]==46731]:
#    for sample_size, n_causal, exp_var in sorted(PP):
#        config = []
#        config += f'--n_causal {n_causal}'.split()
#        config += f'--exp_var {exp_var:.3f}'.split()
#        config += f'--sample_size {sample_size}'.split()

#        config += f'--gan_name {GAN_NAME}'.split()
#        config += f'--n_pcs {n_pcs}'.split()
#        config += f'--mult_scale {MULT_SCALE:.3f}'.split()
#        config += f'--threads {THREADS}'.split()
#        config += f'--seed {seed}'.split()

#        config += f'--model {model}'.split()
#        config += f'--layer {layer}'.split()
#        config += f'--size {size}'.split()
#        config += f'--tfms {tfms}'.split()
#        config += f'--spatial {spatial}'.split()

#        if PLOTTING:
#            config.append('--plotting')

#        cmd = ['python', 'run_gwas.py'] + config
#        if verbose:
#            process = Popen(cmd)
#        else:
#            process = Popen(cmd, stdout=PIPE, stderr=PIPE)
#        stdout, stderr = process.communicate()


# def param_stage_4(model='res50IN', layer=LAYERS[0], verbose=False, seed=123):
#     n_causal = 1000
#     exp_var = 0.5
#     sample_size = SAMPLE_SIZES[-2]

#     pbar = tqdm(total=len(SIZES)*len(TFMS)*len(SPATIAL)*len(layer.split()))
#     for size in SIZES:
#         for tfms in TFMS:
#             for spatial in SPATIAL:
#                 for l in layer.split():
#                     if not verbose:
#                         s = f'sz: {size}, tfm: {tfms}, sp: {spatial}, layer: {l}'
#                         pbar.set_description(s)

#                     config = []
#                     config += f'--n_causal {n_causal}'.split()
#                     config += f'--exp_var {exp_var:.3f}'.split()
#                     config += f'--gan_name {GAN_NAME}'.split()
#                     config += f'--n_pcs {N_PCS}'.split()
#                     config += f'--mult_scale {MULT_SCALE:.3f}'.split()
#                     config += f'--threads {THREADS}'.split()
#                     config += f'--sample_size {sample_size}'.split()
#                     config += f'--seed {seed}'.split()


#                     config += f'--model {model}'.split()
#                     config += f'--layer {l}'.split()
#                     config += f'--size {size}'.split()
#                     config += f'--tfms {tfms}'.split()
#                     config += f'--spatial {spatial}'.split()

#                     config.append('--param_select')
#                     if PLOTTING:
#                         config.append('--plotting')

#                     cmd = ['python', 'run_gwas.py'] + config
#                     if verbose:
#                         process = Popen(cmd)
#                     else:
#                         process = Popen(cmd, stdout=PIPE, stderr=PIPE)
#                     stdout, stderr = process.communicate()

#                     if not verbose:
#                         pbar.update(1)


































##### delete? #####

def run_all_stage_1(n_causals=[100, 1000], exp_vars=[0.5, 0.6], verbose=False):
    if not verbose:
        pbar = tqdm(total=len(n_causals)*len(exp_vars))
    for n_causal in n_causals:
        for exp_var in exp_vars:
            if not verbose:
                pbar.set_description(f'nc: {n_causal}, ev: {exp_var:.1f}')

            config = []
            config += f'--n_causal {n_causal}'.split()
            config += f'--exp_var {exp_var:.3f}'.split()
            config += f'--seed {SEED}'.split()
            if NORMALIZE:
                config.append('--normalize')
            if DEBUG:
                config.append('--debug')

            cmd = ['python', 'synth_latent.py'] + config
            if verbose:
                process = Popen(cmd)
            else:
                process = Popen(cmd, stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()

            if not verbose:
                pbar.update(1)

'''
def test_stage_2_params():
    n_causal = 2500
    exp_var = 0.5
    psis = [0.2, 0.4, 0.6, 0.8]
    mult_scales = [0.5, 1.0, 1.5, 2., 3., 4.]
    pbar = tqdm(total=len(psis)*len(mult_scales))
    for psi in psis:
        for scale in mult_scales:
            pbar.set_description(f'psi: {psi}, ms: {scale:.1f}')
            config = []
            config += f'--n_causal {n_causal}'.split()
            config += f'--exp_var {exp_var:.3f}'.split()

            config += f'--psi {psi:.1f}'.split()
            config += f'--gan_name {GAN_NAME}'.split()
            config += f'--mult_scale {scale:.3f}'.split()
            if DIFF_NOISE:
                config.append('--diff_noise')
            config.append('--pretest')

            cmd = ['python', 'synth_img.py'] + config
            if True:
                process = Popen(cmd)
            else:
                process = Popen(cmd, stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()

            pbar.update(1)
'''

def run_all_stage_2(n_causals=[100], exp_vars=[0.5], verbose=False):
    if not verbose:
        pbar = tqdm(total=len(n_causals)*len(exp_vars))
    for n_causal in n_causals:
        for exp_var in exp_vars:
            if not verbose:
                pbar.set_description(f'nc: {n_causal}, ev: {exp_var:.1f}')

            config = []
            config += f'--n_causal {n_causal}'.split()
            config += f'--exp_var {exp_var:.3f}'.split()

            config += f'--psi {PSI:.1f}'.split()
            config += f'--gan_name {GAN_NAME}'.split()
            config += f'--mult_scale {MULT_SCALE:.3f}'.split()
            if DIFF_NOISE:
                config.append('--diff_noise')
            if DEBUG:
                config.append('--debug')

            cmd = ['python', 'synth_img.py'] + config
            if verbose:
                process = Popen(cmd)
            else:
                process = Popen(cmd, stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()

            if not verbose:
                pbar.update(1)

def run_all_stage_3(
        n_causals=[100],
        exp_vars=[0.5],
        sizes=[224],
        tfmss=['basic'],
        #tfmss=['tta'],
        models=['res18IN'],
        layers=['layer3.1.conv2 layer4.1.conv2'],
        spatials=['mean'],
        verbose=False,
        ):
    if not verbose:
        pbar = tqdm(total=len(n_causals)*len(exp_vars)*len(sizes)*len(tfmss)*len(models)*len(spatials))
    for n_causal in n_causals:
        for exp_var in exp_vars:
            for size in sizes:
                for tfms in tfmss:
                    for spatial in spatials:
                        for i, (model, layer) in enumerate(zip(models, layers)):
                            if not verbose:
                                s = f'nc: {n_causal}, ev: {exp_var:.1f}, sz: {size}, tfm: {tfms}, sp: {spatial}, model: {i+1}'
                                pbar.set_description(s)

                            config = []
                            config += f'--n_causal {n_causal}'.split()
                            config += f'--exp_var {exp_var:.3f}'.split()

                            config += f'--tfms {tfms}'.split()
                            config += f'--model {model}'.split()
                            config += f'--layer {layer}'.split()
                            config += f'--size {size}'.split()
                            config += f'--spatial {spatial}'.split()

                            config += f'--gan_name {GAN_NAME}'.split()
                            config += f'--mult_scale {MULT_SCALE:.3f}'.split()
                            config += f'--n_pcs {N_PCS}'.split()
                            if DEBUG:
                                config.append('--debug')

                            cmd = ['python', 'compute_embeddings.py'] + config
                            if verbose:
                                process = Popen(cmd)
                            else:
                                process = Popen(cmd, stdout=PIPE, stderr=PIPE)
                            stdout, stderr = process.communicate()
                            if not verbose:
                                pbar.update(1)


def run_all_stage_4(
        n_causals=[100],
        exp_vars=[0.5],
        sizes=[224],
        tfmss=['basic'],
        #tfmss=['tta'],
        models=['res18IN'],
        layers=['layer3.1.conv2 layer4.1.conv2'],
        sample_sizes=[1000],
        spatials=['mean'],
        verbose=False,
        ):
    if not verbose:
        pbar = tqdm(total=len(n_causals)*len(exp_vars)*len(sizes)*len(tfmss)*len(models)*max([len(x.split()) for x in layers])*len(spatials)*len(sample_size))
    for n_causal in n_causals:
        for exp_var in exp_vars:
            for size in sizes:
                for tfms in tfmss:
                    for spatial in spatials:
                        for sample_size in sample_sizes:
                            for i, (model, model_layer) in enumerate(zip(models, layers)):
                                for j, layer in enumerate(model_layer.split()):
                                    if not verbose:
                                        s = f'nc: {n_causal}, ev: {exp_var:.1f}, sz: {size}, tfm: {tfms}, model: {i+1}, sp: {spatial}, layer: {j+1}, ss: {sample_size}'
                                        pbar.set_description(s)

                                    config = []
                                    config += f'--n_causal {n_causal}'.split()
                                    config += f'--exp_var {exp_var:.3f}'.split()
                                    config += f'--gan_name {GAN_NAME}'.split()
                                    config += f'--n_pcs {N_PCS}'.split()
                                    config += f'--mult_scale {MULT_SCALE:.3f}'.split()
                                    config += f'--threads {THREADS}'.split()
                                    config += f'--sample_size {sample_size}'.split()
                                    config += f'--seed {SEED}'.split()


                                    config += f'--model {model}'.split()
                                    config += f'--layer {layer}'.split()
                                    config += f'--size {size}'.split()
                                    config += f'--tfms {tfms}'.split()
                                    config += f'--spatial {spatial}'.split()

                                    if DEBUG:
                                        config.append('--debug')
                                    if PLOTTING:
                                        config.append('--plotting')

                                    cmd = ['python', 'run_gwas.py'] + config
                                    if verbose:
                                        process = Popen(cmd)
                                    else:
                                        process = Popen(cmd, stdout=PIPE, stderr=PIPE)
                                    stdout, stderr = process.communicate()

                                    if not verbose:
                                        pbar.update(1)



if __name__ == '__main__':
    main()
