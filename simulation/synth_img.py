import shutil
import os
from os.path import join
import numpy as np
import pandas as pd
import torch
# from torch.utils.data import Dataset
# import torchvision
# from torchvision import models
import PIL
# from glob import glob
import argparse
# from sklearn.decomposition import PCA
from tqdm import tqdm

# from matplotlib import pyplot as plt

# from pyplink import PyPlink
from stylegan2_pytorch import Trainer
from stylegan2_pytorch.stylegan2_pytorch import image_noise

# import toml

from util import LATENT_DIR, get_latent_bolt, IMG_DIR, get_img_dir
torch.set_num_threads(2)


# CONFIG = toml.load('config.toml')
# LATENT_DIR = 'latent'
# LATENT_BOLT_TEMPL = 'exp_var%.3f_nc%d_seed%d.txt'

# EMB_DIR = CONFIG['emb_dir']
# IMG_DIR = 'images'
# IMG_DIR_TEMPL = 'exp_%s_var%.3f_nc%d_sc%.2f_seed%d'
# MDIR = CONFIG['stylegan2_models']
# RDIR = CONFIG['stylegan2_models']
# IMG_DIR_TEMPL = CONFIG['img_dir_templ']

os.makedirs(IMG_DIR, exist_ok=True)

# LATENT_DIR = 'latent'
# IMG_DIR = 'images'
# EMB_DIR = 'embeddings'
# MDIR = 'gan_models'
# RDIR = 'gan_models'


# IMG_DIR_TEMPL = 'exp_%s_var%.3f_nc%d_sc%.2f_seed%d'
#IMG_DIR_TEMPL = 'exp_%s_var%.3f_nc%d_sc%.2f_psi%.1f'
# LATENT_BOLT_TEMPL = 'exp_var%.3f_nc%d_seed%d.txt'
# LATENT_PKL_TEMPL = 'exp_var%.3f_nc%d_seed%d.pkl'

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--debug', dest='debug', action='store_true')

    parser.add_argument('--seed', dest='seed', default=123, type=int)
    parser.add_argument('--n_causal', dest='n_causal', default=100, type=int)
    parser.add_argument('--exp_var', dest='exp_var', default=0.5, type=float)
    parser.add_argument('--psi', dest='psi', default=0.4, type=float)
    parser.add_argument('--gan_name', dest='gan_name', default='stylegan2_healthy', type=str)
    parser.add_argument('--diff_noise', dest='diff_noise', action='store_true')
    parser.add_argument('--mult_scale', dest='mult_scale', default=1., type=float)
    parser.add_argument('--models_dir', default='../models', type=str)


    args = parser.parse_args()


    # if args.debug:
    #     subset = 1000
    # else:
    #     subset = None


    # config = dict(
    #         n_causal=args.n_causal,
    #         exp_var=args.exp_var,
    #         psi=args.psi,
    #         same_noise=not args.diff_noise,
    #         name=args.gan_name,
    #         mult_scale=args.mult_scale,
    #         subset=None,
    #         seed=args.seed,
    #         )

    synthesize_gwas_data(
            n_causal=args.n_causal,
            exp_var=args.exp_var,
            psi=args.psi,
            models_dir=args.models_dir,
            same_noise=not args.diff_noise,
            name=args.gan_name,
            mult_scale=args.mult_scale,
            subset=None,
            seed=args.seed,
            )


def synthesize_gwas_data(
        n_causal=100,
        exp_var=0.5,
        img_size=512,
        models_dir='../models',
        checkpoint=-1,
        same_noise=True,
        psi=0.6,
        name='stylegan2_healthy',
        mult_scale=1.,
        subset=None,
        seed=123,
        ):
    '''synthesize images from latent codes with StyleGAN2

    Load latent codes from corresponding LATENT_BOLT_TEMPL and generate synthetic images via StyleGAN2 model

    # Parameters
    n_causal (int): number of causal SNPs
    exp_var (float in [0, 1]): percentage of explained variance by causal SNPs
    img_size (int): size of images to be created (determined by training scheme of StyleGAN2)
    checkpoint (int): which epoch to use; if -1, load latest epoch in model directory
    same_noise (bool): use the same noise vector for all images
    psi (float in [0, 1]): truncation parameter for images, trade-off between image quality and diversity
    name (str): name of StyleGAN2 model in models directory
    mult_scale (float): multiplier for standard-normal style vector (input), to increase/decrease diversity of images
    subset (None or int): only create subset of images, for debugging
    seed (int): random seed
    '''
    pth = join(LATENT_DIR, get_latent_bolt(exp_var, n_causal, seed))
    latent = pd.read_csv(pth, sep=' ', index_col=1).drop('FID', 1)
    if subset is not None:
        latent = latent.sample(subset, random_state=123)

    T = Trainer(name, models_dir=models_dir, results_dir=models_dir, image_size=img_size, network_capacity=16)
    T.load(checkpoint)

    if psi is None:
        psi = T.trunc_psi

    if same_noise:
        N = image_noise(1, img_size)

    out_img_dir = join(IMG_DIR, get_img_dir(name, exp_var, n_causal, mult_scale, seed))
    if os.path.isdir(out_img_dir):
        shutil.rmtree(out_img_dir)
    os.mkdir(out_img_dir)

    for i, lat in tqdm(latent.iterrows(), total=len(latent)):
        if not same_noise:
            N = image_noise(1, img_size)
        L = [(mult_scale*torch.from_numpy(lat.values).view(1,-1).float().cuda(), 8)]
        gen = T.generate_truncated(T.GAN.S, T.GAN.G, L, N, trunc_psi=psi)
        gen = (gen.cpu().double()[0].permute(1, 2, 0).numpy()*255).astype(np.uint8)
        gen = PIL.Image.fromarray(gen)
        path = join(out_img_dir, f'{i}.jpg')
        gen.save(path)

    cwd = os.getcwd()
    paths = [join(cwd, out_img_dir, f'{i}.jpg') for i in latent.index]
    df = pd.DataFrame(np.array([latent.index, paths]).T, columns=['IID', 'path'])
    csv_path = out_img_dir + '.csv'
    df.to_csv(csv_path, index=False)
 
# def copy_sample(pth, out_dir=join(IMG_DIR, 'sample'), n=100):
#     x = os.listdir(pth)
#     sample = np.random.choice(x, n, replace=False)

#     if os.path.isdir(out_dir):
#         shutil.rmtree(out_dir)
#     os.mkdir(out_dir)

#     for fn in sample:
#         img = PIL.Image.open(join(pth, fn))
#         img.save(join(out_dir, fn))

# def plot_gen(gen, s=20, nrow=2):
#     fig = plt.figure(figsize=(s, s))
#     plt.imshow(torchvision.utils.make_grid(gen, nrow=nrow).permute(1,2,0).cpu())

if __name__ == '__main__':
    main()
