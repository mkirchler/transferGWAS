from os.path import join
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

from tqdm import tqdm

from sklearn.decomposition import PCA

# TODO requires torch.__version__ >= 1.6 (or >= 1.5, but definitely > 1.4)
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms, models

LAYERS_RES18 = [
        lambda m: m.layer1[-1].conv2,
        lambda m: m.layer2[-1].conv2,
        lambda m: m.layer3[-1].conv2,
        lambda m: m.layer4[-1].conv2,
        ]
LAYERS_RES50 = [
        lambda m: m.layer1[-1].conv3,
        lambda m: m.layer2[-1].conv3,
        lambda m: m.layer3[-1].conv3,
        lambda m: m.layer4[-1].conv3,
        ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            'img_csv',
            type=str,
            help='input image .csv-file. Needs to have columns "IID" and "path". '
            'Can contain additional column "instance" for multiple images per '
            'IID (e.g. left and right eye).',
            )
    parser.add_argument(
            'out_dir',
            type=str,
            default='results',
            help='Directory to save the results',
            )
    parser.add_argument('--save_str', type=str, help='Optional name of file to save to. Needs to contain two `%s` substrings (for layer and explained variance).')
    parser.add_argument('--img_size', type=int, default=448, help='Input image size')
    parser.add_argument(
            '--tfms',
            type=str,
            default='basic',
            help='What kind of image transformations to use.',
            )
    parser.add_argument('--dev', default='cuda:0', type=str, help='cuda device to use')
    parser.add_argument('--n_pcs', type=int, default=50, help='How many PCs to export')
    parser.add_argument('--num_threads', type=int, default=1, help='How many threads to use')
    parser.add_argument(
            '--model',
            type=str,
            default='resnet50',
            choices=['resnet18', 'resnet34', 'resnet50'],
            help='Model architecture.',
            )
    parser.add_argument(
            '--pretraining',
            type=str,
            default='imagenet',
            help='What weights to load into the model. If `imagenet`, load the default'
            ' pytorch weights; otherwise, specify path to `.pt` with state dict',
            )
    parser.add_argument(
            '--layer',
            type=str,
            default=['L4'],
            nargs='+',
            help='At what layer to extract the embeddings from the network. '
            'Can be either `L1`, `L2`, `L3`, `L4` for the default locations in layer 1-4, '
            'or can name a specific module in the architecture such as `layer4.2.conv3`. '
            'Multiple layers can be specified and will be exported to corresponding files.',
            )
    args = parser.parse_args()

    dsets = load_data_from_csv(
            args.img_csv,
            tfms=args.tfms,
            img_size=args.img_size,
            )
    model = load_model(
            args.model,
            args.pretraining,
            args.dev,
            )
    layer_funcs = load_layers(
            args.model,
            args.layer,
            )

    embeddings = []
    for dset in dsets:
        embeddings_i = compute_embeddings(
                model=model,
                layer_funcs=layer_funcs,
                dset=dset,
                dev=args.dev,
                num_threads=args.num_threads,
                )
        embeddings.append(embeddings_i)
    embeddings = join_embeddings(embeddings, dsets)

    os.makedirs(args.out_dir, exist_ok=True)
    for layer_name, layer_embedding in zip(args.layer, embeddings):
        pca = PCA(n_components=args.n_pcs)
        pca_embedding = pca.fit_transform(layer_embedding)
        explained_var = pca.explained_variance_ratio_

        pret_part = args.pretraining.split('/')[-1].split('.')[0]
        if not args.save_str:
            save_str = join(
                    args.out_dir,
                    f'{args.model}_{pret_part}_{layer_name}.txt',
                    )
            save_str_evr = join(
                    args.out_dir,
                    f'{args.model}_{pret_part}_{layer_name}_explained_variance.txt',
                    )
        else:
            save_str = join(
                    args.out_dir,
                    args.save_str % (layer_name, ''),
                    )
            save_str_evr = join(
                    args.out_dir,
                    args.save_str % (layer_name, '_explained_variance'),
                    )

        to_file(
                pca_embedding,
                explained_var,
                dsets[0].ids,
                save_str=save_str,
                save_str_evr=save_str_evr,
                )


def load_data_from_csv(fn, tfms='basic', img_size=448):
    '''load csv into ImageData instance(s)'''
    df = pd.read_csv(fn)
    subset = None
    instance_grouping = 'instance'
    if instance_grouping in df.columns:
        dsets = []
        iid = np.unique(df.IID)
        for level, sub_df in df.groupby(instance_grouping):
            iid = np.intersect1d(iid, sub_df.IID)

        for level, sub_df in df.groupby(instance_grouping):
            sub_df.index = sub_df.IID
            dset = ImageData(
                    sub_df.loc[iid],
                    tfms=tfms,
                    img_size=img_size,
                    subset=subset,
                    )
            dsets.append(dset)
    else:
        dsets = [ImageData(
                    df,
                    tfms=tfms,
                    img_size=img_size,
                    subset=subset,
                    )]
    return dsets


def get_tfms_basic(size=224):
    '''only Resize and Normalize'''
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    tfms = transforms.Compose([
        transforms.Resize(size=size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        ])
    return tfms


def get_tfms_augmented(size=224):
    '''get test-time augmentation tfms'''
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    resize = transforms.Resize(size=size)

    tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        ])

    def tfm(x):
        x = resize(x)
        flipped = x.transpose(Image.FLIP_LEFT_RIGHT)

        x90 = x.transpose(Image.ROTATE_90)
        flipped90 = flipped.transpose(Image.ROTATE_90)
        x180 = x.transpose(Image.ROTATE_180)
        flipped180 = flipped.transpose(Image.ROTATE_180)
        x270 = x.transpose(Image.ROTATE_270)
        flipped270 = flipped.transpose(Image.ROTATE_270)

        imgs = [x, flipped, x90, flipped90, x180, flipped180, x270, flipped270]
        imgs = torch.stack([tfms(img) for img in imgs])
        return imgs

    return tfm


def load_model(model, pretraining='imagenet', dev='cuda:0'):
    '''prepare model and load pretrained weights'''
    if model == 'resnet18':
        m_func = models.resnet18
    elif model == 'resnet34':
        m_func = models.resnet34
    elif model == 'resnet50':
        m_func = models.resnet50

    M = m_func(pretrained=True)
    if pretraining != 'imagenet':
        M.load_state_dict(torch.load(pretraining, map_location='cpu'))
    return prep_model(M, dev)


def prep_model(model, dev):
    '''set to eval, move to device and remove `inplace` operations'''
    model = model.eval().to(dev)
    for mod in model.modules():
        if hasattr(mod, 'inplace'):
            mod.inplace = False
    return model


def load_layers(model, layers):
    '''get functions to return layers, for hooks'''
    if model in ['resnet18', 'resnet34']:
        LF = LAYERS_RES18
    elif model == 'resnet50':
        LF = LAYERS_RES50
    lfs = []
    for layer in layers:
        if layer in ['L1', 'L2', 'L3', 'L4']:
            layer_func = LF[int(layer[1])-1]
        else:
            layer_func = lambda m, l_str=layer: dict(m.named_modules())[l_str]
        lfs.append(layer_func)
    return lfs


def compute_embeddings(
        model,
        layer_funcs,
        dset,
        dev='cuda:0',
        num_threads=1,
       ):
    '''compute all embeddings in dset at layer_funcs in model
    
    # Parameters:
    model (nn.Module): pretrained pytorch model
    layer_funcs (list of functions): each element is a function that takes model
                as argument and returns the corresponding submodule (for registering hooks)
    dset (ImageData): dataset for which to extract the embeddings
    dev (torch.device or str): where to perform the computations
    num_threads (int): how many torch CPU-threads
    '''
    torch.set_num_threads(num_threads)

    output = [[] for _ in layer_funcs]
    hooks = []
    for idx, layer_func in enumerate(layer_funcs):
        hook = layer_func(model).register_forward_hook(
                lambda m, i, o, idx=idx: output[idx].append(o.detach()))
        hooks.append(hook)

    tmp_img = dset[0][0]
    if len(tmp_img.shape) == 3:
        tmp_img.unsqueeze_(0)
    n_tfm = tmp_img.shape[0]
    _ = model(tmp_img.to(dev))
    shapes = [out[0].shape[1]*n_tfm for out in output]
    for i in range(len(output)): output[i] = []

    embeddings = [np.empty((len(dset), shape)) for shape in shapes]

    for sample_idx, (img, iid) in tqdm(enumerate(dset), total=len(dset)):
        with torch.no_grad():
            if len(img.shape) == 3:
                img = img.view(1, *img.shape)
            _ = model(img.to(dev))
        for layer_idx, out in enumerate(output):
            # conv layer
            if len(out[0].shape) > 2:
                embedding = out[0].mean([-1, -2]).flatten()
            # non-conv layer
            else:
                embedding = out[0].flatten()
            embeddings[layer_idx][sample_idx, :] = embedding.cpu().numpy()
        for i in range(len(output)): output[i] = []

    # clean up model afterwards again
    for hook in hooks:
        hook.remove()
    return embeddings


def join_embeddings(embeddings, dsets):
    '''take list of list of embeddings and return joined embeddings

    # Parameters:
    embeddings (list of list of np.array): nested list of embeddings with structure:
                [[dset1-layer1, dset1-layer2, ...], [dset2-layer1, dset2-layer2, ...], ...]
    dsets (list of ImageData): corresponding to embeddings, only used for checking IIDs
    '''
    all_ids = np.array([d.ids for d in dsets])
    # make sure all dsets have the same ids and ordering
    assert np.all(np.repeat(np.expand_dims(all_ids[0], 0), len(dsets), axis=0) == all_ids)

    out = []
    for i in range(len(embeddings[0])):
        out.append(np.concatenate([emb[i] for emb in embeddings], axis=1))
    return out


def to_file(
        embeddings,
        explained_var,
        iid,
        save_str,
        save_str_evr,
        pheno_name='PC_%d',
        ):
    '''save embeddings and explained variance to files'''
    evr = pd.DataFrame(
            [(pheno_name % d, e) for d, e in enumerate(explained_var)],
            columns=['PC', 'explained_variance_ratio'],
            )
    evr.to_csv(save_str_evr, index=False, header=True)

    data = pd.DataFrame(
            dict(
                [('FID', iid), ('IID', iid)] +
                [(pheno_name % d, embeddings[:, d]) for d in range(embeddings.shape[1])]
                )
            )
    data.to_csv(save_str, sep=' ', index=False, header=True)


class ImageData(Dataset):
    def __init__(self, df, tfms='basic', img_size=448, subset=100):
        self.ids = df.IID.values
        self.path = df.path.values
        self.img_size = img_size
        self.tfms = get_tfms_basic(img_size) if tfms == 'basic' else get_tfms_augmented(img_size)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = self._load_item(idx)
        id = self.ids[idx]
        return img, id

    def _load_item(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        p = self.path[idx]
        img = Image.open(p)
        if self.tfms:
            img = self.tfms(img)
        return img


if __name__ == '__main__':
    main()
