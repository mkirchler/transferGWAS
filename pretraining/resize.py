import warnings
import os
from os.path import join
import argparse
from glob import glob

from tqdm import tqdm

import PIL

from torchvision import transforms



warnings.filterwarnings('ignore', category=UserWarning)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dir', type=str, help='path to unzipped train directory')
    parser.add_argument('test_dir', type=str, help='path to unzipped test directory')
    parser.add_argument('out_dir', type=str, help='path for the resized images')
    parser.add_argument('--size', type=int, default=672, help='resize to which size')
    args = parser.parse_args()

    resize_all_imgs(args.train_dir, args.out_dir, args.size)
    resize_all_imgs(args.test_dir, args.out_dir, args.size)


def resize_all_imgs(input_dir, output_dir, max_size, extension='*_*.jpeg'):
    img_files = glob(join(input_dir, extension))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for fn in tqdm(img_files):
        resize_one(fn, max_size, output_dir)


def resize_one(fn, max_size, output_dir):
    tfm = transforms.Compose([
        transforms.CenterCrop(size=int(2.25*max_size)),
        transforms.Resize(size=max_size, interpolation=PIL.Image.BICUBIC)
        ])
    try:
        img = PIL.Image.open(fn)
        size_to = resize_to(img, 1792, use_min=False)
        res_img = img.resize(size_to, resample=PIL.Image.BICUBIC).convert('RGB')
        res_img = tfm(res_img).convert('RGB')
        out_fn = join(output_dir, fn.split('/')[-1])
        res_img.save(out_fn)
    except:
        print('cannot handle file', fn)


def resize_to(img, target_size, use_min=False):
    w, h = img.size
    min_size = (min if use_min else max)(w, h)
    ratio = target_size / min_size
    return int(w * ratio), int(h * ratio)


if __name__ == '__main__':
    main()
