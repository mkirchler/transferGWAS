import argparse

import numpy as np

from tqdm import tqdm

import torch.multiprocessing
import torch
from torch import nn
from torch import optim

from data import build_retina_dataset
from models import SpatialDecoder, FlattenLayer, ResNetFeatures

WANDB = False
if WANDB:
    import wandb
    PROJECT_NAME = 'Multi-task DR/AE training'

torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir', type=str, help='path to EyePACs dataset')
    parser.add_argument('labels', type=str, help='path to full labels csv')
    parser.add_argument('--epochs', dest='epochs', default=100, type=int)
    parser.add_argument('--res_depth', dest='res_depth', choices=[18, 34, 50], default=50, type=int, help='depth of ResNet body')
    parser.add_argument('--size', dest='size', default=448, type=int, choices=[224, 448], help='input image size')
    parser.add_argument('--bs', dest='bs', default=25, type=int, help='batch size')
    parser.add_argument('--lr', dest='lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--num_workers', dest='num_workers', default=1, type=int, help='number of parallel CPU workers for data loading')
    parser.add_argument('--dev', dest='dev', default='cuda:0', type=str, help='cuda device to use')
    parser.add_argument('--save_path', dest='save_path', default='models/best_model.pt', type=str, help='where to save models')
    args = parser.parse_args()

    n_neurons = 2048 if args.res_depth == 50 else 512
    train_pct = 0.8

    configs = get_configs(
            img_dir=args.img_dir,
            labels_path=args.labels,
            subset=1000,
            size=args.size,
            bs=args.bs,
            num_workers=args.num_workers,
            train_pct=train_pct,
            n_neurons=n_neurons,
            )

    M, C = train(
            configs=configs,
            epochs=args.epochs,
            lr=args.lr,
            res_depth=args.res_depth,
            dev=args.dev,
            # start_from=start_from,
            save_path=args.save_path,
            )


def get_configs(img_dir, labels_path, size=224, bs=25, num_workers=12, train_pct=0.8, subset=None, verbose=False, n_neurons=512):
    configs = []
    train_loader, valid_loader = build_retina_dataset(
            img_dir=img_dir,
            labels_path=labels_path,
            ae=False,
            size=size,
            batch_size=bs,
            num_workers=num_workers,
            train_pct=train_pct,
            subset=subset,
            seed=123,
            )
    configs.append({
                'name': 'Diabetic Retinopathy - MSE',
                'train_loader': train_loader,
                'valid_loader': valid_loader,
                'head': nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    FlattenLayer(),
                    nn.Linear(n_neurons, 1)
                    ),
                'train_loss': nn.MSELoss(reduction='sum'),
                'valid_loss': nn.MSELoss(reduction='mean'),
            })

    train_loader, valid_loader = build_retina_dataset(
            img_dir=img_dir,
            labels_path=labels_path,
            ae=True,
            size=size,
            batch_size=bs,
            num_workers=num_workers,
            train_pct=train_pct,
            subset=subset,
            seed=123,
            )
    configs.append({
                'name': 'AE - MSE',
                'train_loader': train_loader,
                'valid_loader': valid_loader,
                'head': SpatialDecoder(d=n_neurons),
                'train_loss': spatial_mse,
                'valid_loss': nn.MSELoss(reduction='mean'),
                })

    return configs


def train(
        configs,
        epochs=10,
        subset=100,
        res_depth=50,
        lr=1e-3,
        dev='cuda:0',
        save_path='models/best_model.pt',
        # start_from=1,
        ):
    if WANDB:
        wandb.init(project=PROJECT_NAME)

    model = ResNetFeatures(resnet=res_depth)
    model = model.to(dev)
    if WANDB:
        mmodels = [model]
        for dic in configs:
            mmodels.append(dic['head'])
        wandb.watch(mmodels, log='all', log_freq=5)

    params = list(model.parameters())
    for dic in configs:
        dic['head'] = dic['head'].to(dev)
        params += list(dic['head'].parameters())

    opt = optim.Adam(
            params,
            betas=(0.9, 0.99),
            )
    steps_per_epoch = min([len(dic['train_loader']) for dic in configs])
    scheduler = optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch
            )

    # if start_from > 1:
    #     model, configs, opt, scheduler = load_states(model, configs, opt, scheduler, epoch=start_from, dev=dev)
    start_from = 1

    best_loss = np.inf
    for epoch in range(start_from, epochs+1):
        print(f'starting epoch {epoch}:')
        train_losses = train_one_epoch(model, configs, opt, scheduler, dev=dev)
        if WANDB:
            for i, t in enumerate(train_losses):
                wandb.log({f'train_loss_{i}': t}, commit=False)
        valid_losses = eval_one_epoch(model, configs, dev=dev)
        if WANDB:
            for i, v in enumerate(valid_losses):
                commit = i == (len(valid_losses)-1)
                wandb.log({f'valid_loss_{i}': v}, commit=commit)
        s = '\t'.join([('%s: %.4f' % (dic['name'], loss)) for dic, loss in zip(configs, valid_losses)])

        print(f'validation losses:\n\t{s}\n')
        # save_states(model, configs, opt, scheduler, epoch=epoch+1)
        if best_loss > valid_losses[0]:
            print(f'updating model: last best loss: {best_loss:.3f}, new best loss: {valid_losses[0]:.3f}')
            best_loss = valid_losses[0]
            torch.save(model.r.state_dict(), save_path)
    return model, configs


# def save_states(model, configs, opt, scheduler, epoch=1):
#     param = {
#             'model_sd': model.state_dict(),
#             'opt': opt.state_dict(),
#             'sched': scheduler.state_dict(),
#             }
#     for i, dic in enumerate(configs):
#         param[f'head_{i}_sd'] = dic['head'].state_dict()

#     torch.save(param, f'models/states{epoch}_dr.tar')


# def load_states(model, configs, opt, scheduler, epoch=1, dev='cpu'):
#     param = torch.load(f'models/states{epoch}_dr.tar', map_location='cpu')
#     model.load_state_dict(param['model_sd'])
#     model = model.to(dev)
#     for i, dic in enumerate(configs):
#         dic['head'].load_state_dict(param[f'head_{i}_sd'])
#         dic['head'] = dic['head'].to(dev)

#     opt.load_state_dict(param['opt'])
#     scheduler.load_state_dict(param['sched'])

#     return model, configs, opt, scheduler


def train_one_epoch(model, configs, opt, scheduler=None, dev='cuda:0'):
    model.train()
    for dic in configs: dic['head'].train()

    loaders = [dic['train_loader'] for dic in configs]
    loss_funcs = [dic['train_loss'] for dic in configs]
    heads = [dic['head'] for dic in configs]
    running_losses, num_items = [0. for _ in configs], [0 for _ in configs]
    n_batches = min([len(loader) for loader in loaders])

    pbar = tqdm(enumerate(zip(*loaders)), total=n_batches, ncols=80)
    for i, batches in pbar:
        for i, ((inp, target), lf, head) in enumerate(zip(batches, loss_funcs, heads)):
            inp, target = inp.to(dev), target.to(dev)
            opt.zero_grad()
            output = model(inp)
            output = head(output)
            if len(target.shape) > 3:
                target = nn.functional.adaptive_avg_pool2d(target, (224, 224))
            loss = lf(output.view(*target.shape), target)
            loss.backward()
            opt.step()
            with torch.no_grad():
                running_losses[i] += loss.item()
                num_items[i] += len(target)
        if scheduler:
            scheduler.step()

        description = (
                'running losses:' + len(batches)*'\t%.4f'
                ) % (*[loss/num for loss, num in zip(running_losses, num_items)],)
        pbar.set_description(description)
    return [loss/num for loss, num in zip(running_losses, num_items)]

def eval_one_epoch(model, configs, dev='cuda:0'):
    model.eval()
    for dic in configs: dic['head'].eval()

    loaders = [dic['valid_loader'] for dic in configs]
    loss_funcs = [dic['valid_loss'] for dic in configs]
    heads = [dic['head'] for dic in configs]
    losses = []

    with torch.no_grad():
        for i, (loader, lf, head) in enumerate(zip(loaders, loss_funcs, heads)):
            targets, outputs = [], []
            for inp, target in tqdm(loader):
                inp = inp.to(dev)
                output = model(inp)
                output = head(output).cpu()
                if len(target.shape) > 3:
                    target = nn.functional.adaptive_avg_pool2d(target, (224, 224))
                targets.append(target)
                outputs.append(output)
            targets, outputs = torch.cat(targets), torch.cat(outputs)
            loss = lf(outputs.view(*targets.shape), targets)
            losses.append(loss.item())
            if WANDB:
                if len(targets.shape) > 3:
                    wandb.log({'examples': [wandb.Image(img) for img in outputs[:10]]}, commit=False)
                    wandb.log({'originals': [wandb.Image(img) for img in targets[:10]]}, commit=False)
    return losses


def spatial_mse(x, y):
    return nn.MSELoss(reduction='sum')(x, y) / (x.shape[-3] * x.shape[-2] * x.shape[-1])

if __name__ == '__main__':
    main()
