import click
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import dnnlib
import legacy
from training.dataset_encoder import InferenceDataset
from training.networks_encoder import Encoder
from training.training_loop_encoder import save_image
from gen_images import make_transform


@click.command()
@click.option('--encoder_pkl', type=str, required=True, help='pkl file for encoder network')
@click.option('--generator_pkl', type=str, help='pkl file for stylegan3 network')
@click.option('--srcdir', type=str, help='input directory')
@click.option('--outdir', type=str, default='results', help='output directory', show_default=True)
@click.option('--num_gpus', type=int, default=1, help='number of GPUS to use', show_default=True )
@click.option('--save_generated', type=bool, default=False, help='save generated images', show_default=True)
@click.option('--image_size', type=int, default=512, help='output image size', show_default=True)
@click.option('-b', '--batch_size', type=int, default=1, help='batch size', show_default=True)
def main(
    encoder_pkl: str,
    generator_pkl: str,
    srcdir: str,
    outdir: str,
    num_gpus: int,
    save_generated: bool,
    image_size: int,
    batch_size: int,
):
    out_dir = Path(outdir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    latents_dir = out_dir / 'latents'
    latents_dir.mkdir(parents=True, exist_ok=True)
    with dnnlib.util.open_url(generator_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema']
    latent_avg = G.mapping.w_avg

    E = Encoder(pretrained=encoder_pkl, w_avg=latent_avg)
    if num_gpus > 1:
        print("using DataParallel")
        E = torch.nn.DataParallel(E, device_ids=list(range(num_gpus))).cuda()
    else:
        E = E.cuda()
    E.eval()
    
    if save_generated:
        generated_dir = out_dir / 'gen'
        generated_dir.mkdir(parents=True, exist_ok=True)
        
        
        if num_gpus > 1:
            G = torch.nn.DataParallel(G, device_ids=list(range(num_gpus))).cuda()
        else:
            G = G.cuda()
        G.eval()

    infer_set = InferenceDataset(srcdir)
    infer_loader = torch.utils.data.DataLoader(dataset=infer_set, batch_size=batch_size, num_workers=16)

    with torch.no_grad():
        for X, stems in tqdm(infer_loader, total=len(infer_loader)):
            X = X.cuda()
            w = E(X)
            if save_generated:
                synth = G.synthesis(w)
                for stem in stems:
                    save_image(synth, str(generated_dir / (stem + '.png')), image_size, image_size)
            w = w.cpu().numpy()
            for i,stem in enumerate(stems):
                np.save(str(latents_dir / (stem + '.npy')), w[i])


if __name__ == '__main__':
    main()