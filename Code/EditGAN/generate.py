# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import numpy as np
import torch
import random
torch.manual_seed(0)
import json

from tqdm import tqdm, trange
from PIL import Image

from utils.data_utils import *
from utils.model_utils import *

import argparse
import imageio

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(args, total, seed):
    np.random.seed(seed)
    g_all, _, upsamplers, _, avg_latent = prepare_model(args)
    outdir = Path(args['training_data_path'])
    with torch.no_grad():
        for i in trange(total):
            latent = np.random.randn(1, 512)
            latent = torch.from_numpy(latent).type(torch.FloatTensor).to(device)
            sample_img, sample_latnet = latent_to_image(
                    g_all, upsamplers, latent, return_only_im=True, process_out=True)
            # sample_img = (sample_img + 1.0) / 2.0
            latent_out = sample_latnet.detach().cpu().numpy()[0]
            np.save(str(outdir / 'latents_image_{}.npy'.format(i)), latent_out)
            # img_save = 255 * np.transpose((sample_img.detach().cpu().numpy()[0]), (1,2,0))
            # print(img_save.shape)
            imageio.imsave(str(outdir / 'image_{}.png'.format(i)), sample_img[0])
            # img_out = Image.fromarray(np.clip(img_save.astype(np.uint8), 0, 255))
            # img_out.save(str(outdir / 'image_{}.png'.format(i)))


        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str)
    parser.add_argument('--total', type=int, default=100)
    parser.add_argument('--seed', type=int, default=41)

    args = parser.parse_args()
    opts = json.load(open(args.exp, 'r'))

    main(opts, args.total, args.seed)

