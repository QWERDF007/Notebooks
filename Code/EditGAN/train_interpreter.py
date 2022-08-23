# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import sys
sys.path.append('..')
import imageio
import torch
import torch.nn as nn
torch.manual_seed(0)
import scipy.misc
import json
import numpy as np
device_ids = [0]
from PIL import Image
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
from utils.model_utils import *
from utils.data_utils import *
from models.DatasetGAN.classifer import pixel_classifier

import scipy.stats
import torch.optim as optim
import argparse
from utils.data_utils import face_palette as palette

device = 'cuda' if torch.cuda.is_available() else 'cpu'
import cv2
from tqdm import tqdm, trange


class trainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        print(self.X_data[index].shape, self.y_data[index].shape)
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class TrainData(Dataset):

    def __init__(self, args, palette) -> None:
        super().__init__()
        self.X_data, self.y_data, self._num_data = prepare_data(args, palette)
        self._num_classes = int(max(np.unique(self.y_data)))

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_data(self):
        return self._num_data

    def __getitem__(self, index):
        return torch.FloatTensor(self.X_data[index]), torch.FloatTensor(np.array(self.y_data[index]))
    
    def __len__(self):
        return len(self.X_data)


def prepare_data(args, palette):

    g_all, _, upsamplers, _, avg_latent = prepare_model(args)
    if args['optimized_latent_path']['train'][-4:] == ".npy":
        latent_all = np.load(args['optimized_latent_path']['train'])
    else:
        latent_all = []
        for i in range(args['max_training']):
            # quickly resolve id mismatch
            # if i >= 28:
            #     i += 1
            name = 'latents_image_%0d.npy' % i

            im_frame = np.load(os.path.join(args['optimized_latent_path']['train'], name))
            latent_all.append(im_frame)
        latent_all = np.array(latent_all)

    latent_all = torch.from_numpy(latent_all).cuda()


    # load annotated mask
    mask_list = []
    im_list = []
    latent_all = latent_all[:args['max_training']]
    num_data = len(latent_all)

    for i in trange(len(latent_all)):

        if i >= args['max_training']:
            break
        name = 'image_mask%0d.npy' % i

        im_frame = np.load(os.path.join( args['annotation_mask_path'] , name))
        mask = np.array(im_frame)
        mask = cv2.resize(np.squeeze(mask), dsize=(args['dim'][1], args['dim'][0]), interpolation=cv2.INTER_NEAREST)

        mask_list.append(mask)

        im_name = os.path.join( args['annotation_mask_path'], 'image_%d.png' % i)
        img = Image.open(im_name)
        img = img.resize((args['dim'][1], args['dim'][0]))

        im_list.append(np.array(img))

    # delete small annotation error
    for i in trange(len(mask_list)):  # clean up artifacts in the annotation, must do
        for target in range(1, 50):
            if (mask_list[i] == target).sum() < 30:
                mask_list[i][mask_list[i] == target] = 0


    all_mask = np.stack(mask_list)


    # 3. Generate ALL training data for training pixel classifier
    all_feature_maps_train = np.zeros((args['dim'][0] * args['dim'][1] * len(latent_all), args['dim'][2]), dtype=np.float16)
    all_mask_train = np.zeros((args['dim'][0] * args['dim'][1] * len(latent_all),), dtype=np.float16)


    vis = []
    for i in trange(len(latent_all) ):

        gc.collect()

        latent_input = latent_all[i].float()

        img, feature_maps = latent_to_image(g_all, upsamplers, latent_input.unsqueeze(0), dim=args['dim'][1],
                                            return_upsampled_layers=True, use_style_latents=args['annotation_data_from_w'])
        if args['dim'][0]  != args['dim'][1]:
            # only for car
            img = img[:, 64:448]
            feature_maps = feature_maps[:, :, 64:448]
        mask = all_mask[i:i + 1]
        feature_maps = feature_maps.permute(0, 2, 3, 1)

        feature_maps = feature_maps.reshape(-1, args['dim'][2])
        #new_mask =  np.squeeze(mask)

        mask = mask.reshape(-1)

        all_feature_maps_train[args['dim'][0] * args['dim'][1] * i: args['dim'][0] * args['dim'][1] * i + args['dim'][0] * args['dim'][1]] = feature_maps.cpu().detach().numpy().astype(np.float16)
        all_mask_train[args['dim'][0] * args['dim'][1] * i:args['dim'][0] * args['dim'][1] * i + args['dim'][0] * args['dim'][1]] = mask.astype(np.float16)

        # img_show =  cv2.resize(np.squeeze(img[0]), dsize=(args['dim'][1], args['dim'][1]), interpolation=cv2.INTER_NEAREST)

        # curr_vis = np.concatenate( [im_list[i], img_show, colorize_mask(new_mask, palette)], 0 )

        # vis.append( curr_vis )


    # vis = np.concatenate(vis, 1)
    # imageio.imsave(os.path.join(args['exp_dir'], "train_data.jpg"),
    #                   vis)

    return all_feature_maps_train, all_mask_train, num_data


def generate_data(args,  total, seed=41, start_step=0, vis=True):
    np.random.seed(seed)

    result_path = os.path.join(args['exp_dir'], 'vis')
    g_all, _, upsamplers, _, avg_latent = prepare_model(args)
    classifier_list = []
    for MODEL_NUMBER in trange(args['model_num']):
        classifier = pixel_classifier(numpy_class=(args['number_class'] + 1), dim=args['dim'][-1])
        checkpoint = torch.load(os.path.join(args['exp_dir'], 'model_' + str(MODEL_NUMBER) + '.pth'))
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier = classifier.to(device)
        classifier.eval()
        classifier_list.append(classifier)

    with torch.no_grad():
        for i in trange(total):
            latent = np.random.randn(1, 512)
            latent = torch.from_numpy(latent).type(torch.FloatTensor).to(device)
            img, affine_layers = latent_to_image(g_all, upsamplers, latent, dim=args['dim'][1], return_upsampled_layers=True)

            if args['dim'][0] != args['dim'][1]:
                img = img[:, 64:448]
            img = img[0]

            if args['dim'][0] != args['dim'][1]:
                affine_layers = affine_layers[:, :, 64:448]
            affine_layers = affine_layers[0]
            affine_layers = affine_layers.reshape(args['dim'][-1], -1).transpose(1, 0)

            seg_mode_ensemble = []
            for MODEL_NUMBER in range(args['model_num']):
                classifier = classifier_list[MODEL_NUMBER]
                img_seg = classifier(affine_layers)
                img_seg = img_seg.squeeze()

                img_seg_final = oht_to_scalar(img_seg)
                img_seg_final = img_seg_final.reshape(args['dim'][0], args['dim'][1], 1)
                img_seg_final = img_seg_final.cpu().detach().numpy()

                seg_mode_ensemble.append(img_seg_final)
            

            img_seg_final = np.concatenate(seg_mode_ensemble, axis=-1)
            img_seg_final = scipy.stats.mode(img_seg_final, 2)[0].reshape(args['dim'][0], args['dim'][1])
            del (affine_layers)
            if vis:
                color_mask = 0.7 * colorize_mask(img_seg_final, palette) + 0.3 * img

                image_label_name = os.path.join(result_path, "vis_" + str(i + start_step) + '.png')
                image_name = os.path.join(result_path, "vis_" + str(i + start_step) + '_image.png')

                imageio.imwrite(image_label_name,  color_mask.astype(np.uint8))
                imageio.imwrite(image_name, img.astype(np.uint8))
            else:
                image_label_name = os.path.join(result_path, 'label_' + str(i + start_step) + '.png')
                image_name = os.path.join(result_path,  str(i + start_step) + '.png')

                imageio.imwrite(image_label_name,  img_seg_final.astype(np.uint8))
                imageio.imwrite(image_name, img.astype(np.uint8))




def main(args):

    # all_feature_maps_train_all, all_mask_train_all, num_data = prepare_data(args, palette)
    # print(all_feature_maps_train_all.shape, all_mask_train_all.shape)
    # print(type(all_feature_maps_train_all), type(all_mask_train_all), type(num_data))


    # train_data = trainData(torch.FloatTensor(all_feature_maps_train_all),
                        #    torch.FloatTensor(all_mask_train_all))
    train_data = TrainData(args, palette)
    max_label = train_data.num_classes
    num_data = train_data.num_data

    # count_dict = get_label_stas(train_data)

    # max_label = max([*count_dict])
    print(" *********************** max_label " + str(max_label) + " ***********************", flush=True)


    print(" *********************** Current number data " + str(num_data) + " ***********************", flush=True)


    batch_size = args['batch_size']

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8)

    print(" *********************** Current dataloader length " +  str(len(train_loader)) + " ***********************", flush=True)

    for MODEL_NUMBER in range(args['model_num']):

        gc.collect()

        classifier = pixel_classifier(numpy_class=(max_label + 1), dim=args['dim'][-1])

        classifier.init_weights()

        classifier = nn.DataParallel(classifier, device_ids=device_ids).cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)
        classifier.train()


        iteration = 0
        break_count = 0
        best_loss = 10000000
        stop_sign = 0
        best_state_dict = None
        for epoch in range(100):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_batch = y_batch.type(torch.long)
                y_batch = y_batch.type(torch.long)

                optimizer.zero_grad()
                y_pred = classifier(X_batch)
                loss = criterion(y_pred, y_batch)

                loss.backward()
                optimizer.step()

                iteration += 1
                if iteration % 1000 == 0:
                    print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item(), flush=True)
                    gc.collect()


                # if iteration % 5000 == 0:
                #     model_path = os.path.join(args['exp_dir'],
                #                               'model_iter' +  str(iteration) + '_number_' + str(MODEL_NUMBER) + '.pth')
                #     print('Save checkpoint, Epoch : ', str(epoch), ' Path: ', model_path, flush=True)

                #     torch.save({'model_state_dict': classifier.state_dict()},
                #                model_path)
                               
                if loss.item() < best_loss:
                    best_state_dict = classifier.module.state_dict()

                if epoch > 3:
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        break_count = 0
                    else:
                        break_count += 1

                    if break_count > 50:
                        stop_sign = 1
                        print("*************** Break, Total iters,", iteration, ", at epoch", str(epoch), "***************", flush=True)
                        break

            if stop_sign == 1:
                break

        gc.collect()
        model_path = os.path.join(args['exp_dir'],
                                  'model_' + str(MODEL_NUMBER) + '.pth')
        MODEL_NUMBER += 1
        print('Epoch : ', str(epoch), 'best loss', best_loss, flush=True)
        print('save to:',model_path, flush=True)
        torch.save({'model_state_dict': best_state_dict},
                   model_path)
        gc.collect()

        gc.collect()
        torch.cuda.empty_cache()    # clear cache memory on GPU


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str, help='exp json file')
    parser.add_argument('--test', type=bool, default=False, help='test mode')
    parser.add_argument('--total', type=int, default=1000, help='number to generate')
    parser.add_argument('--seed', type=int, default=41, help='random seed')
    parser.add_argument('--start_step', type=int, default=0, help='start step for image name')


    args = parser.parse_args()

    opts = json.load(open(args.exp, 'r'))
    print("Opt", opts, flush=True)

    path =opts['exp_dir']
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path), flush=True)

    os.system('cp %s %s' % (args.exp, opts['exp_dir']))
    if args.test:
        generate_data(opts, args.total, args.seed, args.start_step, vis=False)
    else:
        main(opts)

