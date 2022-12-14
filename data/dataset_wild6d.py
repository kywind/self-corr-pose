# Copyright 2021 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os, re
import sys
sys.path.insert(0,'third-party')

import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import json
import time


class Wild6DDataset(Dataset):

    def __init__(self, opts):
        self.opts = opts

        with open(opts.train_list) as f:
            self.train_list = f.read().strip().split()
        self.imglist = []
        self.masklist = []
        self.depthlist = []
        self.metalist = []

        self.total_frames = 0

        for list_idx, seqname in enumerate(self.train_list):
            # print('processing:', seqname)
            seqname = seqname.split('_')
            obj_index, seq_index = eval(seqname[-2]), eval(seqname[-1])

            obj_list = sorted(os.listdir(opts.dataset_path))
            seq_list = sorted(os.listdir(os.path.join(opts.dataset_path, obj_list[obj_index])))
            img_list_total = glob.glob(os.path.join(opts.dataset_path, obj_list[obj_index], seq_list[seq_index], 'images/*.jpg'))
            
            mask_list = glob.glob(os.path.join(opts.dataset_path, obj_list[obj_index], seq_list[seq_index], 'images/*-mask.png'))
            mask_list.sort(key=lambda item: int(item.split('/')[-1].split('-')[0]))
            
            img_list = [i.replace('-mask.png', '.jpg') for i in mask_list]
            depth_list = [i.replace('-mask.png', '-depth.png') for i in mask_list]
            self.imglist.append(img_list)
            self.masklist.append(mask_list)
            self.depthlist.append(depth_list)

            meta_path = os.path.join(opts.dataset_path, obj_list[obj_index], seq_list[seq_index], 'metadata')
            metadata = json.load(open(meta_path, 'rb'))

            K = np.array(metadata['K']).reshape(3, 3).T if 'K' in metadata.keys() else None # first x, then y
            w = metadata['w'] if 'w' in metadata.keys() else None
            h = metadata['h'] if 'h' in metadata.keys() else None
            fps = metadata['fps'] if 'fps' in metadata.keys() else None
            self.metalist.append((K, w, h, fps))

            self.total_frames += len(mask_list)

        self.transform = transforms.ToTensor()
        
        print('')
        print('total number of videos:', len(self.train_list))
        print('total number of valid frames:', self.total_frames)
        print('per gpu batch size:', opts.batch_size * opts.repeat)
        print('number of gpus:', opts.ngpu)
        print('number of iterations per gpu', opts.total_iters)
        print('')

        self.samples_per_iter =  opts.batch_size * opts.repeat * opts.ngpu
        self.samples_total = opts.total_iters * self.samples_per_iter
        self.sample_list = None
        self.reset()

    
    def __len__(self):
        return self.samples_total


    def reset(self): # create sampler
        total_samples = []
        for batch_idx in range(self.opts.total_iters):
            n_videos = len(self.masklist)
            video_samples = np.random.randint(0, n_videos, size=(self.opts.batch_size,))
            frame_samples = []
            for video_sample_idx in range(video_samples.shape[0]):
                n_frames = len(self.masklist[video_samples[video_sample_idx]])
                n_gap = n_frames // self.opts.repeat
                for i in range(self.opts.repeat):
                    for _ in range(self.opts.ngpu):
                        frame_samples.append((video_samples[video_sample_idx], n_gap * i + np.random.randint(0, n_gap)))
            total_samples.append(frame_samples)
        self.sample_list = total_samples
            

    def __getitem__(self, index):
        batch_id = index // self.samples_per_iter
        item_id = index % self.samples_per_iter
        video_id, frame_id = self.sample_list[batch_id][item_id]

        # random factors that needs to be consistent over repeats
        # only random in training
        rand_scale = np.random.uniform(1.2, 1.5, size=(2,))

        ## start reading
        img = cv2.imread(self.imglist[video_id][frame_id])[:,:,::-1]
        mask = cv2.imread(self.masklist[video_id][frame_id], cv2.IMREAD_GRAYSCALE)
        intr = self.metalist[video_id][0]
        if self.opts.use_depth:
            depth = cv2.imread(self.depthlist[video_id][frame_id], -1) * 1.0

        mask = mask.astype(bool)

        # complement color 
        img = img * 1.0

        # crop box
        indices = np.where(mask > 0)
        xid = indices[1]
        yid = indices[0]
        center = [(xid.max() + xid.min()) // 2, (yid.max() + yid.min()) // 2]
        length = [(xid.max() - xid.min()) // 2, (yid.max() - yid.min()) // 2]
        max_length = max(length[0], length[1])
        if self.opts.no_stretch: length = [int(rand_scale[0] * max_length), int(rand_scale[0] * max_length)]
        else: length = [int(rand_scale[0] * length[0]), int(rand_scale[1] * length[1])]
        foc = [intr[0, 0], intr[1, 1]]
        pp = [intr[0, 2], intr[1, 2]]

        maxw = self.opts.img_size
        maxh = self.opts.img_size
        crop_factor  = [maxw / 2 / length[0], maxh / 2 / length[1]]
        foc_crop = [foc[0] * crop_factor[0], foc[1] * crop_factor[1]]
        pp_crop = [(pp[0]-(center[0]-length[0])) * crop_factor[0], (pp[1]-(center[1]-length[1])) * crop_factor[1]]

        img = self.transform(img) / 255.
        mask = torch.tensor(mask, dtype=torch.float32)[None]

        
        img = transforms.functional.resized_crop(img, center[1]-length[1], center[0]-length[0], 2*length[1], 2*length[0], \
                    size=(maxh, maxw), interpolation=InterpolationMode.BILINEAR)
        mask = transforms.functional.resized_crop(mask, center[1]-length[1], center[0]-length[0], 2*length[1], 2*length[0], \
                    size=(maxh, maxw), interpolation=InterpolationMode.NEAREST)

        if self.opts.use_depth:
            depth = torch.tensor(depth, dtype=torch.float32)[None]
            depth = transforms.functional.resized_crop(depth, center[1]-length[1], center[0]-length[0], 2*length[1], 2*length[0], \
                        size=(maxh, maxw), interpolation=InterpolationMode.NEAREST)

        elem = {
            'img': img,
            'mask': mask,
            'depth': depth if self.opts.use_depth else torch.zeros(1),
            'center': torch.tensor(center),
            'length': torch.tensor(length),
            'foc': torch.tensor(foc),
            'foc_crop': torch.tensor(foc_crop),
            'pp': torch.tensor(pp),
            'pp_crop': torch.tensor(pp_crop),
            'idx': torch.tensor([video_id]),
            'frame_idx': torch.tensor([frame_id])
        }

        return elem

