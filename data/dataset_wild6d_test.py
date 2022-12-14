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
import os
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
import pickle as pkl



class Wild6DTestDataset(Dataset):

    def __init__(self, opts):
        self.opts = opts

        with open(opts.test_list) as f:
            self.test_list = f.read().strip().split()
        self.imglist = []
        self.masklist = []
        self.depthlist = []
        self.metalist = []


        assert(opts.test == True)

        self.rot_gt_list = []
        self.trans_gt_list = []
        self.scale_gt_list = []


        self.total_length = 0

        for list_idx, seqname in enumerate(self.test_list):
            # print('processing:', seqname)
            seqname = seqname.split('_')
            obj_index, seq_index = eval(seqname[-2]), eval(seqname[-1])

            obj_list = sorted(os.listdir(opts.test_dataset_path))
            seq_list = sorted(os.listdir(os.path.join(opts.test_dataset_path, obj_list[obj_index])))
            img_list_total = glob.glob(os.path.join(opts.test_dataset_path, obj_list[obj_index], seq_list[seq_index], 'images/*.jpg'))
            
            mask_list = glob.glob(os.path.join(opts.test_dataset_path, obj_list[obj_index], seq_list[seq_index], 'images/*-mask.png'))
            mask_list.sort(key=lambda item: int(item.split('/')[-1].split('-')[0]))

            img_list = [i.replace('-mask.png', '.jpg') for i in mask_list]
            depth_list = [i.replace('-mask.png', '-depth.png') for i in mask_list]
            self.imglist.append(img_list)
            self.masklist.append(mask_list)
            self.depthlist.append(depth_list)

            meta_path = os.path.join(opts.test_dataset_path, obj_list[obj_index], seq_list[seq_index], 'metadata')
            metadata = json.load(open(meta_path, 'rb'))

            K = np.array(metadata['K']).reshape(3, 3).T if 'K' in metadata.keys() else None # first x, then y
            w = metadata['w'] if 'w' in metadata.keys() else None
            h = metadata['h'] if 'h' in metadata.keys() else None
            fps = metadata['fps'] if 'fps' in metadata.keys() else None
            self.metalist.append((K, w, h, fps))

            self.total_length += len(mask_list)


            self.rot_gt_list.append([])
            self.trans_gt_list.append([])
            self.scale_gt_list.append([])
            if not opts.eval: continue
            prefix = opts.test_dataset_path.rfind("test_set") + 9
            class_name = opts.test_dataset_path[prefix:-1]
            gt_path = opts.test_dataset_path[:prefix] + 'pkl_annotations/' + \
                    opts.test_dataset_path[prefix:] + '{}-{}-{}.pkl'.format(class_name, obj_list[obj_index], seq_list[seq_index])
            with open(gt_path, 'rb') as f:
                gt_data = pkl.load(f)
            for frame_id_enum, gt_anno in enumerate(gt_data['annotations']):
                cls_n, obj_idx, seq_idx, frame_idx = gt_anno['name'].split('/')
                frame_id = int(frame_idx)
                assert(frame_id_enum == frame_id)
                try:
                    rotation = gt_data['annotations'][frame_id]['rotation']
                except:
                    print('gt data not found!', gt_path, frame_id)
                    raise Exception
                translation = gt_data['annotations'][frame_id]['translation']
                size = gt_data['annotations'][frame_id]['size']
                rotation = np.array(rotation)
                translation = np.array(translation)
                size = np.array(size)

                self.rot_gt_list[list_idx].append(rotation)
                self.trans_gt_list[list_idx].append(translation)
                self.scale_gt_list[list_idx].append(size)

        self.transform = transforms.ToTensor()
        self.dframe = self.opts.dframe_eval
        total_samples = []
        for video_idx in range(len(self.masklist)):  # relative video id
            for i in range(0, len(self.masklist[video_idx]), self.dframe):
                total_samples.append((video_idx, i))  # n_videos * n_frames, 1

        self.sample_list = total_samples
        self.total_length = len(total_samples)
        print('total number of frames:', self.total_length)

    
    def __len__(self):
        return self.total_length
            

    def __getitem__(self, index):

        video_id, frame_id = self.sample_list[index]

        rand_scale = np.array([1.35, 1.35])

        ## start reading
        img = cv2.imread(self.imglist[video_id][frame_id])[:,:,::-1]
        mask = cv2.imread(self.masklist[video_id][frame_id], cv2.IMREAD_GRAYSCALE)
        intr = self.metalist[video_id][0]
        if self.opts.use_depth:
            depth = cv2.imread(self.depthlist[video_id][frame_id], -1) * 1.0

        mask = mask.astype(bool)

        img = img * 1.0

        # crop box
        indices = np.where(mask > 0)
        xid = indices[1]
        yid = indices[0]
        center  = [(xid.max() + xid.min()) // 2, (yid.max() + yid.min()) // 2]
        length  = [(xid.max() - xid.min()) // 2, (yid.max() - yid.min()) // 2]
        length  = [int(rand_scale[0] * length[0]), int(rand_scale[1] * length[1])]
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
            'frame_idx': torch.tensor([frame_id]),
        }

        if self.opts.eval:
            rotation = self.rot_gt_list[video_id][frame_id]
            translation = self.trans_gt_list[video_id][frame_id]
            scale = self.scale_gt_list[video_id][frame_id]
            elem['rotation'] = torch.tensor(rotation)
            elem['translation'] = torch.tensor(translation)
            elem['scale'] = torch.tensor(scale)

        return elem



        
        

