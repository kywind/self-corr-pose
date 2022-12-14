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
import pickle as pkl
import time


class NOCSDataset(Dataset):

    def __init__(self, opts):
        self.opts = opts

        with open(opts.train_list) as f:
            self.train_list = f.read().strip().split()
        self.imglist = []
        self.masklist = []
        self.depthlist = []
        self.metalist = []

        self.category_id_list = {'bottle': 1, 'bowl': 2, 'camera': 3, 'can': 4, 'laptop': 5, 'mug': 6}

        self.total_frames = 0

        for list_idx, seqname in enumerate(self.train_list):
            scene_index = eval(seqname)
            scene_list = sorted(os.listdir(opts.dataset_path))
            
            mask_list_total = glob.glob(os.path.join(opts.dataset_path, scene_list[scene_index], '*_mask.png'))
            mask_list_total.sort(key=lambda item: int(item.split('/')[-1].split('_')[0]))
            
            frame_obj_id_dict = {}
            for frame in range(len(mask_list_total)):
                with open(mask_list_total[frame].replace('_mask.png', '_meta.txt')) as f:
                    lines = f.read().strip().split('\n')
                    for ln in lines:
                        ln = ln.split()
                        if eval(ln[1]) == self.category_id_list[opts.category]:
                            if ln[2] not in frame_obj_id_dict.keys():
                                frame_obj_id_dict[ln[2]] = []
                            for item in frame_obj_id_dict:
                                if ln[2] == item:
                                    frame_obj_id_dict[ln[2]].append((frame, eval(ln[0])))
                                    break
            
            for obj_name in frame_obj_id_dict.keys():
                mask_list = []
                meta_list = []

                frame_obj_id_list = frame_obj_id_dict[obj_name]
                for frame, frame_obj_id in frame_obj_id_list:
                    mask_fn = mask_list_total[frame]
                    mask_list.append(mask_fn)
                    
                    meta_fn = mask_fn.replace('_mask.png', '_label.pkl')
                    with open(meta_fn, 'rb') as f:
                        data = pkl.load(f)
                        for iid in range(len(data['instance_ids'])):
                            if frame_obj_id == data['instance_ids'][iid]:

                                class_id = data['class_ids'][iid]
                                name = data['model_list'][iid]
                                assert class_id == self.category_id_list[opts.category]
                                assert name == obj_name

                                rotation = data['rotations'][iid]
                                translation = data['translations'][iid]
                                scale = data['scales'][iid]
                                bbox = data['bboxes'][iid]

                        meta = {'rotation': rotation, 
                                'translation': translation, 
                                'scale': scale,
                                'bbox': bbox,
                                'id': frame_obj_id}
                        meta_list.append(meta)

                img_list = [i.replace('_mask.png', '_color.png') for i in mask_list]
                depth_list = [i.replace('_mask.png', '_depth.png') for i in mask_list]

                self.imglist.append(img_list)
                self.masklist.append(mask_list)
                self.depthlist.append(depth_list)
                self.metalist.append(meta_list)

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
        rand_scale = np.random.uniform(1.1, 1.3, size=(2,))

        ## start reading
        img = cv2.imread(self.imglist[video_id][frame_id])[:,:,::-1]
        mask = cv2.imread(self.masklist[video_id][frame_id], cv2.IMREAD_GRAYSCALE)
        if self.opts.use_depth:
            depth = cv2.imread(self.depthlist[video_id][frame_id], -1) * 1.0

        meta = self.metalist[video_id][frame_id]
        bbox = meta['bbox']
        frame_obj_id = meta['id']
        if self.opts.use_occ: occ = ((mask != frame_obj_id) * (mask != 255)).astype(bool)
        mask = (mask == frame_obj_id).astype(bool)

        img = img * 1.0

        # crop box
        center = [int((bbox[1] + bbox[3]) / 2), int((bbox[0] + bbox[2]) / 2)]
        length = [int((bbox[3] - bbox[1]) / 2), int((bbox[2] - bbox[0]) / 2)]
        max_length = max(length[0], length[1])

        if self.opts.no_stretch: length = [int(rand_scale[0] * max_length), int(rand_scale[0] * max_length)]
        else: length = [int(rand_scale[0] * length[0]), int(rand_scale[1] * length[1])]
        foc = [int(591.0125), int(590.16775)]
        pp = [int(322.525), int(244.11084)]

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
        
        if self.opts.use_occ:
            occ = torch.tensor(occ, dtype=torch.float32)[None]
            occ = transforms.functional.resized_crop(occ, center[1]-length[1], center[0]-length[0], 2*length[1], 2*length[0], \
                    size=(maxh, maxw), interpolation=InterpolationMode.NEAREST)

        if self.opts.use_depth:
            depth = torch.tensor(depth, dtype=torch.float32)[None]
            depth = transforms.functional.resized_crop(depth, center[1]-length[1], center[0]-length[0], 2*length[1], 2*length[0], \
                        size=(maxh, maxw), interpolation=InterpolationMode.NEAREST)

        elem = {
            'img': img,
            'mask': mask,
            'depth': depth if self.opts.use_depth else torch.zeros(1),
            'occ': occ if self.opts.use_occ else torch.zeros(1),
            'center': torch.tensor(center),
            'length': torch.tensor(length),
            'foc': torch.tensor(foc),
            'foc_crop': torch.tensor(foc_crop),
            'pp': torch.tensor(pp),
            'pp_crop': torch.tensor(pp_crop),
            'idx': torch.tensor([video_id]),
            'frame_idx': torch.tensor([frame_id]),
        }

        return elem

