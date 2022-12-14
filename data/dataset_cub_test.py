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
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import json
import time
import scipy.io as sio

import data.ucmr.geom_utils as geom_utils
import data.ucmr.image_utils as image_utils
import data.ucmr.transformations as transformations


def get_campose_dict(campose_path, is_campose):
    '''
    Returns campose dict: {frame_id: (cams_px7, scores_p, gt_st_3)}
    '''
    if is_campose:
        try:
            x = np.load(campose_path, allow_pickle=True)
            campose_dict = x['campose'].item()
        except UnicodeError:
            x = np.load(campose_path, allow_pickle=True, encoding='bytes')
            campose_dict = x['campose'].item()
    else:
        x = np.load(campose_path, allow_pickle=True)
        campose_dict = {}
        gt_cam_nx7 = torch.as_tensor(x['gt_cam'])
        cams_nxpx7 = torch.as_tensor(x['cam_values'])
        score_nxp = torch.as_tensor(x['quat_scores'])
        fids_nx1 = torch.as_tensor(x['frame_id'])

        gt_cam_flip_nx7 = geom_utils.reflect_cam_pose(gt_cam_nx7)
        cams_flip_nxpx7 = geom_utils.reflect_cam_pose(cams_nxpx7)
        fids_flip_nx1 = int(1e6) - fids_nx1

        flip = fids_nx1>int(1e6)/2
        gt_cam_nx7 = torch.where(flip, gt_cam_flip_nx7, gt_cam_nx7)
        cams_nxpx7 = torch.where(flip[:,:,None], cams_flip_nxpx7, cams_nxpx7)
        fids_nx1 = torch.where(flip, fids_flip_nx1, fids_nx1)

        assert((fids_nx1>=0).all())
        assert((fids_nx1<int(1e6)/2).all())

        for i in range(fids_nx1.shape[0]):
            fid = int(fids_nx1[i,0])
            gt_st_3 = gt_cam_nx7[i,0:3]
            cams_px7 = cams_nxpx7[i,:]
            score_p = score_nxp[i,:]
            campose_dict[fid] = (cams_px7, score_p, gt_st_3)

    return campose_dict


class CUBTestDataset(Dataset):

    def __init__(self, opts):
        self.opts = opts
        self.img_size = opts.img_size

        self.flip = False  # TODO
        self.flip_train = False
        self.dataloader_computeMaskDt = False  # TODO
        self.use_cameraPoseDict_as_gt = False
        self.cameraPoseDict_dataloader_isCamPose = True  # depends on use_cameraPoseDict_as_gt == True
        self.cameraPoseDict_dataloader = ''
        self.cameraPoseDict_dataloader_mergewith = ''
        self.data_dir = opts.dataset_path
        self.data_cache_dir = opts.dataset_cache_path
        self.jitter_frac = 0.05
        self.padding_frac = 0.2
        self.tight_crop = False
        self.split = 'test'
        self.num_kps = 15
        self.number_pairs = 10000
        self.rngFlip = np.random.RandomState(0)
        self.flip_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(1),
            transforms.ToTensor()
        ])

        if self.use_cameraPoseDict_as_gt:
            self.cameraPoseDict = get_campose_dict(self.cameraPoseDict_dataloader, self.cameraPoseDict_dataloader_isCamPose)
            print(f'Loaded cam_pose_dict of size {len(self.cameraPoseDict)} (should be 5964) in dataloader')
            if self.cameraPoseDict_dataloader_mergewith:
                for _ll in self.cameraPoseDict_dataloader_mergewith:
                    cameraPoseDict2 = get_campose_dict(_ll, self.cameraPoseDict_dataloader_isCamPose)
                    self.cameraPoseDict = {**cameraPoseDict2, **self.cameraPoseDict}
                    print(f'Merged cam_pose_dict of size {len(self.cameraPoseDict)} (should be 5964) in dataloader')

        # self.opts = opts
        self.img_dir = os.path.join(self.data_dir, 'images')
        self.anno_path = os.path.join(self.data_cache_dir, 'data', '%s_cub_cleaned.mat' % self.split)
        self.anno_sfm_path = os.path.join(self.data_cache_dir, 'sfm', 'anno_%s.mat' % self.split)
        self.anno_train_sfm_path = os.path.join(self.data_cache_dir, 'sfm', 'anno_%s.mat' % 'train')

        if not os.path.exists(self.anno_path):
            print('%s doesnt exist!' % self.anno_path)
            import ipdb
            ipdb.set_trace()

        # Load the annotation file.
        print('loading %s' % self.anno_path)
        self.anno = sio.loadmat(self.anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.anno_sfm = sio.loadmat(self.anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']
        self.kp3d = sio.loadmat(self.anno_train_sfm_path, struct_as_record=False, squeeze_me=True)['S'].transpose().copy()
        self.num_imgs = len(self.anno)
        print('total %d images' % self.num_imgs)


        with open(os.path.join(self.data_dir, 'classes.txt')) as f:
            class_name_data = f.read().strip().split()
        class_name_dict = {}
        for i in range(len(class_name_data)//2):
            class_name_dict[class_name_data[2*i+1]] = int(class_name_data[2*i])
        class_id_list = [0] * self.num_imgs
        img_id_list = [0] * self.num_imgs
        class_id_list_inv = [[] for i in range(len(class_name_data)//2)]
        for index in range(self.num_imgs):
            rel_path = self.anno[index].rel_path
            class_name, img_name = rel_path.split('/')[0], rel_path.split('/')[1]
            class_id_list[index] = class_name_dict[class_name] - 1
            img_id_list[index] = len(class_id_list_inv[class_id_list[index]])
            class_id_list_inv[class_id_list[index]].append(index)
        
        # for index in range(self.num_imgs):
        #     rel_path = self.anno[index].rel_path
        #     print(rel_path)
        # import pdb; pdb.set_trace()
        
        with open(opts.test_list) as f:
            self.test_list = f.read().strip().split()
        self.index_list = []
        self.class_id_list = []
        self.img_id_list = []
        self.class_id_list_inv = []
        for rel_index, index in enumerate(self.test_list):
            self.index_list.extend(class_id_list_inv[int(index)])
            self.class_id_list.extend([rel_index] * len(class_id_list_inv[int(index)]))
            self.img_id_list.extend(list(range(len(class_id_list_inv[int(index)]))))
            self.class_id_list_inv.append(class_id_list_inv[int(index)])

        self.imglist = [
            [os.path.join(self.img_dir, str(self.anno[i].rel_path)) for i in self.class_id_list_inv[j]]
            for j in range(len(self.test_list))
        ]

        print('using %d classes, %d images' % (len(self.test_list), len(self.index_list)))
        
        self.kp_perm = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15]) - 1
        # self.kp_names = ['Back', 'Beak', 'Belly', 'Breast', 'Crown', 'FHead', 'LEye',
        #                  'LLeg', 'LWing', 'Nape', 'REye', 'RLeg', 'RWing', 'Tail', 'Throat']
        # self.mean_shape = sio.loadmat(osp.join(cub_cache_dir, 'uv', 'mean_shape.mat'))
        # self.kp_uv = self.preprocess_to_find_kp_uv(self.kp3d, self.mean_shape['faces'], self.mean_shape[
        #                                            'verts'], self.mean_shape['sphere_verts'])

        self.dframe = self.opts.dframe_eval
        total_samples = []
        for video_idx in range(len(self.test_list)):  # relative video id
            for i in range(0, len(self.class_id_list_inv[video_idx]), self.dframe):
                total_samples.append((video_idx, i))  # n_videos * n_frames, 1

        self.sample_list = total_samples
        self.total_length = len(total_samples)

    def __len__(self):
        return self.total_length

    # def preprocess_to_find_kp_uv(self, kp3d, faces, verts, verts_sphere, ):
    #     mesh = pymesh.form_mesh(verts, faces)
    #     dist, face_ind, closest_pts = pymesh.distance_to_mesh(mesh, kp3d)
    #     dist_to_verts = np.square(kp3d[:, None, :] - verts[None, :, :]).sum(-1)
    #     closest_pts = closest_pts / np.linalg.norm(closest_pts, axis=1, keepdims=1)
    #     min_inds = np.argmin(dist_to_verts, axis=1)
    #     kp_verts_sphere = verts_sphere[min_inds]
    #     kp_uv = geom_utils.convert_3d_to_uv_coordinates(closest_pts)
    #     return kp_uv

    def get_anno(self, index):
        data = self.anno[index]
        data_sfm = self.anno_sfm[index]
        # sfm_pose = (sfm_c, sfm_t, sfm_r)
        sfm_pose = [np.copy(data_sfm.scale), np.copy(data_sfm.trans), np.copy(data_sfm.rot)]
        sfm_rot = np.pad(sfm_pose[2], (0, 1), 'constant')

        sfm_rot[3, 3] = 1
        sfm_pose[2] = transformations.quaternion_from_matrix(sfm_rot, isprecise=True)

        img_path = os.path.join(self.img_dir, str(data.rel_path))

        # Adjust to 0 indexing
        bbox = np.array(
            [data.bbox.x1, data.bbox.y1, data.bbox.x2, data.bbox.y2],
            float) - 1

        parts = data.parts.T.astype(float)
        kp = np.copy(parts)
        vis = kp[:, 2] > 0
        kp[vis, :2] -= 1

        return img_path, data.mask, bbox, sfm_pose, kp, vis

    def forward_img(self, index):

        img_path, mask, bbox, sfm_pose, kp, vis = self.get_anno(index)

        img = cv2.imread(img_path)[:,:,::-1]
        
        if img is None:
            raise FileNotFoundError(img_path)
        img = img / 255.0

        # Some are grayscale:
        if len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
        assert(img.shape[:2] == mask.shape)
        mask = np.expand_dims(mask, 2)

        # Peturb bbox
        if self.tight_crop:
            self.padding_frac = 0.0

        if self.split == 'train':
            bbox = image_utils.peturb_bbox(
                bbox, pf=self.padding_frac, jf=self.jitter_frac)
        else:
            bbox = image_utils.peturb_bbox(
                bbox, pf=self.padding_frac, jf=0)
        if self.tight_crop:
            bbox = bbox
        else:
            bbox = image_utils.square_bbox(bbox)

        xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        center = [(xmin + xmax) / 2, (ymin + ymax) / 2]
        length = [(xmax - xmin) / 2, (ymax - ymin) / 2]
        H = img.shape[0]
        W = img.shape[1]
        fx = fy = max(H, W) * 2
        px = int(W / 2)
        py = int(H / 2)
        foc = [fx, fy]
        pp = [px, py]

        crop_factor  = [self.opts.img_size / 2 / length[0], self.opts.img_size / 2 / length[1]]
        foc_crop = [foc[0] * crop_factor[0], foc[1] * crop_factor[1]]
        pp_crop = [(pp[0]-(center[0]-length[0])) * crop_factor[0], (pp[1]-(center[1]-length[1])) * crop_factor[1]]

        intrinsics = (foc, pp, foc_crop, pp_crop, center, length)

        # crop image around bbox, translate kps
        img, mask, kp, sfm_pose = self.crop_image(img, mask, bbox, kp, vis, sfm_pose)
        # scale image, and mask. And scale kps.
        if self.tight_crop:
            img, mask, kp, sfm_pose = self.scale_image_tight(img, mask, kp, vis, sfm_pose)
        else:
            img, mask, kp, sfm_pose = self.scale_image(img, mask, kp, vis, sfm_pose)


        # Mirror image on random.
        if self.split == 'train':
            flipped, img, mask, kp, sfm_pose = self.mirror_image(img, mask, kp, sfm_pose)
        else:
            flipped = False

        # Normalize kp to be [-1, 1]
        img_h, img_w = img.shape[:2]
        kp_norm, sfm_pose = self.normalize_kp(kp, sfm_pose, img_h, img_w)

        # Finally transpose the image to 3xHxW
        img = np.transpose(img, (2, 0, 1))

        return flipped, img, kp_norm, mask, sfm_pose, bbox, intrinsics, img_path

    def normalize_kp(self, kp, sfm_pose, img_h, img_w):
        if kp is not None:
            vis = kp[:, 2, None] > 0
            kp = np.stack([2 * (kp[:, 0] / img_w) - 1,
                            2 * (kp[:, 1] / img_h) - 1,
                            kp[:, 2]]).T
            kp = vis * kp
        sfm_pose[0] *= (1.0 / img_w + 1.0 / img_h)
        sfm_pose[1][0] = 2.0 * (sfm_pose[1][0] / img_w) - 1
        sfm_pose[1][1] = 2.0 * (sfm_pose[1][1] / img_h) - 1

        return kp, sfm_pose

    def crop_image(self, img, mask, bbox, kp, vis, sfm_pose):
        # crop image and mask and translate kps
        img = image_utils.crop(img, bbox, bgval=0)
        mask = image_utils.crop(mask, bbox, bgval=0)
        if kp is not None:
            assert(vis is not None)
            kp[vis, 0] -= bbox[0]
            kp[vis, 1] -= bbox[1]

            kp[vis,0] = np.clip(kp[vis,0], a_min=0, a_max=bbox[2] -bbox[0])
            kp[vis,1] = np.clip(kp[vis,1], a_min=0, a_max=bbox[3] -bbox[1])

        sfm_pose[1][0] -= bbox[0]
        sfm_pose[1][1] -= bbox[1]

        return img, mask, kp, sfm_pose

    def scale_image_tight(self, img, mask, kp, vis, sfm_pose):
        # Scale image so largest bbox size is img_size
        bwidth = np.shape(img)[1]
        bheight = np.shape(img)[0]

        scale_x = self.img_size/bwidth
        scale_y = self.img_size/bheight

        # scale = self.img_size / float(max(bwidth, bheight))
        # pdb.set_trace()
        img_scale = cv2.resize(img, (self.img_size, self.img_size))
        # img_scale, _ = image_utils.resize_img(img, scale)
        # if img_scale.shape[0] != self.img_size:
        #     print('bad!')
        #     import ipdb; ipdb.set_trace()
        # mask_scale, _ = image_utils.resize_img(mask, scale)

        mask_scale = cv2.resize(mask, (self.img_size, self.img_size))

        if kp is not None:
            assert(vis is not None)
            kp[vis, 0:1] *= scale_x
            kp[vis, 1:2] *= scale_y
        sfm_pose[0] *= scale_x  # TODO might be a bug
        sfm_pose[1] *= scale_y

        return img_scale, mask_scale, kp, sfm_pose

    def scale_image(self, img, mask, kp, vis, sfm_pose):
        # Scale image so largest bbox size is img_size
        bwidth = np.shape(img)[0]
        bheight = np.shape(img)[1]
        scale = self.img_size / float(max(bwidth, bheight))
        img_scale, _ = image_utils.resize_img(img, scale)
        # if img_scale.shape[0] != self.img_size:
        #     print('bad!')
        #     import ipdb; ipdb.set_trace()
        mask_scale, _ = image_utils.resize_img(mask, scale)
        if kp is not None:
            assert(vis is not None)
            kp[vis, :2] *= scale
        sfm_pose[0] *= scale
        sfm_pose[1] *= scale

        return img_scale, mask_scale, kp, sfm_pose

    def mirror_image(self, img, mask, kp, sfm_pose):
        if self.rngFlip.rand(1) > 0.5 and self.flip:
            # Need copy bc torch collate doesnt like neg strides
            img_flip = img[:, ::-1, :].copy()
            mask_flip = mask[:, ::-1].copy()

            if kp is not None:
                # Flip kps.
                new_x = img.shape[1] - kp[:, 0] - 1
                kp = np.hstack((new_x[:, None], kp[:, 1:]))
                kp = kp[self.kp_perm, :]
                # kp_uv_flip = kp_uv[self.kp_perm, :]
            # Flip sfm_pose Rot.
            R = transformations.quaternion_matrix(sfm_pose[2])
            flip_R = np.diag([-1, 1, 1, 1]).dot(R.dot(np.diag([-1, 1, 1, 1])))
            sfm_pose[2] = transformations.quaternion_from_matrix(flip_R, isprecise=True)
            # Flip tx
            tx = img.shape[1] - sfm_pose[1][0] - 1
            sfm_pose[1][0] = tx
            return True, img_flip, mask_flip, kp, sfm_pose
        else:
            return False, img, mask, kp, sfm_pose

    def __getitem__(self, raw_index):
        class_id, img_id = self.sample_list[raw_index]
        index = self.class_id_list_inv[class_id][img_id]
        flipped, img, kp, mask, sfm_pose, bbox, intrinsics, img_path = self.forward_img(index)
        foc, pp, foc_crop, pp_crop, center, length = intrinsics
        sfm_pose[0].shape = 1
        elem = {
            'img': torch.tensor(img),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'idx': torch.tensor([class_id]),
            'frame_idx': torch.tensor([img_id]),
            'global_idx': torch.tensor([index]),
            'foc': torch.tensor(foc),
            'pp': torch.tensor(pp),
            'foc_crop': torch.tensor(foc_crop),
            'pp_crop': torch.tensor(pp_crop),
            'center': torch.tensor(center),
            'length': torch.tensor(length),
            'kp': torch.tensor(kp),
            'sfm_pose': torch.tensor(np.concatenate(sfm_pose)),  # scale (1), trans (2), quat(4)
            # 'kp_uv': kp_uv,
            # 'anchor': anchor,
            # 'pos_inds': positive_samples,
            # 'neg_inds': negative_samples,
        }
        return elem


def collate_fn(batch):
    '''Globe data collater.

    Assumes each instance is a dict.
    Applies different collation rules for each field.

    Args:
        batch: List of loaded elements via Dataset.__getitem__
    '''
    collated_batch = {'empty': True}
    # iterate over keys
    # new_batch = []
    # for valid,t in batch:
    #     if valid:
    #         new_batch.append(t)
    #     else:
    #         'Print, found a empty in the batch'

    # # batch = [t for t in batch if t is not None]
    # # pdb.set_trace()
    # batch = new_batch
    if len(batch) > 0:
        for key in batch[0]:
            collated_batch[key] = default_collate([elem[key] for elem in batch])
        collated_batch['empty'] = False
    return collated_batch

