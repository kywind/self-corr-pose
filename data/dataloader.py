from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import torch
import os

from data.dataset_wild6d import Wild6DDataset
from data.dataset_wild6d_test import Wild6DTestDataset
from data.dataset_cub import CUBDataset
from data.dataset_cub_test import CUBTestDataset
from data.dataset_nocs import NOCSDataset
from data.dataset_nocs_test import NOCSTestDataset
from torch.utils.data import DataLoader


flags.DEFINE_integer('img_size', 256, 'image size')
flags.DEFINE_integer('repeat', 8, 'number of frames sampled from a video each iteration')
flags.DEFINE_bool('shuffle_test', False, 'shuffle data order when testing')
flags.DEFINE_bool('no_stretch', False, 'do not stretch image to fit input size')
flags.DEFINE_bool('use_occ', False, 'use occlusion')

flags.DEFINE_string('dataset_path', 'data', 'the data dir')
flags.DEFINE_string('dataset_cache_path', 'data', 'the data cache dir')
flags.DEFINE_string('test_dataset_path', 'data', 'the test data dir')

flags.DEFINE_string('dataset_name', 'Wild6D', 'name of the dataset')
flags.DEFINE_string('category', 'bottle', '')


def get_dataset(opts, training):
    if opts.dataset_name == 'Wild6D':
        if training:
            dataset = Wild6DDataset(opts)
        else:
            dataset = Wild6DTestDataset(opts)
        return dataset
    elif opts.dataset_name == 'cub':
        if training:
            dataset = CUBDataset(opts)
        else:
            dataset = CUBTestDataset(opts)
        return dataset
    elif opts.dataset_name == 'nocs':
        if training:
            dataset = NOCSDataset(opts)
        else:
            dataset = NOCSTestDataset(opts)
        return dataset
    else:
        raise NotImplementedError


def data_loader(opts):
    dataset = get_dataset(opts, training=True)
    if opts.local_rank != -1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=opts.ngpu,
            rank=opts.local_rank,
            shuffle=False
        )
        dataloader = DataLoader(dataset, batch_size=opts.batch_size*opts.repeat, num_workers=opts.num_workers, drop_last=True, pin_memory=False, sampler=sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=opts.batch_size*opts.repeat, num_workers=opts.num_workers, drop_last=True, pin_memory=False, shuffle=False)
    return dataloader, dataset


def test_loader(opts):
    dataset = get_dataset(opts, training=False)  # repeat = 1
    # dataset.training = False
    # dataset.portion = 1
    if opts.local_rank != -1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=opts.ngpu,
            rank=opts.local_rank,
            shuffle=False
        )
        dataloader = DataLoader(dataset, batch_size=opts.batch_size, num_workers=opts.num_workers, pin_memory=False, sampler=sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=opts.batch_size, num_workers=opts.num_workers, pin_memory=False, shuffle=opts.shuffle_test)
    return dataloader, dataset

