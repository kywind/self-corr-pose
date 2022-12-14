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
import pdb

from absl import flags, app

import torch

from model.tester import Tester
import config

opts = flags.FLAGS


def main(_):
    if opts.local_rank != -1:
        torch.cuda.set_device(opts.local_rank)
        torch.distributed.init_process_group(
            'nccl',
            init_method='env://',
            world_size=opts.ngpu,
            rank=opts.local_rank,
        )
        print('using distributed training...')
        print('world size: {}, local rank: {}'.format(opts.ngpu, opts.local_rank))
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.ngpu)
    opts.train = False
    tester = Tester(opts)
    tester.test()


if __name__ == '__main__':
    app.run(main)
