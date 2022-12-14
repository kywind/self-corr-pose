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

from absl import app
from absl import flags

import numpy as np
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from model.trainer import Trainer
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
    torch.cuda.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    trainer = Trainer(opts)
    trainer.train()

if __name__ == '__main__':
    app.run(main)
