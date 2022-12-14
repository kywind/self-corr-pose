import torch
import torch.nn.functional as F
import numpy as np


def get_symm_rots(division):
    # symm_rots = torch.eye(3)[None]
    symm_rots = torch.zeros(division, 3, 3)
    for i in range(division):
        theta = 2 * torch.pi / division * i
        rot = torch.tensor([[np.cos(theta), 0, np.sin(theta)],
                            [0, 1, 0],
                            [-np.sin(theta), 0, np.cos(theta)]])
        symm_rots[i] = rot
    return symm_rots
