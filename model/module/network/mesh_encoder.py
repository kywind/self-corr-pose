import torch
import torch.nn as nn
import torch.nn.functional as F


class STN3d_noBN(nn.Module):
    def __init__(self):
        super(STN3d_noBN, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 128, 1)
        self.fc = nn.Linear(128, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 128)
        x = self.fc(x)
        x[:, 0] += 1
        x[:, 4] += 1
        x[:, 8] += 1
        x = x.view(-1, 3, 3)
        return x


class MeshEncoder(nn.Module):
    def __init__(self, n_feat):
        super(MeshEncoder, self).__init__()
        self.stn = STN3d_noBN()
        self.conv1 = torch.nn.Conv1d(3, n_feat, 1)

    def forward(self, x):  # x: b,n,3
        x = x.transpose(2, 1)  # b,3,n
        trans = self.stn(x)
        x = x.transpose(2, 1)  # b,n,3
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)  # b,3,n
        x = F.relu(self.conv1(x))  # b,c,n
        x = x.transpose(2, 1)
        return x  # b,n,c