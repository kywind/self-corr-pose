import torch
import numpy as np
import enum


class QuaternionCoeffOrder(enum.Enum):
    XYZW = 'xyzw'
    WXYZ = 'wxyz'

def quat_product(qa, qb):
    """Multiply qa by qb.

    Args:
        qa: B X N X 4 quaternions
        qb: B X N X 4 quaternions
    Returns:
        q_mult: B X N X 4
    """
    qa_0 = qa[..., 0]
    qa_1 = qa[..., 1]
    qa_2 = qa[..., 2]
    qa_3 = qa[..., 3]

    qb_0 = qb[..., 0]
    qb_1 = qb[..., 1]
    qb_2 = qb[..., 2]
    qb_3 = qb[..., 3]

    # See https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    q_mult_0 = qa_0*qb_0 - qa_1*qb_1 - qa_2*qb_2 - qa_3*qb_3
    q_mult_1 = qa_0*qb_1 + qa_1*qb_0 + qa_2*qb_3 - qa_3*qb_2
    q_mult_2 = qa_0*qb_2 - qa_1*qb_3 + qa_2*qb_0 + qa_3*qb_1
    q_mult_3 = qa_0*qb_3 + qa_1*qb_2 - qa_2*qb_1 + qa_3*qb_0

    return torch.stack([q_mult_0, q_mult_1, q_mult_2, q_mult_3], dim=-1)

def quat_rotate(X, quat):
    """Rotate points by quaternions.

    Args:
        X: B X N X 3 points
        quat: B X 4 quaternions

    Returns:
        X_rot: B X N X 3 (rotated points)
    """
    quat = quat[:,None,:].expand(-1,X.shape[1],-1)
    quat_conj = torch.cat([ quat[:, :, 0:1] , -1*quat[:, :, 1:4] ], dim=-1)
    X = torch.cat([ X[:, :, 0:1]*0, X ], dim=-1)
    X_rot = quat_product(quat, quat_product(X, quat_conj))
    return X_rot[:, :, 1:4]

def quat_inverse(quat):
    """
    quat: B x 4: [quaternions]
    returns inverted quaternions
    """
    flip = torch.tensor([1,-1,-1,-1],dtype=quat.dtype,device=quat.device)
    quat_inv = quat * flip.view((1,)*(quat.dim()-1)+(4,))
    return quat_inv

def q_unit():
    return np.asarray([1, 0, 0, 0], np.float32)

def q_rnd_m(b=1):
    randnum = np.random.uniform(0.0, 1.0, size=[3*b])
    u, v, w = randnum[:b,None], randnum[b:2*b,None], randnum[2*b:3*b,None]
    v *= 2.0 * np.pi
    w *= 2.0 * np.pi
    return np.concatenate([(1.0-u)**0.5 * np.sin(v), (1.0-u)**0.5 * np.cos(v), u**0.5 * np.sin(w), u**0.5 * np.cos(w)],-1).astype(np.float32)

def q_scale_m(q, t):
    out = q.copy()
    p=q_unit()
    d = np.dot(p, q.T)
    cond1 = d<0.0
    q[cond1] = -q[cond1]
    d[cond1] = -d[cond1]

    cond2 = d>0.999
    if cond2.sum()>0:
        a = p[None] + t[cond2][:,None] * (q[cond2]-p[None])
        out[cond2] =  a / np.linalg.norm(a,2,-1)[:,None]

    t0 = np.arccos(d)
    tt = t0 * t
    st = np.sin(tt)
    st0 = np.sin(t0)
    s1 = st / st0
    s0 = np.cos(tt) - d*s1
    if (~cond2).sum()>0:
        out[~cond2] =  (s0[:,None]*p[None] + s1[:,None]*q)[~cond2]
    return out
