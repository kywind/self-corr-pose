import numpy as np
# import torch

def rotation_err_matrix_torch(R_pred, R_gt):
    # compute the angle distance between rotation matrix in degrees
    R_err = torch.acos(((torch.sum(R_pred * R_gt, 1)).clamp(-1., 3.) - 1.) / 2)
    R_err = R_err * 180. / np.pi
    return R_err

def rotation_err_matrix_np(R_pred, R_gt):
    # compute the angle distance between rotation matrix in degrees
    R_err = np.arccos(((np.sum(R_pred * R_gt, 1)).clip(-1., 3.) - 1.) / 2)
    R_err = R_err * 180. / np.pi
    return R_err

def rotation_acc_matrix_torch(preds, targets, th=30.):
    R_err = rotation_err_matrix_torch(preds, targets)
    return 100. * torch.mean((R_err <= th).float())

def rotation_acc_matrix_np(preds, targets, th=30.):
    R_err = rotation_err_matrix_np(preds, targets)
    return 100. * np.mean(R_err <= th)

def quaternion_distance_np(p, q):
    """Quaternion distance (angle, in radians) in numpy"""
    if p.ndim > 1:
        batch_dot = np.sum(p*q, axis=-1)
        return 2*np.arccos(np.abs(batch_dot))
    else:
        return 2*np.arccos(np.abs(p.dot(q)))

def bdot(a, b):
    """Batched vector dot product in pytorch"""
    B = a.shape[0]
    S = a.shape[1]
    return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).reshape(-1)

def quaternion_distance_torch(p, q):
    """Quaternion distance (angle, in radians) in pytorch
    
    Args:
        p (torch.Tensor, size=(b, 4)): Predicted quaternion (normed to S3 sphere)
        q (torch.Tensor, size=(b, 4)): Target quaternion (on S3 sphere)
    """
    if p.ndim > 1:
        return 2*torch.acos(bdot(p, q).clamp(-1., 1.))    
    else:
        return 2*torch.acos(torch.dot(p,q).clamp(-1., 1.))

def least_squares_solution(x1, x2):
    """
    x1, x2 both define i=1,...,I 3-dimensional points, with known correspondence
    https://www.youtube.com/watch?v=re08BUepRxo
    Assumption: points x1, x2 related by a similarity:
      E(x2_i) = λ*R*x1_i+t, {i = 1, ..., I}
    Task: estimate the parameters, provide uncertainty

    Args:
        x1 (array): Shape (3, N)
        x2 (array): Shape (3, N)

    Returns:
        rot (array): Shape (3, 3), a rotation matrix
        t (array):   Shape (3, 1), a translation vector
        lam (float): Scaling factor

    """
    w1 = (1/0.1**2) * np.ones(x1.shape[1]) # 'weights' are (1/sig^2) - fix for now
    w2 = (1/0.1**2) * np.ones(x2.shape[1]) # 'weights' are (1/sig^2) - fix for now
    # Find centroid (weighted) of the 'observed' (x2) points
    x2_C = (np.sum(x2*w2, axis=1) / np.sum(w2)).reshape(3,1)
    x1_C = (np.sum(x1*w1, axis=1) / np.sum(w1)).reshape(3,1)

    # Unknown params: rotation R, scale λ, modified translation vector u, residuals v_x2_i
    # t = x2_C - λ*R*u

    # Minimising the weighted sum of the residuals, we arrive at
    u = x1_C

    # Approximate solution for lambda, holds for small noise - analytic soln is dependent on R!
    def get_sums(x, xc, w):
        # total = 0
        # for x_i, x_ci, w_i in zip(x, xc, w):
        #     total += w_i * (x_i-x_ci).T @ (x_i-x_ci)
        # return total
        # Or, parallel implementation!
        return np.sum(np.sum((x - xc) * (x - xc), axis=0) * w)
    
    lam_sq = get_sums(x2, x2_C, w2) / get_sums(x1, x1_C, w2)
    lam = np.sqrt(lam_sq)
    # lam = 1

    # Estimation of rotation
    # 1. Centre coordinates
    c_x1 = x1 - x1_C
    c_x2 = x2 - x2_C
    # 2. Create H matrix (3x3)
    H = c_x1 @ np.diag(w2) @ c_x2.T
    # 3. Use SVD to find estimated rotation R
    U, S, Vh = np.linalg.svd(H)
    R = Vh.T @ U.T

    # Get translation parameter using t = x2_C - λ*R*u
    # import pdb
    # pdb.set_trace()
    t = x2_C - lam * R @ u
    return R, t, lam


def rigid_transform_3D(A, B):
    """
    Un-weighted version, from https://github.com/nghiaho12/rigid_transform_3D

    Implementation of "Least-Squares Fitting of Two 3-D Point Sets", Arun, K. S. 
    and Huang, T. S. and Blostein, S. D, [1987]
    """
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    print("R before correcting for reflection:")
    print(R)
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

def umeyama(src, dst, estimate_scale=True):
    """Estimate N-D similarity transformation with or without scaling.
    Taken from skimage!

    homo_src = np.hstack((src, np.ones((len(src), 1))))
    homo_dst = np.hstack((src, np.ones((len(src), 1))))

    homo_dst = T @ homo_src, where T is the returned transformation

    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.eye(4), 1
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V
    
    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale
    
    return T, scale

    # R = T[:dim, :dim]
    # t = T[:dim, dim:]
    # return R, t, scale
