import torch
import numpy as np



def triangle_direction_intersection(tri, trg):
    '''
    Finds where an origin-centered ray going in direction trg intersects a triangle.
    Args:
        tri: 3 X 3 vertex locations. tri[0, :] is 0th vertex.
    Returns:
        alpha, beta, gamma
    '''
    p0 = np.copy(tri[0, :])
    # Don't normalize
    d1 = np.copy(tri[1, :]) - p0;
    d2 = np.copy(tri[2, :]) - p0;
    d = trg / np.linalg.norm(trg)

    mat = np.stack([d1, d2, d], axis=1)

    try:
      inv_mat = np.linalg.inv(mat)
    except np.linalg.LinAlgError:
      return False, 0

    # inv_mat = np.linalg.inv(mat)
    
    a_b_mg = -1*np.matmul(inv_mat, p0)
    is_valid = (a_b_mg[0] >= 0) and (a_b_mg[1] >= 0) and ((a_b_mg[0] + a_b_mg[1]) <= 1) and (a_b_mg[2] < 0)
    if is_valid:
        return True, -a_b_mg[2]*d
    else:
        return False, 0

def project_verts_on_mesh(verts, mesh_verts, mesh_faces):
    verts_out = np.copy(verts)
    for nv in range(verts.shape[0]):
        max_norm = 0
        vert = np.copy(verts_out[nv, :])
        for f in range(mesh_faces.shape[0]):
            face = mesh_faces[f]
            tri = mesh_verts[face, :]
            # is_v=True if it does intersect and returns the point
            is_v, pt = triangle_direction_intersection(tri, vert)
            # Take the furthest away intersection point
            if is_v and np.linalg.norm(pt) > max_norm:
                max_norm = np.linalg.norm(pt)
                verts_out[nv, :] = pt

    return verts_out

def random_point(face_vertices):
    """ Sampling point using Barycentric coordiante.

    """
    r = torch.zeros((*face_vertices.shape[0:2], 2), device=face_vertices.device).uniform_()
    sqrt_r1 = torch.sqrt(r[:, :, 0:1])
    r2 = r[:, :, 1:2]
    point = (1 - sqrt_r1) * face_vertices[:, :, 0, :] + \
        sqrt_r1 * (1 - r2) * face_vertices[:, :, 1, :] + \
        sqrt_r1 * r2 * face_vertices[:, :, 2, :]

    return point

def pairwise_distance(A, B):
    """ Compute pairwise distance of two point clouds.point

    Args:
        A: b x n x 3 numpy array
        B: b x m x 3 numpy array

    Return:
        C: b x n x m numpy array

    """
    diff = A[:, :, None] - B[:, None, :]
    C = torch.sqrt(torch.sum(diff**2, dim=-1))

    return C

def uniform_sample(mean_v, faces, vertices, n_samples, with_normal=False):
    """ Sampling points according to the area of mesh surface.

    """
    # sampled_points = torch.zeros((n_samples, 3))
    # normals = torch.zeros((n_samples, 3))
    
    # bsz, nf, _ = faces.shape
    # faces_v = torch.gather(vertices, 1, faces.reshape(bsz, nf*3)[:,:,None].repeat(1,1,3)).reshape(bsz, nf, 3, 3)
    # vec_cross = torch.cross(faces_v[:,:,1,:] - faces_v[:,:,0,:], faces_v[:,:,2,:] - faces_v[:,:,0,:])  # bsz, nf, 3
    # face_area = 0.5 * vec_cross.norm(2, -1)
    # cum_area = torch.cumsum(face_area, -1)  # bsz, nf
    # random_values = torch.zeros((bsz, n_samples), device=vertices.device).uniform_() * cum_area[:, -1:]
    # face_id = torch.searchsorted(cum_area, random_values)  # bsz, n_samples
    # sampled_points = random_point(torch.gather(faces_v, 1, face_id[:,:,None,None].repeat(1,1,3,3)))  # face_vertices: (b,k,3,3), k is number of faces

    bsz = vertices.shape[0]
    nf = faces.shape[0]
    faces_meanv = mean_v[faces]  # nf,3,3
    vec_cross = torch.cross(faces_meanv[:,1,:] - faces_meanv[:,0,:], faces_meanv[:,2,:] - faces_meanv[:,0,:])  # nf, 3
    face_area = 0.5 * vec_cross.norm(2, -1)
    cum_area = torch.cumsum(face_area, -1)
    random_values = torch.zeros(n_samples, device=vertices.device).uniform_() * cum_area[-1]
    face_id = torch.searchsorted(cum_area, random_values)

    faces = faces
    faces_v = torch.gather(vertices, 1, faces.reshape(-1)[None,:,None].repeat(bsz,1,3)).reshape(bsz, nf, 3, 3)
    face_id = face_id[None].repeat(bsz, 1)
    faces_v_sampled = torch.gather(faces_v, 1, face_id[:,:,None,None].repeat(1,1,3,3))  # (b,k,3,3)

    r = torch.zeros((1, n_samples, 2), device=vertices.device).uniform_()
    sqrt_r1 = torch.sqrt(r[:, :, 0:1])
    r2 = r[:, :, 1:2]
    sampled_points = (1 - sqrt_r1) * faces_v_sampled[:, :, 0, :] + \
                    sqrt_r1 * (1 - r2) * faces_v_sampled[:, :, 1, :] + \
                    sqrt_r1 * r2 * faces_v_sampled[:, :, 2, :]
    
    # if with_normal:
    #     normals = torch.gather(vec_cross, 1, face_id[:,:,None].repeat(1,1,3))
    #     normals = normals / normals.norm(2, -1, keepdim=True)
    #     sampled_points = torch.cat((sampled_points, normals), dim=2)
    return sampled_points

def farthest_point_sampling(points, n_samples):
    """ Farthest point sampling.

    """
    bsz, n_points, _ = points.shape

    # selected_pts = torch.zeros((bsz, n_samples), device=points.device, dtype=torch.int64)
    # dist_mat = pairwise_distance(points, points)  # bsz, n_points, n_points
    # pt_idx = torch.zeros(bsz, device=points.device, dtype=torch.int64)
    # dist_to_set = torch.gather(dist_mat, 1, pt_idx[:, None, None].repeat(1,1,n_points)).reshape(bsz, n_points)

    selected_pts = torch.zeros(n_samples, device=points.device, dtype=torch.int64)
    dist_mat = pairwise_distance(points, points).mean(0)
    pt_idx = 0
    dist_to_set = dist_mat[:, pt_idx]

    for i in range(n_samples):
        # selected_pts[:, i] = pt_idx
        # dist_to_set = torch.minimum(dist_to_set, \
        #     torch.gather(dist_mat, 1, pt_idx[:, None, None].repeat(1,1,n_points)).reshape(bsz, n_points))
        # pt_idx = torch.argmax(dist_to_set, dim=1)

        selected_pts[i] = pt_idx
        dist_to_set = torch.minimum(dist_to_set, dist_mat[:, pt_idx])
        pt_idx = torch.argmax(dist_to_set)
        
    return selected_pts

def sample_points_from_mesh(mean_v, faces, verts, n_pts, with_normal=False, fps=False, ratio=2):
    """ Uniformly sampling points from mesh model.

    Args:
        path: path to OBJ file.
        n_pts: int, number of points being sampled.
        with_normal: return points with normal, approximated by mesh triangle normal
        fps: whether to use fps for post-processing, default False.
        ratio: int, if use fps, sample ratio*n_pts first, then use fps to sample final output.

    Returns:
        points: n_pts x 3, n_pts x 6 if with_normal = True

    """
    if fps:
        points = uniform_sample(mean_v, faces, verts, ratio*n_pts, with_normal)
        pts_idx = farthest_point_sampling(points[:, :, :3], n_pts)
        # points = torch.gather(points, 1, pts_idx[:, :, None].repeat(1,1,3))
        points = points[:, pts_idx, :]
    else:
        points = uniform_sample(mean_v, faces, verts, n_pts, with_normal)
    return points
