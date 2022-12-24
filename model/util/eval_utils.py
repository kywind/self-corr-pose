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

"""
Visualizing Utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import math
import sys
sys.path.insert(0,'third-party')

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import kornia

from objectron.dataset import box, iou
from model.util.colormap import label_colormap



def map_kp(kps_vis1, kps_vis2, kps1, kps2, match1, match2, mask1, mask2):
    bsz, n_kps = kps1.shape[0], kps1.shape[1]
    # common_kp_indices = torch.nonzero(kps1[:, :, 2] * kps2[:, :, 2] > 0)
    # kp_mask = torch.zeros((bsz, n_kps)).cuda()
    # kp_mask[common_kp_indices] = 1
    kp_mask = kps_vis1 * kps_vis2
    
    img_H = match2.shape[-2]
    img_W = match2.shape[-1]
    kps1_3d = F.grid_sample(match1, kps1[:, None, :, :2], align_corners=False)[:, :, 0]  # b,3,h,w & b,1,15,2 -> b,3,1,15 -> b,3,15
    distances = torch.norm((kps1_3d[:,:,:,None] - match2.reshape(bsz, 3, 1, img_H*img_W)), dim=1)  # find a specific pixel location on match2 that is closest to kps1_3d
    distances = distances + (1 - mask2.reshape(bsz, 1, img_H*img_W)) * 1000  # b,15,h*w

    min_dist, min_indices = torch.min(distances, dim=2)
    min_dist = min_dist + (1 - kps_vis1).float() * 1000  # bsz,15
    transfer_kps = torch.stack([min_indices % img_W, min_indices // img_W], dim=2)  # bsz,15,2   x,y
    transfer_kps = transfer_kps.float()
    transfer_kps[:, :, 0] = transfer_kps[:, :, 0] * 2 / img_W - 1
    transfer_kps[:, :, 1] = transfer_kps[:, :, 1] * 2 / img_H - 1

    kp_transfer_error = torch.norm((transfer_kps - kps2[:, :, :2]), dim=2)  # bsz,15
    return transfer_kps, kp_transfer_error, min_dist, kp_mask



def draw_kp(img1, img2, kps1, kps2, trans_kps2, kps_mask):  # np arrays, no batch, img: cv2, kps:-1~1
    color_map = label_colormap()
    h, w = img1.shape[0], img1.shape[1]
    trans_img2 = img2.copy()
    kps1[:, 0] = (kps1[:, 0] * 0.5 + 0.5) * w
    kps1[:, 1] = (kps1[:, 1] * 0.5 + 0.5) * h
    kps2[:, 0] = (kps2[:, 0] * 0.5 + 0.5) * w
    kps2[:, 1] = (kps2[:, 1] * 0.5 + 0.5) * h
    trans_kps2[:, 0] = (trans_kps2[:, 0] * 0.5 + 0.5) * w
    trans_kps2[:, 1] = (trans_kps2[:, 1] * 0.5 + 0.5) * h
    for i in range(kps1.shape[0]):
        if kps_mask[i] > 0:
            color = tuple(color_map[i+1])
            cv2.circle(img1, (int(kps1[i, 0]), int(kps1[i, 1])), 3, (int(color[0]), int(color[1]), int(color[2])), -1)
            cv2.circle(img2, (int(kps2[i, 0]), int(kps2[i, 1])), 3, (int(color[0]), int(color[1]), int(color[2])), -1)
            cv2.circle(trans_img2, (int(trans_kps2[i, 0]), int(trans_kps2[i, 1])), 3, (int(color[0]), int(color[1]), int(color[2])), -1)
    img1 = np.flip(img1, -1)
    trans_img2 = np.flip(trans_img2, -1)
    img2 = np.flip(img2, -1)
    # img = np.concatenate((img1, trans_img2, img2), axis=1)
    return img1, trans_img2, img2


def compute_scale(box, plane):
    """Computes scale of the given box sitting on the plane."""
    center, normal = plane
    vertex_dots = [np.dot(vertex, normal) for vertex in box[1:]]
    vertex_dots = np.sort(vertex_dots)
    center_dot = np.dot(center, normal)
    scales = center_dot / vertex_dots[:4]
    return np.mean(scales)


def compute_depth_ratio_objectron(box_point_3d, list_idx, frame_id):
    ## TODO
    plane = None
    scale = compute_scale(box_point_3d, plane)
    raise NotImplementedError
    return scale


def compute_depth_ratio_nocs(pts, depth_render, depth, mask):  # pts: b,n,3; depth: b,h,w, mask: b,1,h,w
    bsz = pts.shape[0]
    pts_depth = pts[:, :, 2]  # bsz,n
    pts_proj = pts[:, :, :2]  # bsz,n,2
    pts_depth_front = F.grid_sample(depth_render[:, None], pts_proj[:, None], align_corners=False)[:, 0, 0]  # b,1,h,w & b,1,n,2 -> b,1,1,n -> b,n
    depth_weight = -10 * F.relu(pts_depth - pts_depth_front)
    depth_weight = depth_weight.exp()
    depth_mask = 1.0 * (mask > 0) * (depth > 0)
    depth[depth_mask == 0] = -1e4
    pts_depth_gt = F.grid_sample(depth[:, None], pts_proj[:, None], mode='nearest', align_corners=False)[:, 0, 0]  # b,1,h,w & b,1,n,2 -> b,1,1,n --> b,n
    
    depth_ratio = torch.zeros(bsz, device=pts.device)
    for i in range(bsz):
        if pts_depth_gt[i].max().item() > 0 and depth_weight[i].max().item() > 0.5:
            pts_depth_i = pts_depth[i]
            pts_depth_gt_i = pts_depth_gt[i]
            depth_weight_i = depth_weight[i]
            pts_depth_i = pts_depth_i[depth_weight_i > 0.5]
            pts_depth_gt_i = pts_depth_gt_i[depth_weight_i > 0.5]
            pts_depth_i = pts_depth_i[pts_depth_gt_i > 0]
            pts_depth_gt_i = pts_depth_gt_i[pts_depth_gt_i > 0]
            pts_depth_gt_i /= 1000.
            ratio = pts_depth_gt_i.mean().item() / pts_depth_i.mean().item()
        else:
            ratio = 0.3 / pts_depth[i].mean().item()
        depth_ratio[i] = ratio
    return depth_ratio


def get_best_iou(symmetry_idx, box_pred, rot_gt, trans_gt, scale_gt):
    angle_ratio = 0
    if symmetry_idx == 0:
        y_axis = rot_gt[:, 1].copy()
        best_iou = best_val = best_ae = best_pe = 0
        best_rot_gt = None

        division = 18
        for i in range(division):
            angle = i * 2 * np.pi / 18
            angle_axis = y_axis * angle
            angle_axis = torch.tensor(angle_axis)
            rot_z = kornia.geometry.angle_axis_to_rotation_matrix(angle_axis[None])
            rot_z = np.array(rot_z[0])
            rot_gt_temp = rot_z @ rot_gt

            box_gt = box.Box.from_transformation(rot_gt_temp, trans_gt, scale_gt)
            iou_pred = iou.IoU(box_pred, box_gt)
            try:
                iou_rel = iou_pred.iou()
            except:
                iou_rel = 0
            ae, pe = evaluate_viewpoint(box_pred.vertices, box_gt.vertices)
            ae_rel = -min(1, ae / 90)
            pe_rel = -min(1, pe / 90)
            # ae = pe = ae_rel = pe_rel = 0

            if iou_rel + (ae_rel + pe_rel)*angle_ratio >= best_val:
                best_iou = iou_rel
                best_ae = ae
                best_pe = pe
                best_val = iou_rel + (ae_rel + pe_rel)*angle_ratio
                best_rot_gt = rot_gt_temp.copy()
    
    else:
        best_rot_gt = rot_gt.copy()
        box_gt = box.Box.from_transformation(best_rot_gt, trans_gt, scale_gt)
        iou_pred = iou.IoU(box_pred, box_gt)
        try:
            best_iou = iou_pred.iou()
        except:
            best_iou = 0
        best_ae, best_pe = evaluate_viewpoint(box_pred.vertices, box_gt.vertices)

    return best_iou, best_ae, best_pe


def get_best_deg_cm(symmetry_idx, box_pred, rot_gt, trans_gt, scale_gt):
    trans_error = 100 * np.linalg.norm(box_pred.vertices[0] - trans_gt)

    # angle error
    if symmetry_idx == 0:
        box_gt = box.Box.from_transformation(rot_gt, trans_gt, scale_gt)
        y_axis_gt = box_gt.vertices[3] - box_gt.vertices[1]
        y_axis_pred = box_pred.vertices[3] - box_pred.vertices[1]
        angle = np.arccos(y_axis_pred.dot(y_axis_gt) / (np.linalg.norm(y_axis_pred) * np.linalg.norm(y_axis_gt)))
    
    else:
        rot_pred = box_pred.rotation
        R = rot_pred @ rot_gt.transpose()
        angle = np.arccos((np.trace(R) - 1) / 2)

    angle *= 180 / np.pi
    return angle, trans_error



def compute_viewpoint(box):
    """Computes viewpoint of a 3D bounding box.

    We use the definition of polar angles in spherical coordinates
    (http://mathworld.wolfram.com/PolarAngle.html), expect that the
    frame is rotated such that Y-axis is up, and Z-axis is out of screen.

    Args:
      box: A 9*3 array of a 3D bounding box.

    Returns:
      Two polar angles (azimuth and elevation) in degrees. The range is between
      -180 and 180.
    """
    x, y, z = compute_ray(box)
    theta = math.degrees(math.atan2(z, x))
    phi = math.degrees(math.atan2(y, math.hypot(x, z)))
    return theta, phi


def compute_ray(bbox):
    """Computes a ray from camera to box centroid in box frame.

    For vertex in camera frame V^c, and object unit frame V^o, we have
      R * Vc + T = S * Vo,
    where S is a 3*3 diagonal matrix, which scales the unit box to its real size.

    In fact, the camera coordinates we get have scale ambiguity. That is, we have
      Vc' = 1/beta * Vc, and S' = 1/beta * S
    where beta is unknown. Since all box vertices should have negative Z values,
    we can assume beta is always positive.

    To update the equation,
      R * beta * Vc' + T = beta * S' * Vo.

    To simplify,
      R * Vc' + T' = S' * Vo,
    where Vc', S', and Vo are known. The problem is to compute
      T' = 1/beta * T,
    which is a point with scale ambiguity. It forms a ray from camera to the
    centroid of the box.

    By using homogeneous coordinates, we have
      M * Vc'_h = (S' * Vo)_h,
    where M = [R|T'] is a 4*4 transformation matrix.

    To solve M, we have
      M = ((S' * Vo)_h * Vc'_h^T) * (Vc'_h * Vc'_h^T)_inv.
    And T' = M[:3, 3:].

    Args:
      box: A 9*3 array of a 3D bounding box.

    Returns:
      A ray represented as [x, y, z].
    """
    # if bbox[0, -1] > 0:
    #   warnings.warn('Box should have negative Z values.')

    size_x = np.linalg.norm(bbox[5] - bbox[1])
    size_y = np.linalg.norm(bbox[3] - bbox[1])
    size_z = np.linalg.norm(bbox[2] - bbox[1])
    size = np.asarray([size_x, size_y, size_z])
    box_o = box.UNIT_BOX * size
    box_oh = np.ones((4, 9))
    box_oh[:3] = np.transpose(box_o)

    box_ch = np.ones((4, 9))
    box_ch[:3] = np.transpose(bbox)
    box_cht = np.transpose(box_ch)

    box_oct = np.matmul(box_oh, box_cht)
    box_cct_inv = np.linalg.inv(np.matmul(box_ch, box_cht))
    transform = np.matmul(box_oct, box_cct_inv)
    return transform[:3, 3:].reshape((3))
    

def evaluate_viewpoint(box, instance):
    """Evaluates a 3D box by viewpoint.

    Args:
      box: A 9*3 array of a predicted box.
      instance: A 9*3 array of an annotated box, in metric level.

    Returns:
      Two viewpoint angle errors.
    """
    predicted_azimuth, predicted_polar = compute_viewpoint(box)
    gt_azimuth, gt_polar = compute_viewpoint(instance)

    polar_error = abs(predicted_polar - gt_polar)
    # Azimuth is from (-180,180) and a spherical angle so angles -180 and 180
    # are equal. E.g. the azimuth error for -179 and 180 degrees is 1'.
    azimuth_error = abs(predicted_azimuth - gt_azimuth)
    if azimuth_error > 180:
      azimuth_error = 360 - azimuth_error

    return azimuth_error, polar_error


def draw_bboxes(img, img_pts, dir_pts=None, color=(0,0,255), width=3, cx=(0,0,255), cy=(0,255,0), cz=(255,0,0)):  # colorx, colory, colorz
    img_pts = np.int32(img_pts).reshape(-1, 2)
    # draw ground layer in darker color
    color_ground = (int(color[0]*0.3), int(color[1]*0.3), int(color[2]*0.3))
    for i, j in zip([3,4,8,7], [4,8,7,3]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_ground, width)
    # draw pillars in minor darker color
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip([1,2,5,6], [3,4,7,8]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_pillar, width)
    # draw top layer in original color
    for i, j in zip([1,2,6,5], [2,6,5,1]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color, width)

    if dir_pts is None:
        img_pts = img_pts[1:]
        center = tuple(np.int32(img_pts.mean(0)))
        center_x = tuple(np.int32((img_pts[[1,3,5,7]].mean(0) - img_pts.mean(0)) * 0.8 + img_pts.mean(0)))
        center_y = tuple(np.int32((img_pts[[0,1,4,5]].mean(0) - img_pts.mean(0)) * 0.8 + img_pts.mean(0)))
        center_z = tuple(np.int32((img_pts[[4,5,6,7]].mean(0) - img_pts.mean(0)) * 0.8 + img_pts.mean(0)))
    else:
        dir_pts = np.int32(dir_pts).reshape(-1, 2)
        center = dir_pts[0]
        center_x, center_y, center_z = dir_pts[1], dir_pts[2], dir_pts[3]
    img = cv2.line(img, center, center_x, cx, width)
    img = cv2.line(img, center, center_y, cy, width)
    img = cv2.line(img, center, center_z, cz, width)
    return img


def draw_bboxes_3d(savedir, boxes, clips=[], colors=['r','b','g','k'], alpha=30, beta=12):
    """Draw a list of boxes.
    The boxes are defined as a list of vertices
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i, b in enumerate(boxes):
        x, y, z = b[:, 0], b[:, 1], b[:, 2]
        ax.scatter(x, y, z, c = 'r')
        for e in box.EDGES:
            ax.plot(x[e], y[e], z[e], linewidth=2, c=colors[i % len(colors)])

    if (len(clips)):
        points = np.array(clips)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=100, c='k')

    plt.gca().patch.set_facecolor('white')
    ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))

    # rotate the axes and update
    ax.view_init(alpha, beta)
    plt.draw()
    plt.savefig(savedir)
    plt.close()



