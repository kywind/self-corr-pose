import os
import numpy as np
import trimesh
import configparser
import argparse
from objectron.dataset.box import Box
from objectron.dataset.graphics import draw_annotation_on_image
import cv2
import matplotlib.pyplot as plt


aux_path = '17_video-aux-00000.txt'
pred_path = '17_video-cam-00000.txt'
obj_path = '17_video-mesh-00000.obj'

# aux = None
# if (os.path.exists(aux_path)):
#     with open(aux_path) as f:
#         aux = f.read().split()
# for i in range(len(aux)):
#     aux[i] = eval(aux[i])
# foc = aux[:2]
# center = aux[2:4]
# ppoint = aux[4:6]
# cams = aux[6:]

pred = None
if (os.path.exists(pred_path)):
    with open(pred_path) as f:
        pred = f.read().split()
for i in range(len(pred)):
    pred[i] = eval(pred[i])
pred = np.array(pred).reshape(4, 4)
print(pred)

fx, fy = pred[3, 0], pred[3, 1]
px, py = pred[3, 2], pred[3, 3]

transformation = pred.copy()
transformation[3, :3] = 0
transformation[3, 3] = 1
print(transformation)


# obj_path = 'SoftRas/data/obj/sphere/sphere_642.obj'
obj = None
if (os.path.exists(obj_path)):
    obj = trimesh.load_mesh(obj_path)
print(obj.vertices) # (n, 3)

minx, maxx, miny, maxy, minz, maxz = obj.vertices[:,0].min(), obj.vertices[:,0].max(), \
    obj.vertices[:,1].min(), obj.vertices[:,1].max(), obj.vertices[:,2].min(), obj.vertices[:,2].max()
scale = np.array([maxx - minx, maxy - miny, maxz - minz])

bbox = np.array([
    [(minx+maxx)/2, (miny+maxy)/2, (minz+maxz)/2],
    [minx, miny, minz], 
    [minx, miny, maxz],
    [minx, maxy, minz],
    [minx, maxy, maxz],
    [maxx, miny, minz],
    [maxx, miny, maxz],
    [maxx, maxy, minz],
    [maxx, maxy, maxz]
])
# adjacent_list = [[], [2, 3, 4, 5], [3, 4, 6], [4, 7], [8], [6, 7], [8], [8], []]
# print(maxx-minx, maxy-miny, maxz-minz)


rotation = transformation[:3, :3].T
translation = transformation[:3, 3]
bbox = bbox @ rotation + translation
print(bbox)
# obj.vertices = obj.vertices @ rotation + translation

# box = Box(bbox)
# box = box.apply_transformation(transformation)
box = Box.from_transformation(transformation[:3, :3], transformation[:3, 3], scale)
print(box.vertices)

def draw_boxes(boxes = [], clips = [], colors = ['r', 'b', 'g' , 'k']):
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
  ax.view_init(30, 12)
  plt.draw()
  plt.show()

# img_orig = cv2.imread('17_video-00000.jpg')
# img = img_orig.copy()
# img = draw_annotation_on_image(img, box.vertices, np.array([9]))
# cv2.imshow(img)

# cv2.circle(img, (int(px), int(py)), 3, (0, 255, 0), -1)
# proj = [(0, 0)]
# for i in range(1, 9):
#     point = bbox[i]
#     proj.append((int(fx * point[0] / point[2] + px), int(fy * point[1] / point[2] + py)))
# print(i, proj)
# 
# for i in range(obj.vertices.shape[0]):
#     point = obj.vertices[i]
#     vproj = (int(fx * point[0] / point[2] + px), int(fy * point[1] / point[2] + py))
#     cv2.circle(img, vproj, 5, (255, 0, 0), -1)
# img = img * 0.5 + img_orig * 0.5
# 
# for i in range(1, 9):
#     cv2.circle(img, proj[i], 2, (0, 0, 255), -1)
#     for j in adjacent_list[i]:
#         cv2.line(img, proj[i], proj[j], (0, 0, 255), 5)
# 
# cv2.imwrite('%s/pose-%05d.jpg' % (testdir, frameid), img)
# # rotation, translation, scale
# # print(minx, maxx, miny, maxy, minz, maxz)
# frameid += dframe
