import cv2
import numpy as np
import torch
from pytorch3d.renderer.cameras import get_ndc_to_screen_transform
from pytorch3d.renderer.implicit import NDCGridRaysampler

def cv2_depth_fillmask(depth_crop, depth_nan_mask):
    kernel = np.ones((3, 3),np.uint8)
    depth_nan_mask = cv2.dilate(depth_nan_mask, kernel, iterations=1)

    depth_crop[depth_nan_mask==1] = 0

    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    depth_scale = np.abs(depth_crop).max()
    depth_crop = depth_crop.astype(np.float32) / depth_scale  # Has to be float32, 64 not supported.

    depth_crop = cv2.inpaint(depth_crop, depth_nan_mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.
    depth_crop = depth_crop[1:-1, 1:-1]
    depth_crop = depth_crop * depth_scale
    return depth_crop

def cv2_depth_fillzero(depth_crop):
    # OpenCV inpainting does weird things at the border.
    depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    depth_nan_mask = (depth_crop==0).astype(np.uint8)
    depth_crop = cv2_depth_fillmask(depth_crop, depth_nan_mask)
    return depth_crop

def cv2_depth_fillna(depth_crop):
    # OpenCV inpainting does weird things at the border.
    depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)
    depth_crop = cv2_depth_fillmask(depth_crop, depth_nan_mask)
    return depth_crop


def transform_cameraframe_to_screen(cam, P_cam, **kwargs):
    """My version of Camera.transform_points_screen, goes from camera rather than world coords
    
    Assumes that P_cam is already in the camera coordinate frame (though not yet NDC).
    This means that, given world coordinate points and a camera with a viewpoint given by
    R and T, P_cam is the result of:
        P_cam = cam.get_world_to_view_transform().transform_points(P_world)
    
    This function then takes all of the other steps required to use the camera intrinsics to
    form an image (or more properly "screen" coordinates) of the pointcloud.
    """
    # Can't just call cam.transform_points_ndc as this assumes points in world
    # points_ndc = cam.transform_points_ndc(points, eps=eps, **kwargs)
    view_to_ndc_transform = cam.get_projection_transform()
    if not cam.in_ndc():
        to_ndc_transform = cam.get_ndc_camera_transform()
        view_to_ndc_transform = view_to_ndc_transform.compose(to_ndc_transform)

    points_ndc = view_to_ndc_transform.transform_points(P_cam)
    image_size = kwargs.get("image_size", cam.get_image_size())
    
    return get_ndc_to_screen_transform(
                cam, with_xyflip=True, image_size=image_size
            ).transform_points(points_ndc)

def get_structured_pcd(frame, inpaint=True, world_coordinates=False):
    """Takes a frame from CO3D dataset, returns a structured pointcloud

    Pointcloud returned is in world-coordinates, with shape (H, W, 3), 
    with the 3 dimensions encoding X, Y and Z world coordinates.
    Optionally infills the depth map from the CO3D dataset, which tends
    to be ~50% zeros (the equivalent of NaNs in the .png encoding). This
    leads to fewer NaNs in the unprojected structured pointcloud.

    world_coordinates passed to cam.unproject_points: if it is false, the
    points are unprojected to the *camera* frame, else to the world frame
    """
    # --- 1. Get transform from screen to NDC space ---
    # image_size in Pytorch3D camera is (H, W) ordering
    H, W = frame.image_rgb.shape[1:]
    # get the ndc_to_screen transform, invert it for screen to NDC
    ndc_to_screen_transform = get_ndc_to_screen_transform(
        frame.camera, with_xyflip=False, image_size=(H,W)).inverse().get_matrix().squeeze()

    # --- 2. Make a grid of the coordinates in Pytorch3D screen space ---
    x = torch.arange(W)
    y = torch.arange(H)
    grid_y, grid_x = torch.meshgrid(y, x) # both returns have shape (H, W)

    if inpaint:
        depth_proc = torch.Tensor(cv2_depth_fillzero(frame.depth_map.squeeze().numpy()))
    else:
        depth_proc = frame.depth_map.squeeze()

    # Stack the xy coords with depth, which is in NDC/screen format (no difference in this case, I think)
    xy_grid = torch.stack((grid_y, grid_x, depth_proc, torch.ones_like(grid_x)), dim=-1)
    
    # --- 3. Convert the 4D vectors to NDC space ---
    # Transpose the transform to operate on column vectors (i.e. normal convention, but not in Pytorch3D)
    xy_grid_ndc = ndc_to_screen_transform.T @ xy_grid.float().view(-1,4).T # (4, H*W)
    # --- 4. Unproject ---
    unproj = frame.camera.unproject_points(xy_grid_ndc.T[...,:3], world_coordinates) # (H*W, 3)
    # --- 5. Convert back to image shape, remove homogeneous dimension ---
    structured_pcd = unproj.view(H,W,3)
    
    return structured_pcd

def get_structured_pcd2(frame, inpaint=True, world_coordinates=False):
    """Takes a frame from CO3D dataset, returns a structured pointcloud

    Pointcloud returned is in world-coordinates, with shape (H, W, 3), 
    with the 3 dimensions encoding X, Y and Z world coordinates.
    Optionally infills the depth map from the CO3D dataset, which tends
    to be ~50% zeros (the equivalent of NaNs in the .png encoding). This
    leads to fewer NaNs in the unprojected structured pointcloud.

    world_coordinates passed to cam.unproject_points: if it is false, the
    points are unprojected to the *camera* frame, else to the world frame
    """
    # --- 1. Get transform from screen to NDC space ---
    # image_size in Pytorch3D camera is (H, W) ordering
    H, W = frame.image_rgb.shape[1:]
    # get the ndc_to_screen transform, invert it for screen to NDC
    ndc_to_screen_transform = get_ndc_to_screen_transform(
        frame.camera, with_xyflip=False, image_size=(H,W)).inverse().get_matrix().squeeze()

    # --- 2. Make a grid of the coordinates in Pytorch3D screen space ---
    gridsampler = NDCGridRaysampler(image_width=W, image_height=H, n_pts_per_ray=1, min_depth=0, max_depth=1)
    xy_grid = gridsampler._xy_grid

    if inpaint:
        depth_proc = torch.Tensor(cv2_depth_fillzero(frame.depth_map.squeeze().numpy()))
    else:
        depth_proc = frame.depth_map.squeeze()

    # Stack the xy coords with depth, which is in NDC/screen format (no difference in this case, I think)
    xy_grid_ndc = torch.cat((xy_grid,
                           depth_proc.unsqueeze(-1)), dim=-1)
    xy_grid_ndc = xy_grid_ndc.view(-1,3)
    #     return xy_grid
    # --- 4. Unproject ---
    unproj = frame.camera.unproject_points(xy_grid_ndc, world_coordinates) # (H*W, 3)
    # --- 5. Convert back to image shape, remove homogeneous dimension ---
    structured_pcd = unproj.view(H,W,3)
    return structured_pcd