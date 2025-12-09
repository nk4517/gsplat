import pickle
from typing import Tuple

import numpy as np
import torch

# from scene import Scene, GaussianModel


inv = np.linalg.inv

class WaymoCoordsHelper:

    xf_waymo2ocv = np.array([
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [1,  0,  0, 0],
        [0,  0,  0, 1]
    ])

    waymo_cam_ids = {
        1: "FRONT",
        2: "FRONT_LEFT",
        3: "FRONT_RIGHT",
        4: "SIDE_LEFT",
        5: "SIDE_RIGHT"
    }

    def __init__(self, path):
        # интринсики и экстринсики (позы камер в координатах фрейма)
        with open(path / "calibs.dump", "rb") as f:
            self.calibs = pickle.load(f)

        # позы всех камер всех фреймов сразу в мировых координатах
        with open(path / "cam_poses.dump", "rb") as f:
            self.cam_poses = pickle.load(f)

        # позы всех фреймов в мировых координатах
        with open(path / "frame_poses.dump", "rb") as f:
            self.frame_poses = pickle.load(f)

        self.frame0_pose = self.frame_poses[0]


    def get_calib_as_ocv_rel_fr0(self, frame_idx: int, cam: int | str) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(cam, int):
            cam = WaymoCoordsHelper.waymo_cam_ids[cam]
        cam = cam.upper()

        w, h, K, d, xform_to_sdc = self.calibs[cam]
        cam_pose_orig = self.frame_poses[frame_idx]
        pose_of_cam_rel_to_fr0 = inv(inv(cam_pose_orig) @ self.frame0_pose)

        cam_pose = self.xform_waymo2ocv(pose_of_cam_rel_to_fr0 @ xform_to_sdc)

        return w, h, K, d, cam_pose

    def points_frame2fr0(self, frame_id, points):
        # ВОТ ИМЕННО ТАК. в координатную систему самого первого фрейма
        pcd_rel_pose = self.pose_frame2fr0(frame_id)
        return self.transform_points(points, pcd_rel_pose)

    def pose_frame2fr0(self, frame_id):
        pcd_rel_pose = inv(inv(self.frame_poses[frame_id]) @ self.frame0_pose)
        return pcd_rel_pose

    @staticmethod
    def transform_points(points, xform):
        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        transformed_points = np.dot(points_homogeneous, xform.T)
        return transformed_points[:, :3]  # / transformed_points[:, 3:4]


    @classmethod
    def points_waymo2ocv(cls, points):
        return cls.transform_points(points, cls.xf_waymo2ocv)


    @classmethod
    def xform_waymo2ocv(cls, xform):
        xform = cls.xf_waymo2ocv @ (xform @ inv(cls.xf_waymo2ocv))
        return xform

        # if xform.ndim == 2:
        #     xform = Z_up_to_fwd @ (xform @ inv(Z_up_to_fwd))
        # elif xform.ndim == 3:
        #     xform = np.einsum('ij,njk,kl->nil', Z_up_to_fwd, xform, inv(Z_up_to_fwd))
        # else:
        #     raise ValueError("xform should be either 2D or 3D array")
        # return xform


def add_waymo_points3d(waymo, scene: "Scene", gaussians: "GaussianModel", points3d):
    from pathlib import Path

    lidar_path = Path(r"x:\_ai\_waymo\tensorflow_extractor\colmap_proj\lidar")
    # for ply_path in lidar_path.glob("*.ply"):
    #     stem = ply_path.stem
    #     if not stem.isdigit():
    #         continue
    #     pcd_id = int(stem.lstrip("0") or "0")
    #
    #     if pcd_id > 10:
    #         break
    #
    from plyfile import PlyData
    import numpy as np

    waymo_frameids = {}

    for viewpoint_cam in scene.getTrainCameras():
        from scene.cameras import Camera
        viewpoint_cam: Camera

        waymo_cam, waymo_frame = viewpoint_cam.image_name.split("/")
        ff = int(waymo_frame.lstrip("0") or 0)

        if ff not in waymo_frameids:
            waymo_frameids[ff] = {}

        waymo_frameids[ff][waymo_cam] = viewpoint_cam

    from utils.pcpr_utils import PCPRRenderer

    def recover_R_t(Rt):
        R = Rt[:3, :3].T
        t = Rt[:3, 3]
        return R, t

    for frame_id, ffzz in waymo_frameids.items():

        ply_path = lidar_path / f"{frame_id:04d}.ply"

        plydata = PlyData.read(ply_path)
        vertices = plydata['vertex']
        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

        points3d = torch.from_numpy(waymo.points_waymo2ocv(waymo.points_frame2fr0(frame_id, positions))).cuda()

        for waymo_cam, viewpoint_cam in ffzz.items():
            viewpoint_cam: Camera

            # c2w_xformW = waymo.get_calib_as_ocv_rel_fr0(frame_idx=frame_id, cam=waymo_cam)[-1]
            # w2c_xformW = torch.tensor(np.linalg.inv(c2w_xformW)).float()

            w2c_xform = torch.eye(4)
            w2c_xform[:3, :3] = torch.tensor(viewpoint_cam.R.T) # rasterizer GLM
            w2c_xform[:3, 3] = torch.tensor(viewpoint_cam.T)

            w2c_xform = torch.inverse(w2c_xform)

            w = viewpoint_cam.image_width
            h = viewpoint_cam.image_height
            K = viewpoint_cam.K

            lidar_depthmap, index_map_coarse = PCPRRenderer.render_depthmap(
                points3d, w, h, K, w2c_xform, max_splatting_size=0.1)

            # from modules.common_img import turbo_img
            # turbo_img(f"lidar_{frame_id:04d}_{waymo_cam}.png", lidar_depthmap.detach().cpu().numpy())

            lidar_sparsity = 1
            lidar_mask = (index_map_coarse != 0).cuda()

            random_mask = torch.rand(lidar_mask.shape, device=lidar_mask.device)
            lidar_mask_to_propagate = lidar_mask & (random_mask >= (1-lidar_sparsity))

            gt_image = viewpoint_cam.original_image.cuda()

            gaussians.densify_from_depthmap(
                viewpoint_cam,
                lidar_depthmap.clone().cuda(),
                lidar_mask_to_propagate.to(torch.bool).clone().cuda(),
                gt_image
            )

    # raise 1

