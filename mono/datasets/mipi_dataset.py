from __future__ import absolute_import, division, print_function

import os
import scipy.misc
import numpy as np
import PIL.Image as pil
import datetime
import cv2
import torch

from .folder_dataset import FolderDataset

np.seterr(divide='ignore', invalid='ignore')


class MIPIDataset(FolderDataset):
    """Superclass for different types of MIPI dataset loaders
    """

    def __init__(self, *args, **kwargs):
        super(MIPIDataset, self).__init__(*args, **kwargs)

        # fx_rgb = 246.015983366588
        # fy_rgb = 246.368835728228
        # cx_rgb = 142.091599383949
        # cy_rgb = 101.569221725907
        # K_rgb = np.array([[fx_rgb / 240, 0, cx_rgb / 240],
        #                   [0, fy_rgb / 180, cy_rgb / 180],
        #                   [0, 0, 1]])
        self.K = np.array([[1.0250666, 0., 0.59204833, 0.],
                           [0., 1.36871575, 0.56427345, 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]], dtype=np.float32)

        # fx_d = 212.10850384897276
        # fy_d = 212.35908920377838
        # cx_d = 118.677436313877
        # cy_d = 86.74686752467763
        # K_d = np.array([[fx_d * 256 / 240, 0, cx_d * 256 / 240],
        #                 [0, fy_d * 192 / 180, cy_d * 192 / 180],
        #                 [0, 0, 1]])
        self.K_d = np.array([[226.24907008, 0., 126.58926592],
                             [0., 226.51636224, 92.52999168],
                             [0., 0., 1.]], dtype=np.float32)

        # Pixel coordinates
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        ones = np.ones((self.height, self.width))
        uvone = np.stack((u, v, ones), axis=2)  # H*W*3
        uvone = uvone.reshape((-1, 3))  # (H*W)*3

        # Camera coordinates
        self.PRJd = (np.linalg.inv(self.K_d) @ (uvone.T)).T.reshape((self.height, self.width, 3))  # K^-1*uvone, size: H*W*3

        self.depth_maxrange = 10.0  # default=10.0, follow MIPI'22 depth completion challenge
        self.depth_minrange = 1e-3

        self.noise = 0.01
        self.cutmask = True

        self.aug = True

    def get_image_path(self, folder, modality, frame_index):

        f_str = "{:04d}{}".format(frame_index, self.img_ext)

        image_path = self.data_path + '/' + folder + '/' + modality + '/' + f_str
        # image_path = self.data_path + '/' + folder + '/rgb/' + modality  # Validset_fixed
        return image_path

    def get_depth_path(self, folder, modality, frame_index):
        img_ext = '.exr'
        f_str = "Image{:04d}{}".format(frame_index, img_ext)

        image_path = self.data_path + '/' + folder + '/' + modality + '/' + f_str
        # image_path = self.data_path + '/' + folder + '/' + modality + '/' + "{}{}".format(frame_index, img_ext) # Validset_fixed
        return image_path

    def get_color(self, folder, modality, frame_index, do_flip):
        color = self.loader_rgb(self.get_image_path(folder, modality, frame_index))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_depth(self, folder, modality, frame_index, do_flip):
        depth = self.loader_dep(self.get_depth_path(folder, modality, frame_index))
        depth = pil.fromarray(depth)

        if do_flip:
            # depth = cv2.flip(depth, 1)
            depth = depth.transpose(pil.FLIP_LEFT_RIGHT)

        return depth

    def get_sparse_depth_grid(self, dep):
        '''
        Simulate pincushion distortion:
        --stride:
        It controls the distance between neighbor spots7
        Suggest stride value:       5~10

        --dist_coef:
        It controls the curvature of the spot pattern
        Larger dist_coef distorts the pattern more.
        Suggest dist_coef value:    0 ~ 5e-5

        --noise:
        standard deviation of the spot shift
        Suggest noise value:        0 ~ 0.5
        '''

        dep = dep[np.newaxis, :, :]
        # Generate Grid points
        channel, img_h, img_w = dep.shape
        assert channel == 1

        stride = np.random.randint(5, 7)

        dist_coef = np.random.rand() * 4e-5 + 1e-5
        noise = np.random.rand() * 0.3

        x_odd, y_odd = np.meshgrid(np.arange(stride // 2, img_h, stride * 2), np.arange(stride // 2, img_w, stride))
        x_even, y_even = np.meshgrid(np.arange(stride // 2 + stride, img_h, stride * 2), np.arange(stride, img_w, stride))
        x_u = np.concatenate((x_odd.ravel(), x_even.ravel()))
        y_u = np.concatenate((y_odd.ravel(), y_even.ravel()))
        x_c = img_h // 2 + np.random.rand() * 50 - 25
        y_c = img_w // 2 + np.random.rand() * 50 - 25
        x_u = x_u - x_c
        y_u = y_u - y_c

        # Distortion
        r_u = np.sqrt(x_u ** 2 + y_u ** 2)
        r_d = r_u + dist_coef * r_u ** 3
        num_d = r_d.size
        sin_theta = x_u / r_u
        cos_theta = y_u / r_u
        x_d = np.round(r_d * sin_theta + x_c + np.random.normal(0, noise, num_d))
        y_d = np.round(r_d * cos_theta + y_c + np.random.normal(0, noise, num_d))
        idx_mask = (x_d < img_h) & (x_d > 0) & (y_d < img_w) & (y_d > 0)
        x_d = x_d[idx_mask].astype('int')
        y_d = y_d[idx_mask].astype('int')

        spot_mask = np.zeros((img_h, img_w))
        spot_mask[x_d, y_d] = 1

        dep_sp = np.zeros_like(dep)
        dep_sp[:, x_d, y_d] = dep[:, x_d, y_d]

        return dep_sp.squeeze(0)

    def cut_mask(self, dep, dep_aug):
        dep = dep[np.newaxis, :, :]
        dep_aug = dep_aug[np.newaxis, :, :]

        _, h, w = dep.shape
        c_x = np.random.randint(h / 4, h / 4 * 3)
        c_y = np.random.randint(w / 4, w / 4 * 3)
        r_x = np.random.randint(h / 4, h / 4 * 3)
        r_y = np.random.randint(h / 4, h / 4 * 3)

        mask = np.zeros_like(dep)
        min_x = max(c_x - r_x, 0)
        max_x = min(c_x + r_x, h)
        min_y = max(c_y - r_y, 0)
        max_y = min(c_y + r_y, w)
        mask[0, min_x:max_x, min_y:max_y] = 1

        return (dep * mask).squeeze(0), (dep_aug * mask).squeeze(0)

    def depth_plane2depth_world(self, imgDepthAbs):
        imgDepthAbs = imgDepthAbs[:, :, np.newaxis]
        xyz = self.PRJd * imgDepthAbs
        return xyz.reshape((-1, 3))

    def depth_world2depth_plane(self, points3d):
        prjuvz = (self.K_d @ points3d.T).T  # K @ (x,y,z)
        prjuvz[:, 0:2] = prjuvz[:, 0:2] / prjuvz[:, -1:]

        return prjuvz

    def rotate_world(self, points3d, R, T):
        # points3d = ATLX.R @ points3d.T
        T = T.reshape((3, 1))
        points3d = R @ points3d.T + T
        return points3d.T

    def rotate_depth(self, imgDepth, deg, T):
        imgDepth = np.reshape(imgDepth, [self.height, self.width])
        h, w = imgDepth.shape

        theta = np.deg2rad(deg)
        R = np.array([[1.0, 0.0, 0.0],
                      [0.0, np.cos(theta), np.sin(theta)],
                      [0.0, -np.sin(theta), np.cos(theta)]])

        T = np.array(T) / 1000

        points3d = self.depth_plane2depth_world(imgDepth)
        points3d = self.rotate_world(points3d, R, T)
        prjuvz = self.depth_world2depth_plane(points3d)

        # Finally, project back onto the RGB plane
        v = np.round(prjuvz[:, 1])
        u = np.round(prjuvz[:, 0])

        goodmask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        good_uvz = prjuvz[goodmask, :]

        # project to image
        depthOut = np.zeros((h, w)).astype(np.float32)
        depthOut[good_uvz[:, 1].astype(int), good_uvz[:, 0].astype(int)] = good_uvz[:, 2]

        # Fix weird values...
        depthOut[depthOut > self.depth_maxrange] = self.depth_maxrange
        depthOut[depthOut < 0] = 0

        return depthOut

    def get_occlusion_sp(self, dep, dep_sp):

        """
        To deal with occlusion issue, this function augments sparse depth map with 2d-3d-2d projection along X-axis
        :param dep: (H, W)
        :param dep_sp: (H, W)
        :return: aug_dep: augmented sparse depth; occlusion_depth for verifying
        """

        updown = np.random.uniform(0.0, 1.0)

        R_X = 10  # in degree
        T_X = np.random.randint(50, 70)  # in meter from 50 to 70
        ROLL_STEP = np.random.randint(5, 11)  # per pixel from 5 to 10

        if updown > 0.5:
            R_X = -R_X  # in degree
            T_X = -T_X  # in meter from 50 to 70
            ROLL_STEP = -ROLL_STEP  # per pixel from 5 to 10

        # 2d to 3d to 2d projection with specific R&T matrix
        rdep = self.rotate_depth(dep, deg=R_X, T=[0.0, -T_X, 0.0])
        rdep = self.rotate_depth(rdep, deg=-R_X, T=[0.0, T_X, 0.0])
        rdep = torch.max_pool2d(torch.tensor(rdep[None, :, :]), kernel_size=3, stride=1, padding=1).squeeze().numpy()
        rdep = np.sign(rdep)

        # Apply the same projection to full-ones * max depth matrix to get the bottom part mask. To do so, out of view region caused by projection will be extracted.
        rdown_part = self.rotate_depth(np.ones_like(dep) * self.depth_maxrange, deg=R_X, T=[0.0, -T_X, 0.0])
        rdown_part = self.rotate_depth(rdown_part, deg=-R_X, T=[0.0, T_X, 0.0])
        rdown_part = torch.max_pool2d(torch.tensor(rdown_part[None, :, :]), kernel_size=3, stride=1, padding=1).squeeze().numpy()
        rdown_part = np.sign(rdown_part)

        # Get occlusion mask
        mask = rdown_part - rdep

        occlusion_depth = mask * dep_sp
        occlusion_depth = np.roll(occlusion_depth, ROLL_STEP, axis=0)

        # blend original sparse depth and shifted sparse depth from occluded part
        _c = np.sign(dep_sp) * np.sign(occlusion_depth)
        aug_dep = np.minimum(dep_sp, occlusion_depth) * _c + np.maximum(dep_sp, occlusion_depth) * (1 - _c)

        return aug_dep, occlusion_depth
