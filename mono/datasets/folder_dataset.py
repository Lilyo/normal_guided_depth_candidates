from __future__ import absolute_import, division, print_function
import random
import numpy as np
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
import cv2


def cv2_loader(path):
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    return img


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class FolderDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """

    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 is_train=False,
                 img_ext='.jpg',
                 gt_depth_path=None
                 ):
        super(FolderDataset, self).__init__()
        self.data_path = data_path
        self.filenames = filenames
        self.resize_height = 288
        self.resize_width = 384
        self.height = height
        self.width = width
        self.frame_idxs = frame_idxs
        self.is_train = is_train
        self.img_ext = img_ext
        self.loader_rgb = pil_loader
        self.loader_dep = cv2_loader
        self.gt_depth_path = gt_depth_path
        self.to_tensor = transforms.ToTensor()
        self.to_numpy = self.ToNumpy()

        # Need to specify augmentations differently in pytorch 1.0 compared with 0.4
        if int(torch.__version__.split('.')[0]) > 0:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
        else:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        # self.t_rgb = transforms.Compose([
        #     transforms.Resize((self.height, self.width), interpolation=Image.ANTIALIAS),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])
        #
        # self.t_dep = transforms.Compose([
        #     transforms.Resize((self.height, self.width), interpolation=Image.NEAREST),
        #     transforms.ToTensor(),
        # ])

        # will be removed by self.t_rgb and t_dep
        self.resize = transforms.Resize((self.resize_height, self.resize_width), interpolation=Image.ANTIALIAS)
        self.resize_dep = transforms.Resize((self.resize_height, self.resize_width), interpolation=Image.NEAREST)
        self.norm_rgb = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.flag = np.zeros(self.__len__(), dtype=np.int64)

        if not is_train and self.gt_depth_path is not None:
            self.gt_depths = np.load(gt_depth_path,
                                     allow_pickle=True,
                                     fix_imports=True, encoding='latin1')["data"]

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """

        top = np.random.randint(0, self.resize_height - self.height)
        left = np.random.randint(0, self.resize_width - self.width)

        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                inputs[(n, im, 0)] = self.resize(inputs[(n, im, - 1)]).crop((left, top, left + self.width, top + self.height))
            if "depth" in k:
                n, im, i = k
                inputs[(n, im, 0)] = self.resize_dep(inputs[(n, im, - 1)]).crop((left, top, left + self.width, top + self.height))

        for k in list(inputs):
            if "color" in k:
                f = inputs[k]
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)

                if i == 0:
                    inputs[(n + "_aug", im, i)] = self.norm_rgb(self.to_tensor(color_aug(f)))

            if "depth" in k:
                f = inputs[k]
                n, im, i = k
                f = self.to_numpy(f)
                f[f > self.depth_maxrange] = 0
                f[f < self.depth_minrange] = 0
                inputs[(n, im, i)] = self.to_tensor(f)

                if im == 0 and i == 0:

                    # Foreshortening effects
                    if self.noise:
                        reflection = np.clip(np.random.normal(1, scale=0.333332, size=(1, 1)), 0.01, 3)[0, 0]
                        noise = np.random.normal(loc=0.0, scale=f * reflection * self.noise, size=None)
                        dep_noise = f + noise
                        dep_noise[dep_noise < 0] = 0
                    else:
                        dep_noise = f

                    dep_sp = self.get_sparse_depth_grid(dep_noise)

                    # if self.aug:
                    #     flip = np.random.uniform(0.0, 1.0)
                    #     if flip > 0.5:
                    #         dep_sp, occlusion_depth = self.get_occlusion_sp(dep_noise, dep_sp)

                    if self.aug:
                        dep_sp_aug, _ = self.get_occlusion_sp(dep_noise, dep_sp)

                    if self.cutmask:
                        dep_sp, dep_sp_aug = self.cut_mask(dep_sp, dep_sp_aug)

                    inputs[(n + "_sp", im, i)] = self.to_tensor(dep_sp)
                    inputs[(n + "_sp_aug", im, i)] = self.to_tensor(dep_sp_aug)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        if not self.is_train and self.gt_depth_path is not None:
            gt_depth = self.gt_depths[index]
            inputs['gt_depth'] = gt_depth

        # print(line[0])
        # print(line[1])

        # RGB
        folder = line[0].split('/')[0]
        modality_rgb = line[0].split('/')[1]
        frame_index = int(line[0].split('/')[-1].split('.')[0])

        # Depth
        modality_depth = line[1].split('/')[1]

        for i in self.frame_idxs:
            try:
                inputs[("color", i, -1)] = self.get_color(folder, modality_rgb, frame_index + i, do_flip)
                inputs[("depth", i, -1)] = self.get_depth(folder, modality_depth, frame_index + i, do_flip)
            except:
                inputs[("color", i, -1)] = self.get_color(folder, modality_rgb, frame_index, do_flip)
                inputs[("depth", i, -1)] = self.get_depth(folder, modality_depth, frame_index, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        K = self.K.copy()
        K[0, :] *= self.width
        K[1, :] *= self.height
        inv_K = np.linalg.pinv(K)

        inputs[("K")] = torch.from_numpy(K)
        inputs[("inv_K")] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            if i != 0:
                del inputs[("color", i, -1)]
            del inputs[("depth", i, -1)]

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_pose(self, folder, frame_index, offset):
        return

    class ToNumpy:
        def __call__(self, sample):
            return np.array(sample).astype(np.float)
