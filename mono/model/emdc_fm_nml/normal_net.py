import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


# architecture introduced in
# Estimating and Exploiting the Aleatoric Uncertainty in Surface Normal Estimation (Bae et al. ICCV 21)
class EESNU(nn.Module):
    def __init__(self, BN=True):
        super(EESNU, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(BN=BN)

    def forward(self, x, **kwargs):
        return self.decoder(self.encoder(x), **kwargs)

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder]
        for m in modules:
            yield from m.parameters()


class Decoder(nn.Module):
    def __init__(self, BN):
        super(Decoder, self).__init__()
        self.min_kappa = 0.01

        # hyper-parameter for sampling
        self.sampling_ratio = 0.4
        self.importance_ratio = 0.7

        # feature-map
        self.conv2 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0)

        UpSample = UpSampleBN if BN else UpSampleGN
        self.up1 = UpSample(skip_input=2048 + 176, output_features=1024)
        self.up2 = UpSample(skip_input=1024 + 64, output_features=512)
        self.up3 = UpSample(skip_input=512 + 40, output_features=256)
        self.up4 = UpSample(skip_input=256 + 24, output_features=128)

        # produces 1/8 res output
        self.out_conv_res8 = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)

        # produces 1/4 res output
        self.out_conv_res4 = nn.Sequential(
            nn.Conv1d(512 + 4, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 4, kernel_size=1),
        )

        # produces 1/2 res output
        self.out_conv_res2 = nn.Sequential(
            nn.Conv1d(256 + 4, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 4, kernel_size=1),
        )

        # produces 1/1 res output
        self.out_conv_res1 = nn.Sequential(
            nn.Conv1d(128 + 4, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 4, kernel_size=1),
        )

    # normalize
    def norm_normalize(self, out):
        norm_x, norm_y, norm_z, kappa = torch.split(out, 1, dim=1)
        norm = torch.sqrt(norm_x ** 2.0 + norm_y ** 2.0 + norm_z ** 2.0) + 1e-10
        kappa = F.elu(kappa) + 1.0 + self.min_kappa
        return torch.cat([norm_x / norm, norm_y / norm, norm_z / norm, kappa], dim=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]

        # generate feature-map
        x_d0 = self.conv2(x_block4)  # x_d0 : [2, 2048, 15, 20]      1/32 res
        x_d1 = self.up1(x_d0, x_block3)  # x_d1 : [2, 1024, 30, 40]      1/16 res
        x_d2 = self.up2(x_d1, x_block2)  # x_d2 : [2, 512, 60, 80]       1/8 res
        x_d3 = self.up3(x_d2, x_block1)  # x_d3: [2, 256, 120, 160]      1/4 res
        x_d4 = self.up4(x_d3, x_block0)  # x_d4: [2, 128, 240, 320]      1/2 res

        # 1/8 res output
        out_res8 = self.out_conv_res8(x_d2)  # out_res8: [2, 4, 60, 80]      1/8 res output
        out_res8 = self.norm_normalize(out_res8)  # out_res8: [2, 4, 60, 80]      1/8 res output

        # 1/4 res output
        feat_map = F.interpolate(x_d2, scale_factor=2, mode='bilinear', align_corners=True)
        init_pred = F.interpolate(out_res8, scale_factor=2, mode='bilinear', align_corners=True)
        feat_map = torch.cat([feat_map, init_pred], dim=1)  # (B, 512+4, H, W)
        B, _, H, W = feat_map.shape

        out_res4 = self.out_conv_res4(feat_map.view(B, 512 + 4, -1))  # (B, 4, N)
        out_res4 = self.norm_normalize(out_res4)  # (B, 4, N) - normalized
        out_res4 = out_res4.view(B, 4, H, W)

        # 1/2 res output
        feat_map = F.interpolate(x_d3, scale_factor=2, mode='bilinear', align_corners=True)
        init_pred = F.interpolate(out_res4, scale_factor=2, mode='bilinear', align_corners=True)
        feat_map = torch.cat([feat_map, init_pred], dim=1)  # (B, 512+4, H, W)
        B, _, H, W = feat_map.shape

        out_res2 = self.out_conv_res2(feat_map.view(B, 256 + 4, -1))  # (B, 4, N)
        out_res2 = self.norm_normalize(out_res2)  # (B, 4, N) - normalized
        out_res2 = out_res2.view(B, 4, H, W)

        # 1/1 res output
        # grid_sample feature-map
        feat_map = F.interpolate(x_d4, scale_factor=2, mode='bilinear', align_corners=True)
        init_pred = F.interpolate(out_res2, scale_factor=2, mode='bilinear', align_corners=True)
        feat_map = torch.cat([feat_map, init_pred], dim=1)  # (B, 512+4, H, W)
        B, _, H, W = feat_map.shape

        out_res1 = self.out_conv_res1(feat_map.view(B, 128 + 4, -1))  # (B, 4, N)
        out_res1 = self.norm_normalize(out_res1)  # (B, 4, N) - normalized
        out_res1 = out_res1.view(B, 4, H, W)

        return out_res1


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        basemodel_name = 'tf_efficientnet_b5_ap'
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
        # Remove last layer
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()
        self.original_model = basemodel

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()
        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class Conv2d_WS(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_WS, self).__init__(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class UpSampleGN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleGN, self).__init__()
        self._net = nn.Sequential(Conv2d_WS(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.GroupNorm(8, output_features),
                                  nn.LeakyReLU(),
                                  Conv2d_WS(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.GroupNorm(8, output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class NNET(nn.Module):
    def __init__(self):
        super(NNET, self).__init__()
        self.min_kappa = 0.01
        self.output_dim = 1
        self.output_type = 'G'
        self.NNET_architecture = 'GN'

        if self.NNET_architecture == 'BN':
            self.n_net = EESNU(BN=True)
        else:
            self.n_net = EESNU(BN=False)

    def forward(self, img, **kwargs):
        return self.n_net(img, **kwargs)


def normal_net(pretrained=True, path=None):
    model = NNET()
    if pretrained:
        print('loading N-Net weights from %s' % path)
        model = load_checkpoint(path, model)

    return model


def load_checkpoint(fpath, model):
    ckpt = torch.load(fpath, map_location='cpu')['model']
    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    model.load_state_dict(load_dict)

    for param in model.parameters():
        param.requires_grad = False
    return model


def normal_to_rgb(norm):
    norm_rgb = ((norm + 1) * 0.5) * 255
    norm_rgb = np.clip(norm_rgb, a_min=0, a_max=255)
    norm_rgb = norm_rgb.astype(np.uint8)  # (B, H, W, 3)
    return norm_rgb


def kappa_to_alpha(pred_kappa):
    import numpy as np
    alpha = ((2 * pred_kappa) / ((pred_kappa ** 2.0) + 1)) \
            + ((np.exp(- pred_kappa * np.pi) * np.pi) / (1 + np.exp(- pred_kappa * np.pi)))
    alpha = np.degrees(alpha)
    return alpha


def util_draw_depth(depth_map, image, max_depth=2, alpha=0.5):
    """
    Author: Arthur Wu, 25.08.22
    Description: Weight or concat two images
    Args:
        depth_map: A ndarray with shape (H, W), depth in meter
        image: A ndarray with shape (H, W, 3), color in uint8
        max_depth: Limit depth of your sensor
        alpha: Transparency

    Returns: A ndarray with shape (H, W, 3) in uint8 if alpha is non-zero while shape (H, 2*W, 3) in uint8 if alpha is zero
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().data.cpu().numpy().squeeze()

    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.detach().data.cpu().numpy().squeeze()

    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)

    if image.dtype == np.float32:
        image = image * np.array([0.229, 0.224, 0.225]).reshape(1, 1, -1) + np.array([0.485, 0.456, 0.406]).reshape(1, 1, -1)
        image = image * 256

    # Normalize estimated depth to color it
    if max_depth:
        min_depth = 0
    else:
        min_depth = depth_map.min()
        max_depth = depth_map.max()

    norm_depth_map = 255 * (depth_map - min_depth) / (max_depth - min_depth)
    norm_depth_map[norm_depth_map < 0] = 0
    norm_depth_map[norm_depth_map >= 255] = 255

    # Normalize and color the image
    color_depth = cv2.applyColorMap(cv2.convertScaleAbs(norm_depth_map, 1), cv2.COLORMAP_TURBO)  # cv2.COLORMAP_PLASMA

    # Resize to match the image shape
    color_depth = cv2.resize(color_depth, (depth_map.shape[1], depth_map.shape[0]))

    # Fuse both images
    if (alpha == 0):
        combined_img = np.hstack((image, color_depth))
    else:
        combined_img = cv2.addWeighted(image, alpha, color_depth, (1 - alpha), 0)

    return combined_img


if __name__ == '__main__':
    import os

    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # set the running GPU

    NNET_ckpt = '/home/arthur/workspace/Projects/depthcompletiontoolbox/weights/normal_nyuv2.pt'
    model = normal_net(path=NNET_ckpt)
    model.cuda()

    import cv2
    from PIL import Image
    import torchvision.transforms as T

    cv2_img = cv2.imread('/home/arthur/workspace/Projects/depthcompletiontoolbox/assets/0000000022.png')
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    cv2_img = cv2.resize(cv2_img, (640, 480), interpolation=cv2.INTER_LINEAR)
    # cv2_img = cv2.resize(cv2_img, (256, 192), interpolation=cv2.INTER_LINEAR)
    print(cv2_img.shape)
    # from nparray to PIL
    rgb = Image.fromarray(cv2_img)

    t_rgb = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    rgb = t_rgb(rgb).unsqueeze(0).cuda()
    norm_out = model(rgb)
    # surface normal prediction
    pred_norm = norm_out[:, :3, :, :].detach().cpu().permute(0, 2, 3, 1).numpy().squeeze()
    pred_kappa = norm_out[:, 3:, :, :].detach().cpu().permute(0, 2, 3, 1).numpy().squeeze()

    norm_rgb = normal_to_rgb(pred_norm)

    # norm_rgb = cv2.resize(norm_rgb, (256, 192))

    # surface normal uncertainty
    pred_kappa = kappa_to_alpha(pred_kappa)

    vis = util_draw_depth(pred_kappa, norm_rgb, max_depth=np.max(pred_kappa), alpha=0)

    cv2.imwrite('normal.png', vis, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    cv2.imshow("Estimated depth", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
