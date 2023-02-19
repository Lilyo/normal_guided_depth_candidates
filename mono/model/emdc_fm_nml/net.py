from __future__ import absolute_import, division, print_function
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from .layers import SSIM, Backproject, Project
# from .depth_encoder import DepthEncoder
# from .depth_decoder import DepthDecoder
from .network_exp_emdc import EMDC
from .normal_net import normal_net

from .pose_encoder import PoseEncoder
from .pose_decoder import PoseDecoder
from .encoder import Encoder
from .decoder import Decoder
from ..registry import MONO

from .loss_zoo import EMDCLoss, MAELoss


@MONO.register_module
class emdc_fm_nml(nn.Module):
    def __init__(self, options):
        super(emdc_fm_nml, self).__init__()
        self.opt = options
        self.DepthNet = EMDC(self.opt.depth_pretrained_path)

        self.PoseEncoder = PoseEncoder(self.opt.pose_num_layers, self.opt.pose_pretrained_path)
        self.PoseDecoder = PoseDecoder(self.PoseEncoder.num_ch_enc)

        self.Encoder = Encoder(self.opt.rgb_num_layers, self.opt.rgb_pretrained_path)
        self.Decoder = Decoder(self.Encoder.num_ch_enc)

        self.ssim = SSIM()
        self.backproject = Backproject(self.opt.imgs_per_gpu, self.opt.height, self.opt.width)
        self.project = Project(self.opt.imgs_per_gpu, self.opt.height, self.opt.width)

        self.NormalNet = normal_net(pretrained=True, path=self.opt.normal_pretrained_path)

        self.endc_loss = EMDCLoss()
        self.mae_loss = MAELoss()

        self.width, self.height = 256, 192
        fx = 518.85790117450188
        fy = 519.46961112127485
        cx = 325.58244941119034 - 0.5
        cy = 253.73616633400465 - 0.5

        intrins = np.eye(3).astype(np.float32)
        intrins[0, 0] = fx
        intrins[1, 1] = fy
        intrins[0, 2] = cx
        intrins[1, 2] = cy
        self.pos = self.get_pos(intrins, self.width, self.height).cuda()

        self.ps = 5
        self.center_idx = (self.ps * self.ps - 1) // 2
        self.pad = (self.ps - 1) // 2

    def forward(self, inputs):
        outputs = {}
        outputs.update(self.DepthNet(inputs["depth_sp", 0, 0], inputs["color_aug", 0, 0], frame_id=0))

        if self.training:
            norm_out = self.NormalNet(inputs["color", 0, -1])
            pred_norm = norm_out[:, :3, :, :]
            # pred_kappa = norm_out[:, 3:, :, :]
            inputs['pred_norm_down'] = F.interpolate(pred_norm, size=(self.height, self.width), mode='bilinear', align_corners=False)
            norm = torch.sqrt(torch.sum(torch.square(inputs['pred_norm_down']), dim=1, keepdim=True))
            norm[norm < 1e-10] = 1e-10
            inputs['pred_norm_down'] = inputs['pred_norm_down'] / norm

            outputs.update(self.predict_poses(inputs))
            features = self.Encoder(inputs[("color", 0, 0)])  # Why did not apply norm?
            outputs.update(self.Decoder(features, 0))
            loss_dict = self.compute_losses(inputs, outputs, features)
            return outputs, loss_dict
        return outputs

    def forward_onnx(self, inputs_d, inputs_rgb):
        outputs = self.DepthNet(inputs_d, inputs_rgb, frame_id=0)
        return outputs[("depth", 0, 0)]

    def robust_l1(self, pred, target):
        eps = 1e-3
        return torch.sqrt(torch.pow(target - pred, 2) + eps ** 2)

    def compute_perceptional_loss(self, tgt_f, src_f):
        loss = self.robust_l1(tgt_f, src_f).mean(1, True)
        return loss

    def compute_reprojection_loss(self, pred, target):
        photometric_loss = self.robust_l1(pred, target).mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = (0.85 * ssim_loss + 0.15 * photometric_loss)
        return reprojection_loss

    def compute_losses(self, inputs, outputs, features):
        loss_dict = {}
        target = inputs[("color", 0, 0)]

        """
        surface loss
        : Ldis and Lcvt = get_feature_regularization_loss
        """
        # depth_weights (B, ps*ps, H, W)
        inputs['depth_candidate_weights'] = self.get_depth_candidate_weights(inputs)
        gth_depth_pad = F.pad(inputs[('depth', 0, 0)], pad=(self.pad, self.pad, self.pad, self.pad), mode='replicate')
        gth_depth_unfold = F.unfold(gth_depth_pad, [self.ps, self.ps], dilation=1, padding=0)  # (B, ps*ps, H*W)
        gth_depth_unfold = gth_depth_unfold.view(-1, self.ps * self.ps, self.height, self.width)  # (B, ps*ps, H, W)
        gth_depth_candidates = inputs['depth_candidate_weights'] * gth_depth_unfold  # (B, ps*ps, H, W)

        loc_depth_pad = F.pad(outputs[('y_loc', 0, 0)], pad=(self.pad, self.pad, self.pad, self.pad), mode='replicate')
        loc_depth_unfold = F.unfold(loc_depth_pad, [self.ps, self.ps], dilation=1, padding=0)  # (B, ps*ps, H*W)
        loc_depth_unfold = loc_depth_unfold.view(-1, self.ps * self.ps, self.height, self.width)  # (B, ps*ps, H, W)
        loc_depth_candidates = inputs['depth_candidate_weights'] * loc_depth_unfold  # (B, ps*ps, H, W)

        glb_depth_pad = F.pad(outputs[('y_glb', 0, 0)], pad=(self.pad, self.pad, self.pad, self.pad), mode='replicate')
        glb_depth_unfold = F.unfold(glb_depth_pad, [self.ps, self.ps], dilation=1, padding=0)  # (B, ps*ps, H*W)
        glb_depth_unfold = glb_depth_unfold.view(-1, self.ps * self.ps, self.height, self.width)  # (B, ps*ps, H, W)
        glb_depth_candidates = inputs['depth_candidate_weights'] * glb_depth_unfold  # (B, ps*ps, H, W)

        pred_depth_pad = F.pad(outputs[('depth', 0, 0)], pad=(self.pad, self.pad, self.pad, self.pad), mode='replicate')
        pred_depth_unfold = F.unfold(pred_depth_pad, [self.ps, self.ps], dilation=1, padding=0)  # (B, ps*ps, H*W)
        pred_depth_unfold = pred_depth_unfold.view(-1, self.ps * self.ps, self.height, self.width)  # (B, ps*ps, H, W)
        pred_depth_candidates = inputs['depth_candidate_weights'] * pred_depth_unfold  # (B, ps*ps, H, W)

        loss_dict[('loc_prop_loss', 0)] = self.mae_loss(loc_depth_candidates, gth_depth_candidates)
        loss_dict[('glb_prop_loss', 0)] = self.mae_loss(glb_depth_candidates, gth_depth_candidates)
        loss_dict[('depth_prop_loss', 0)] = self.mae_loss(pred_depth_candidates, gth_depth_candidates)

        """
        smooth loss
        : Ldis and Lcvt = get_feature_regularization_loss
        """
        for i in range(5):
            f = features[i]
            regularization_loss = self.get_feature_regularization_loss(f, target)
            loss_dict[('feature_regularization_loss', i)] = regularization_loss / (2 ** i) / 5

        """
        reconstruction
        :depth_reconstruct_loss = get_SiLogLoss
        """
        loss_dict[('depth_reconstruct_loss', 0)] = self.endc_loss(outputs[('depth', 0, 0)], outputs[('y_loc', 0, 0)], outputs[('y_glb', 0, 0)], inputs[('depth', 0, 0)])

        # DEPTH_AGU_IDS = [0]
        # for j in DEPTH_AGU_IDS[1:]:
        #     loss_dict[('depth_reconstruct_loss', 0)] += self.endc_loss(outputs[('depth', f'aug_{j}', 0)], outputs[('y_loc', f'aug_{j}', 0)], outputs[('y_glb', f'aug_{j}', 0)], inputs[('depth', 0, 0)])

        for scale in self.opt.scales:
            """
            initialization
            """
            # disp = outputs[("disp", 0, scale)]

            reprojection_losses = []
            perceptional_losses = []

            """
            autoencoder
            :img_reconstruct_loss = compute_reprojection_loss
            """
            res_img = outputs[("res_img", 0, scale)]
            _, _, h, w = res_img.size()
            target_resize = F.interpolate(target, [h, w], mode="bilinear", align_corners=False)
            img_reconstruct_loss = self.compute_reprojection_loss(res_img, target_resize)
            loss_dict[('img_reconstruct_loss', scale)] = img_reconstruct_loss.mean() / len(self.opt.scales)

            """
            reconstruction
            """
            outputs = self.generate_images_pred(inputs, outputs, scale)
            outputs = self.generate_features_pred(inputs, outputs)

            """
            automask
            :identity_reprojection_loss = compute_reprojection_loss
            """
            if self.opt.automask:
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, 0)]
                    identity_reprojection_loss = self.compute_reprojection_loss(pred, target)
                    identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 1e-5
                    reprojection_losses.append(identity_reprojection_loss)

            """
            minimum reconstruction loss
            :photometric_loss = compute_reprojection_loss
            """
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_loss = torch.cat(reprojection_losses, 1)

            min_reconstruct_loss, outputs[("min_index", scale)] = torch.min(reprojection_loss, dim=1)
            loss_dict[('min_reconstruct_loss', scale)] = min_reconstruct_loss.mean() / len(self.opt.scales)

            """
            minimum perceptional loss
            :Lfm = compute_perceptional_loss = compute_perceptional_loss
            """
            for frame_id in self.opt.frame_ids[1:]:
                src_f = outputs[("feature", frame_id, 0)]
                tgt_f = self.Encoder(inputs[("color", 0, 0)])[0]
                perceptional_losses.append(self.compute_perceptional_loss(tgt_f, src_f))
            perceptional_loss = torch.cat(perceptional_losses, 1)

            min_perceptional_loss, outputs[("min_index", scale)] = torch.min(perceptional_loss, dim=1)
            loss_dict[('min_perceptional_loss', scale)] = self.opt.perception_weight * min_perceptional_loss.mean() / len(self.opt.scales)

            # """
            # disp mean normalization
            # """
            # if self.opt.disp_norm:
            #     mean_disp = disp.mean(2, True).mean(3, True)
            #     disp = disp / (mean_disp + 1e-7)
            #
            # """
            # smooth loss for disp norm
            # """
            # smooth_loss = self.get_smooth_loss(disp, target)
            # loss_dict[('smooth_loss', scale)] = self.opt.smoothness_weight * smooth_loss / (2 ** scale) / len(self.opt.scales)

        return loss_dict

    def predict_poses(self, inputs):
        outputs = {}
        # [192,640] for kitti
        # x[72,96] ?[192,256] ![144,192] x[288,384] for mipi
        pose_feats = {f_i: F.interpolate(inputs["color_aug", f_i, 0], [192, 256], mode="bilinear", align_corners=False) for f_i in self.opt.frame_ids}
        for f_i in self.opt.frame_ids[1:]:
            if not f_i == "s":
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]
                pose_inputs = self.PoseEncoder(torch.cat(pose_inputs, 1))
                axisangle, translation = self.PoseDecoder(pose_inputs)
                outputs[("cam_T_cam", 0, f_i)] = self.transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
        return outputs

    def generate_images_pred(self, inputs, outputs, scale):
        # depth = outputs[("depth", 0, scale)]
        depth = torch.clamp(outputs[("depth", 0, 0)], min=1e-3, max=10)
        depth = F.interpolate(depth, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        for i, frame_id in enumerate(self.opt.frame_ids[1:]):
            if frame_id == "s":
                T = inputs["stereo_T"]
            else:
                T = outputs[("cam_T_cam", 0, frame_id)]
            cam_points = self.backproject(depth, inputs[("inv_K")])
            pix_coords = self.project(cam_points, inputs[("K")], T)  # [b,h,w,2]
            img = inputs[("color", frame_id, 0)]
            outputs[("color", frame_id, scale)] = F.grid_sample(img, pix_coords, padding_mode="border")

        return outputs

    def generate_features_pred(self, inputs, outputs):
        # depth = outputs[("depth", 0, 0)]
        depth = torch.clamp(outputs[("depth", 0, 0)], min=1e-3, max=10)
        depth = F.interpolate(depth, [int(self.opt.height / 2), int(self.opt.width / 2)], mode="bilinear", align_corners=False)
        for i, frame_id in enumerate(self.opt.frame_ids[1:]):
            if frame_id == "s":
                T = inputs["stereo_T"]
            else:
                T = outputs[("cam_T_cam", 0, frame_id)]

            backproject = Backproject(self.opt.imgs_per_gpu, int(self.opt.height / 2), int(self.opt.width / 2))
            project = Project(self.opt.imgs_per_gpu, int(self.opt.height / 2), int(self.opt.width / 2))

            K = inputs[("K")].clone()
            K[:, 0, :] /= 2
            K[:, 1, :] /= 2

            inv_K = torch.zeros_like(K)
            for i in range(inv_K.shape[0]):
                inv_K[i, :, :] = torch.pinverse(K[i, :, :])

            cam_points = backproject(depth, inv_K)
            pix_coords = project(cam_points, K, T)  # [b,h,w,2]

            img = inputs[("color", frame_id, 0)]
            src_f = self.Encoder(img)[0]
            outputs[("feature", frame_id, 0)] = F.grid_sample(src_f, pix_coords, padding_mode="border")
        return outputs

    def transformation_from_parameters(self, axisangle, translation, invert=False):
        R = self.rot_from_axisangle(axisangle)
        t = translation.clone()
        if invert:
            R = R.transpose(1, 2)
            t *= -1
        T = self.get_translation_matrix(t)
        if invert:
            M = torch.matmul(R, T)
        else:
            M = torch.matmul(T, R)
        return M

    def get_translation_matrix(self, translation_vector):
        T = torch.zeros(translation_vector.shape[0], 4, 4).cuda()
        t = translation_vector.contiguous().view(-1, 3, 1)
        T[:, 0, 0] = 1
        T[:, 1, 1] = 1
        T[:, 2, 2] = 1
        T[:, 3, 3] = 1
        T[:, :3, 3, None] = t
        return T

    def rot_from_axisangle(self, vec):
        angle = torch.norm(vec, 2, 2, True)
        axis = vec / (angle + 1e-7)
        ca = torch.cos(angle)
        sa = torch.sin(angle)
        C = 1 - ca
        x = axis[..., 0].unsqueeze(1)
        y = axis[..., 1].unsqueeze(1)
        z = axis[..., 2].unsqueeze(1)
        xs = x * sa
        ys = y * sa
        zs = z * sa
        xC = x * C
        yC = y * C
        zC = z * C
        xyC = x * yC
        yzC = y * zC
        zxC = z * xC
        rot = torch.zeros((vec.shape[0], 4, 4)).cuda()
        rot[:, 0, 0] = torch.squeeze(x * xC + ca)
        rot[:, 0, 1] = torch.squeeze(xyC - zs)
        rot[:, 0, 2] = torch.squeeze(zxC + ys)
        rot[:, 1, 0] = torch.squeeze(xyC + zs)
        rot[:, 1, 1] = torch.squeeze(y * yC + ca)
        rot[:, 1, 2] = torch.squeeze(yzC - xs)
        rot[:, 2, 0] = torch.squeeze(zxC - ys)
        rot[:, 2, 1] = torch.squeeze(yzC + xs)
        rot[:, 2, 2] = torch.squeeze(z * zC + ca)
        rot[:, 3, 3] = 1
        return rot

    def get_smooth_loss(self, disp, img):
        b, _, h, w = disp.size()
        a1 = 0.5
        a2 = 0.5
        img = F.interpolate(img, (h, w), mode='area')

        disp_dx, disp_dy = self.gradient(disp)
        img_dx, img_dy = self.gradient(img)

        disp_dxx, disp_dxy = self.gradient(disp_dx)
        disp_dyx, disp_dyy = self.gradient(disp_dy)

        img_dxx, img_dxy = self.gradient(img_dx)
        img_dyx, img_dyy = self.gradient(img_dy)

        smooth1 = torch.mean(disp_dx.abs() * torch.exp(-a1 * img_dx.abs().mean(1, True))) + \
                  torch.mean(disp_dy.abs() * torch.exp(-a1 * img_dy.abs().mean(1, True)))

        smooth2 = torch.mean(disp_dxx.abs() * torch.exp(-a2 * img_dxx.abs().mean(1, True))) + \
                  torch.mean(disp_dxy.abs() * torch.exp(-a2 * img_dxy.abs().mean(1, True))) + \
                  torch.mean(disp_dyx.abs() * torch.exp(-a2 * img_dyx.abs().mean(1, True))) + \
                  torch.mean(disp_dyy.abs() * torch.exp(-a2 * img_dyy.abs().mean(1, True)))

        return smooth1 + smooth2

    def gradient(self, D):
        D_dy = D[:, :, 1:] - D[:, :, :-1]
        D_dx = D[:, :, :, 1:] - D[:, :, :, :-1]
        return D_dx, D_dy

    def get_feature_regularization_loss(self, feature, img):
        b, _, h, w = feature.size()
        img = F.interpolate(img, (h, w), mode='area')

        feature_dx, feature_dy = self.gradient(feature)
        img_dx, img_dy = self.gradient(img)

        feature_dxx, feature_dxy = self.gradient(feature_dx)
        feature_dyx, feature_dyy = self.gradient(feature_dy)

        img_dxx, img_dxy = self.gradient(img_dx)
        img_dyx, img_dyy = self.gradient(img_dy)

        smooth1 = torch.mean(feature_dx.abs() * torch.exp(-img_dx.abs().mean(1, True))) + \
                  torch.mean(feature_dy.abs() * torch.exp(-img_dy.abs().mean(1, True)))

        smooth2 = torch.mean(feature_dxx.abs() * torch.exp(-img_dxx.abs().mean(1, True))) + \
                  torch.mean(feature_dxy.abs() * torch.exp(-img_dxy.abs().mean(1, True))) + \
                  torch.mean(feature_dyx.abs() * torch.exp(-img_dyx.abs().mean(1, True))) + \
                  torch.mean(feature_dyy.abs() * torch.exp(-img_dyy.abs().mean(1, True)))

        return -self.opt.dis * smooth1 + self.opt.cvt * smooth2

    def get_SiLogLoss(self, pred, target):
        lambd = 0.5
        valid_mask = (target > 0).detach()
        # diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        # loss = torch.sqrt(torch.pow(diff_log, 2).mean() - lambd * torch.pow(diff_log.mean(), 2))
        # return loss

        return torch.abs(target[valid_mask] - pred[valid_mask]).mean()

    def get_depth_candidate_weights(self, input_dict):
        with torch.no_grad():
            B, _, H, W = input_dict['pred_norm_down'].shape

            # pred norm down - nghbr
            pred_norm_down = F.pad(input_dict['pred_norm_down'], pad=(self.pad, self.pad, self.pad, self.pad), mode='replicate')
            pred_norm_down_unfold = F.unfold(pred_norm_down, [self.ps, self.ps], padding=0)  # (B, 3*ps*ps, H*W)
            pred_norm_down_unfold = pred_norm_down_unfold.view(B, 3, self.ps * self.ps, H, W)  # (B, 3, ps*ps, H, W)

            batch_pos = torch.cat(B * [self.pos])
            # pos down - nghbr
            pos_down_nghbr = F.pad(batch_pos, pad=(self.pad, self.pad, self.pad, self.pad), mode='replicate')
            pos_down_nghbr_unfold = F.unfold(pos_down_nghbr, [self.ps, self.ps], padding=0)  # (B, 2*ps*ps, H*W)
            pos_down_nghbr_unfold = pos_down_nghbr_unfold.view(B, 2, self.ps * self.ps, H, W)  # (B, 2, ps*ps, H, W)

            # norm and pos - nghbr
            nx, ny, nz = pred_norm_down_unfold[:, 0, ...], pred_norm_down_unfold[:, 1, ...], pred_norm_down_unfold[:, 2, ...]  # (B, ps*ps, H, W) or (B, 1, H, W)
            pos_u, pos_v = pos_down_nghbr_unfold[:, 0, ...], pos_down_nghbr_unfold[:, 1, ...]  # (B, ps*ps, H, W)

            # pos - center
            pos_u_center = pos_u[:, self.center_idx, :, :].unsqueeze(1)  # (B, 1, H, W)
            pos_v_center = pos_v[:, self.center_idx, :, :].unsqueeze(1)  # (B, 1, H, W)

            ddw_num = nx * pos_u + ny * pos_v + nz
            ddw_denom = nx * pos_u_center + ny * pos_v_center + nz
            ddw_denom[torch.abs(ddw_denom) < 1e-8] = torch.sign(ddw_denom[torch.abs(ddw_denom) < 1e-8]) * 1e-8

            ddw_weights = ddw_num / ddw_denom  # (B, ps*ps, H, W)
            ddw_weights[ddw_weights != ddw_weights] = 1.0  # nan
            ddw_weights[torch.abs(ddw_weights) == float("Inf")] = 1.0  # inf
        return ddw_weights

    def get_pos(self, intrins, W, H):

        pos = np.ones((H, W, 2))
        x_range = np.concatenate([np.arange(W).reshape(1, W)] * H, axis=0)
        y_range = np.concatenate([np.arange(H).reshape(H, 1)] * W, axis=1)
        pos[:, :, 0] = x_range + 0.5
        pos[:, :, 1] = y_range + 0.5
        pos[:, :, 0] = np.arctan((pos[:, :, 0] - intrins[0, 2]) / intrins[0, 0])
        pos[:, :, 1] = np.arctan((pos[:, :, 1] - intrins[1, 2]) / intrins[1, 1])
        pos = torch.from_numpy(pos.astype(np.float32)).permute(2, 0, 1)
        return pos.unsqueeze(0)
