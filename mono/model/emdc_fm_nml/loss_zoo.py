import torch
import torch.nn as nn
import torch.nn.functional as F


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, outputs, target, *args):
        val_pixels = torch.ne(target, 0).float().cuda()
        loss = target * val_pixels - outputs * val_pixels
        return torch.sum(torch.abs(loss)) / torch.sum(val_pixels)


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, outputs, target, *args):
        val_pixels = torch.ne(target, 0).float().cuda()
        loss = target * val_pixels - outputs * val_pixels
        return torch.sum(loss ** 2) / torch.sum(val_pixels)


class TwinLoss(nn.Module):
    def __init__(self):
        super(TwinLoss, self).__init__()
        self.gramma = 2

    def forward(self, outputs, target, *args):
        """
        output: A tensor with shape [B, 3, H, W]
        target: A tensor with shape [B, 1, H, W]
        """
        val_pixels = torch.ne(target, 0).float().cuda()

        split_deps = torch.split(outputs, 1, 1)
        split_deps = list(split_deps)

        fg_depth = split_deps[0]
        bg_depth = split_deps[1]
        sigma = torch.sigmoid(split_deps[2])

        fg_err = (fg_depth - target) * val_pixels
        bg_err = (bg_depth - target) * val_pixels
        ale = torch.max((-1 / self.gramma) * fg_err, self.gramma * fg_err)
        rale = torch.max((1 / self.gramma) * bg_err, -self.gramma * bg_err)
        fused_depth = torch.abs(
            sigma * fg_depth * val_pixels + (1 - sigma) * bg_depth * val_pixels - target * val_pixels)

        # ambiguities_consistancy_err = torch.max(torch.zeros_like(fg_depth), (fg_depth - bg_depth) * val_pixels)

        # entropy_regularization = - sigma * torch.log(sigma) * val_pixels

        loss = ale + rale + fused_depth
        return torch.sum(loss) / torch.sum(val_pixels)


class EMDCLoss(nn.Module):
    def __init__(self):
        super(EMDCLoss, self).__init__()
        self.criterion1 = L1Loss()
        self.criterion1_ = BerhuLoss()
        self.criterion1__ = BMCLoss(init_noise_sigma=1., device='cuda')
        self.criterion2 = L2Loss()
        self.criterion3 = RMAEloss()
        self.criterion4 = GradientLoss()
        self.criterion5 = Sparse_Loss()

        self.losses = ['criterion1', 'criterion4']

    def forward(self, output, y_loc, y_glb, target):
        """
        output: A tensor with shape [B, 1, H, W]
        target: A tensor with shape [B, 1, H, W]

        Regarding to criterion1:
        loss = loss + loss_sam1*(loss1/loss_sam1).detach().
        The purpose is to make loss_sam1 and loss_sam2 the same scale, even the same value here.
        This way, the global branch1 and local branch of their back-propagation are learned with a balanced step.
        Avoid model mismatch in the middle of training, which will lead to unstable training of the fusion module.
        """
        loss = 0
        if 'criterion1' in self.losses:
            loss1 = self.criterion1(output, target)
            loss = loss1
            loss_sam1 = self.criterion1(y_loc, target)
            loss += loss_sam1 / (loss_sam1 / loss1).detach()
            loss_sam2 = self.criterion1(y_glb, target)
            loss += loss_sam2 / (loss_sam2 / loss1).detach()

        if 'criterion2' in self.losses:
            loss2 = self.criterion2(output, target)
            loss += loss2

        if 'criterion3' in self.losses:
            loss3 = self.criterion3(output, target)
            loss += loss3

        if 'criterion4' in self.losses:
            loss4 = self.criterion4(output, target)
            loss += 0.7 * loss4 / (loss4 / loss1).detach()  # default 0.2

        return loss


class GradientLoss(nn.Module):

    def __init__(self):
        super(GradientLoss, self).__init__()
        self.depth_valid1 = 0.001

    def forward(self, sr, hr):
        mask1 = (hr > self.depth_valid1).type_as(sr).detach()

        km = torch.Tensor([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]]).view(1, 1, 3, 3).to(hr)
        # km = torch.Tensor([[1, 1, 1, 1, 1],
        #                    [1, 1, 1, 1, 1],
        #                    [1, 1, 1, 1, 1],
        #                    [1, 1, 1, 1, 1],
        #                    [1, 1, 1, 1, 1]]).view(1, 1, 5, 5).to(hr)

        kx = torch.Tensor([[1, 0, -1], [2, 0, -2],
                           [1, 0, -1]]).view(1, 1, 3, 3).to(hr)
        ky = torch.Tensor([[1, 2, 1], [0, 0, 0],
                           [-1, -2, -1]]).view(1, 1, 3, 3).to(hr)

        erode = F.conv2d(mask1, km, padding=1)
        # erode = F.conv2d(mask1, km, padding=2)
        mask1_erode = (erode == 9).type_as(sr).detach()
        # mask1_erode = (erode == 25).type_as(sr).detach()
        pred_grad_x = F.conv2d(sr, kx, padding=1)
        pred_grad_y = F.conv2d(sr, ky, padding=1)
        target_grad_x = F.conv2d(hr, kx, padding=1)
        target_grad_y = F.conv2d(hr, ky, padding=1)

        d = torch.abs(pred_grad_x - target_grad_x) * mask1_erode
        d = torch.sum(d, dim=[1, 2, 3])
        num_valid = torch.sum(mask1_erode, dim=[1, 2, 3])
        loss_x = d / (num_valid + 1e-8)

        d = torch.abs(pred_grad_y - target_grad_y) * mask1_erode
        d = torch.sum(d, dim=[1, 2, 3])
        num_valid = torch.sum(mask1_erode, dim=[1, 2, 3])
        loss_y = d / (num_valid + 1e-8)

        loss = loss_x.mean() + loss_y.mean()
        return loss


class Sparse_Loss(nn.Module):
    def __init__(self):
        super(Sparse_Loss, self).__init__()

        # self.args = args
        self.depth_valid1 = 0.001
        # self.depth_valid2 = 20

    def forward(self, pred, depsp, gt):
        zeros = torch.zeros(depsp.size(), device=depsp.device)
        pred_ = torch.where(depsp > 0.001, pred, zeros)
        depsp_ = torch.where(depsp > 0.001, gt, zeros)
        error = torch.abs(pred_ - depsp_)
        loss = torch.mean(error)
        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

        # self.args = args
        self.depth_valid1 = 0.001
        # self.depth_valid2 = 20

    def forward(self, pred, gt):
        mask1 = (gt > self.depth_valid1).type_as(pred).detach()
        # mask2 = (gt < self.depth_valid2).type_as(pred).detach()

        d = torch.abs(pred - gt) * mask1  # * mask2

        d = torch.sum(d, dim=[1, 2, 3])
        num_valid = torch.sum(mask1, dim=[1, 2, 3])

        loss = d / (num_valid + 1e-8)

        loss = loss.mean()  # batch mean

        return loss


class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-9

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


from torch.distributions import MultivariateNormal as MVN


class BMCLoss(nn.Module):
    def __init__(self, init_noise_sigma, device):
        super(BMCLoss, self).__init__()
        self.device = device
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device=device))

    def bmc_loss_md(self, pred, target, noise_var):
        """Compute the Multidimensional Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
        Args:
        pred: A float tensor of size [batch, d].
        target: A float tensor of size [batch, d].
        noise_var: A float number or tensor.
        Returns:
        loss: A float tensor. Balanced MSE Loss.
        """
        I = torch.eye(pred.shape[-1], device=self.device)
        logits = MVN(pred.unsqueeze(1), noise_var * I).log_prob(target.unsqueeze(0))  # logit size: [batch, batch]
        loss = F.cross_entropy(logits, torch.arange(pred.shape[0], device=self.device))  # contrastive-like loss
        loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable

        return loss

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return self.bmc_loss_md(pred, target, noise_var)


class BerhuLoss(nn.Module):
    def __init__(self, delta=0.05):
        super(BerhuLoss, self).__init__()
        self.delta = delta

    def forward(self, prediction, gt):
        err = torch.abs(prediction - gt)
        mask = (gt > 0.001).detach()
        err = torch.abs(err[mask])
        c = self.delta * err.max().item()
        squared_err = (err ** 2 + c ** 2) / (2 * c)
        linear_err = err
        return torch.mean(torch.where(err > c, squared_err, linear_err))


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

        self.depth_valid1 = 0.001

    def forward(self, pred, gt):
        mask1 = (gt > self.depth_valid1).type_as(pred).detach()

        d = torch.pow(pred - gt, 2) * mask1  # * mask2

        d = torch.sum(d, dim=[1, 2, 3])
        num_valid = torch.sum(mask1, dim=[1, 2, 3])

        loss = d / (num_valid + 1e-8)

        loss = loss.sum()

        return loss


class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-9

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class RMAEloss(nn.Module):
    def __init__(self, reduction='mean'):
        super(RMAEloss, self).__init__()
        self.reduction = reduction

        self.depth_valid1 = 0.15
        self.depth_valid2 = 20

    def forward(self, pred_dep, gt):

        ''' If forget to add parentheses to (gt+1e-6), it will cause nan loss
        mask1 = (gt > self.depth_valid1).type_as(pred_dep).detach()
        mask2 = (gt < self.depth_valid2).type_as(pred_dep).detach()
        loss = torch.abs((pred_dep-gt)/(gt+1e-6)) * mask1 * mask2
        '''
        loss = torch.abs((pred_dep[gt > 0.15] - gt[gt > 0.15]) / gt[gt > 0.15])

        if self.reduction == 'mean':
            rmae = torch.mean(loss)
        else:
            rmae = torch.sum(loss)
        return rmae
