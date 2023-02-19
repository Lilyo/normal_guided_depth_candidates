from __future__ import absolute_import, division, print_function
import os
import cv2
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from mmcv import Config
import argparse

import torch

sys.path.append('.')
sys.path.append('..')
from mono.model.registry import MONO
from fvcore.nn import FlopCountAnalysis, parameter_count_table

from PIL import Image
import torchvision.transforms as T
from mpl_toolkits.axes_grid1 import ImageGrid

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # set the running GPU

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

MAX_DEPTH = 20.0


def plt_visualize_emdc(rgb, depth, depsp, y_glb, y_loc, jet_path):
    from vis_utils import util_draw_depth, util_draw_heatmap
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    img = util_draw_depth(depsp, rgb, max_depth=np.max(depth), alpha=0)
    img = util_draw_depth(depth, img, max_depth=np.max(depth), alpha=0)

    img = util_draw_depth(cv2.resize(y_glb, (rgb.shape[1], rgb.shape[0])), img, max_depth=np.max(depth), alpha=0)
    img = util_draw_depth(cv2.resize(y_loc, (rgb.shape[1], rgb.shape[0])), img, max_depth=np.max(depth), alpha=0)

    cv2.imwrite(jet_path, img)



def plt_visualize(rgb, depth, depsp, jet_path):
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth[depth > MAX_DEPTH] = 0
    vmin, vmax = depth.min(), depth.max()
    spot_depth = depsp

    x_sp, y_sp = np.where(spot_depth > 0)
    d_sp = spot_depth[x_sp, y_sp]

    title_names = ['RGB', 'Predict', 'spot depth']
    fig = plt.figure(figsize=(14, 4))
    axs = ImageGrid(fig, 111,
                    nrows_ncols=(1, 3),
                    axes_pad=0.05,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="5%",
                    cbar_pad=0.05
                    )
    axs[0].imshow(rgb)
    axs[1].imshow(depth, cmap='jet_r', vmin=vmin, vmax=vmax)
    imc = axs[2].scatter(y_sp, x_sp, np.ones_like(x_sp) * 0.1, c=d_sp, cmap='jet_r', vmin=vmin, vmax=vmax)
    axs[2].axis([0, 256, 192, 0])
    asp = abs(np.diff(axs[2].get_xlim())[0] / np.diff(axs[2].get_ylim())[0]) * 192 / 256
    axs[2].set_aspect(asp)
    for ii, title_name in enumerate(title_names):
        axs[ii].set_title(title_name, fontsize=12)
        axs[ii].set_xticks([])
        axs[ii].set_yticks([])
    cbar = plt.colorbar(imc, cax=axs.cbar_axes[0], ticks=np.linspace(vmin, vmax, 5), format='%.1f')
    cbar.ax.set_ylabel('Depth (m)')

    plt.savefig(jet_path)
    plt.close('all')


mean_tensor = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
var_tensor = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()


def predict(rgb, dep_sp, model):
    with torch.no_grad():
        input = {}
        input['color_aug', 0, 0] = rgb
        input['depth_sp', 0, 0] = dep_sp
        outputs = model(input)
    return outputs


def evaluate(cfg_path, model_path, txt_path, out_path, visualization=False):
    if not os.path.exists(out_path): os.makedirs(out_path)

    class ToNumpy:
        def __call__(self, sample):
            return np.array(sample)

    with open(txt_path, 'r') as fh:
        pairs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            pairs.append((txt_path[:-9] + words[0], txt_path[:-9] + words[1]))

    cfg = Config.fromfile(cfg_path)
    cfg['model']['depth_pretrained_path'] = '/home/arthur/workspace/Projects/FeatDepth/weights/mobilenetv2.pth.tar'
    cfg['model']['pose_pretrained_path'] = None
    cfg['model']['extractor_pretrained_path'] = None
    model = MONO.module_dict[cfg.model['name']](cfg.model)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.cuda()
    model.eval()

    with torch.no_grad():
        cost_time = 0
        flist = []
        for pair in pairs:
            print(pair)

            rgb_path, depsp_path = pair

            bgr = cv2.imread(rgb_path)
            np_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            np_dep = cv2.imread(depsp_path, cv2.IMREAD_ANYDEPTH)

            # from nparray to PIL
            rgb = Image.fromarray(np_rgb)
            dep = np_dep.astype(np.float32)
            dep[dep > MAX_DEPTH] = 0
            dep = Image.fromarray(dep)

            t_rgb = T.Compose([
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            t_dep = T.Compose([
                ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb).unsqueeze(0).cuda()
            dep_sp = t_dep(dep).unsqueeze(0).cuda()

            torch.cuda.synchronize()
            begin = time.time()

            outputs = predict(rgb, dep_sp, model)

            torch.cuda.synchronize()
            end = time.time()
            cost_time += end - begin

            output = outputs[("depth", 0, 0)].squeeze().cpu().numpy()
            y_glb = outputs[("y_glb", 0, 0)].squeeze().cpu().numpy()
            y_loc = outputs[("y_loc", 0, 0)].squeeze().cpu().numpy()

            # output dense depth image
            exr_name = depsp_path.split('/')[-2] + '_' + depsp_path.split('/')[-1]
            # exr_name = exr_name.replace('exr', 'png')

            output_path = os.path.join(f'{out_path}', exr_name)

            cv2.imwrite(output_path, output)
            # cv2.imwrite(output_path, np.round(output, 3))

            if visualization:
                jet_path_o = output_path[:-4] + '_jet' + '.png'
                # plt_visualize(np_rgb, output, np_dep, jet_path_o)
                # plt_visualize_spm(np_rgb, output, np_dep, y_glb, y_loc, y_ds1, y_ds2, y_ds3, y_cs1, y_cs2, y_cs3, dep1, dep2, dep3, jet_path_o)
                plt_visualize_emdc(np_rgb, output, np_dep, y_glb, y_loc, jet_path_o)
            flist.append(exr_name)

        with open(f'./{out_path}/data.list', 'w') as f:
            for item in flist:
                f.write("%s\n" % item)

        print(f"done! costtime {cost_time} for {len(pairs)} frames")
        print(f"{cost_time * 1000 / len(pairs)} ms per frames")

        delattr(model, 'PoseDecoder')
        delattr(model, 'PoseEncoder')
        delattr(model, 'Encoder')
        delattr(model, 'Decoder')
        delattr(model, 'NormalNet')
        delattr(model, 'backproject')
        delattr(model, 'endc_loss')
        delattr(model, 'mae_loss')
        delattr(model, 'project')
        delattr(model, 'pos')
        delattr(model, 'ssim')

        pytorch_total_params = sum(p.numel() for p in model.parameters())
        with open(f'./{out_path}/readme.txt', 'w') as f:
            f.write(f"Runtime per image [s] : {cost_time / len(pairs)}\n")
            f.write(f"Parameters : {pytorch_total_params}\n")
            f.write(f"Extra Data [1] / No Extra Data [0] : 0\n")
            f.write(f"Other description : GPU: RTX3090; Pretraind model: from https://github.com/tonylins/pytorch-mobilenet-v2, https://github.com/baegwangbin/surface_normal_uncertainty")


def get_FLOPs(cfg_path, model_path):
    cfg = Config.fromfile(cfg_path)
    cfg['model']['depth_pretrained_path'] = '/home/arthur/workspace/Projects/depthcompletiontoolbox/weights/mobilenetv2.pth.tar'
    cfg['model']['pose_pretrained_path'] = None
    cfg['model']['extractor_pretrained_path'] = None
    model = MONO.module_dict[cfg.model['name']](cfg.model)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.cuda()
    model.eval()

    height, width = (192, 256)  # MIPI

    rgb = np.random.random(size=(1, 3, height, width))
    rgb = rgb.astype(np.float32)
    depth = np.random.random(size=(1, 1, height, width))
    depth = depth.astype(np.float32)

    rgb = torch.from_numpy(rgb)
    depth = torch.from_numpy(depth)

    rgb = rgb.cuda()
    depth = depth.cuda()

    tensor = (depth, rgb, 0, True)

    # FLOPs
    flops = FlopCountAnalysis(model.DepthNet, tensor)
    print('FLOPs = ' + str(flops.total() / 1000 ** 3) + 'G')
    print(parameter_count_table(model))


if __name__ == "__main__":

    # retrieve necessary hyper-parameters
    parser = argparse.ArgumentParser()

    for data_type in ['synthetic', 'iPhone_static', 'iPhone_dynamic', 'modified_phone_static']:
    define model and log path
    parser.add_argument("--cfg_path", default=None, help="The path of cfg file.")
    parser.add_argument("--ckp_path", default=None, help="The path of ckp file.")
    parser.add_argument("--txt_path", default=None, help="The path of test.txt.")
    parser.add_argument("--data_type", default=None, help="The folder name.")
    parser.add_argument("--vis", action='store_true', help="vis as jpg")
    parser.add_argument("--get_FLOPs", default=False, help="Only execute get FLOPs")



    args = parser.parse_args()

    cfg_path = args.cfg_path
    ckp_path = args.ckp_path
    data_type = args.data_type
    txt_path = args.txt_path
    out_path = f'submit/results/{data_type}'

    if args.get_FLOPs:
        get_FLOPs(cfg_path, ckp_path)
        exit()

    assert (data_type == 'synthetic' or
            data_type == 'iPhone_static' or
            data_type == 'iPhone_dynamic' or
            data_type == 'modified_phone_static')
    evaluate(cfg_path, ckp_path, txt_path, out_path, visualization=args.vis)
